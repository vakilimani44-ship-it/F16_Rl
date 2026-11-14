"""
Web UI backend: runs the F-16 simulation with the trained RL agent in real
time and streams the state to the browser over a WebSocket.

    python server.py --model runs/f16_ppo/best_model.zip
    then open http://127.0.0.1:8000

Without --model the scripted pilot flies (handy for trying the UI).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import webbrowser

import numpy as np
from aiohttp import WSMsgType, web

from agent import load_agent
from f16_env import ALT_MAX, ALT_MIN, SPD_MAX, SPD_MIN, F16Env, vmax_for_alt

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


def _round(o, nd=3):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


class Sim:
    def __init__(self, agent, agent_name):
        self.agent = agent
        self.agent_name = agent_name
        self.env = F16Env(terminate_on_success=False, max_chain=0, retarget_prob=0.0)
        self.clients: set[web.WebSocketResponse] = set()
        self.speed = 1.0
        self.paused = False
        self.obs = None
        self.crashes = 0
        self._pending_reset = None
        self.reset_flight({"alt_ft": 25_000.0, "speed_fps": 800.0, "heading_deg": 0.0}, None)

    # ---- control ------------------------------------------------------
    def reset_flight(self, init, target):
        tgt = target or {"alt_ft": init["alt_ft"], "heading_deg": init["heading_deg"], "speed_fps": init["speed_fps"]}
        self.obs, _ = self.env.reset(options={"init": init, "target": tgt})
        self.env.terminate_on_success = False

    def random_scenario(self):
        rng = np.random.default_rng()
        self.env.np_random = rng
        alt, vt, psi = self.env._sample_init(None)
        init = {"alt_ft": alt, "speed_fps": vt, "heading_deg": psi}
        self.env.set_difficulty(1.0)
        tgt = self.env._sample_target(alt, psi, vt)
        self.reset_flight(init, {"alt_ft": float(tgt[0]), "heading_deg": float(tgt[1]), "speed_fps": float(tgt[2])})

    def set_target(self, alt, hdg, spd):
        alt = float(np.clip(alt, ALT_MIN, ALT_MAX))
        spd = float(np.clip(spd, SPD_MIN, SPD_MAX))
        self.env.set_target(alt, hdg % 360.0, spd)
        self.obs = self.env._obs()

    # ---- messaging ----------------------------------------------------
    async def broadcast(self, msg):
        if not self.clients:
            return
        text = json.dumps(_round(msg))
        await asyncio.gather(*(c.send_str(text) for c in list(self.clients)), return_exceptions=True)

    def hello(self):
        return {
            "type": "hello",
            "agent": self.agent_name,
            "limits": {"alt": [ALT_MIN, ALT_MAX], "speed": [SPD_MIN, SPD_MAX]},
            "vmax_table": [[a, vmax_for_alt(a)] for a in range(int(ALT_MIN), int(ALT_MAX) + 1, 2500)],
            "tolerance": {"alt_ft": self.env._tol_alt, "heading_deg": self.env._tol_hdg, "speed_fps": self.env._tol_spd},
            "speed_mult": self.speed, "paused": self.paused,
            "state": self.env.get_state(),
        }

    async def handle(self, data):
        kind = data.get("type")
        if kind == "target":
            self.set_target(data["alt_ft"], data["heading_deg"], data["speed_fps"])
        elif kind == "reset":
            mode = data.get("mode", "level")
            if mode == "random":
                self.random_scenario()
            else:
                self.reset_flight({"alt_ft": 25_000.0, "speed_fps": 800.0, "heading_deg": 0.0}, None)
            await self.broadcast({"type": "reset"})
        elif kind == "pause":
            self.paused = bool(data.get("value"))
        elif kind == "speed":
            self.speed = float(np.clip(data.get("value", 1.0), 0.1, 8.0))

    async def run(self):
        next_t = time.perf_counter()
        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                next_t = time.perf_counter()
                continue
            act, _ = self.agent.predict(self.obs, deterministic=True)
            self.obs, _, terminated, _, info = self.env.step(act)
            st = self.env.get_state()
            st["type"] = "state"
            st["action"] = [float(x) for x in np.asarray(act).ravel()]
            st["speed_mult"] = self.speed
            await self.broadcast(st)
            if terminated:
                self.crashes += 1
                await self.broadcast({"type": "event", "kind": "crash", "t": st["t"], "alt_ft": st["alt_ft"]})
                await asyncio.sleep(1.5)
                self.reset_flight({"alt_ft": 25_000.0, "speed_fps": 800.0, "heading_deg": 0.0}, None)
                await self.broadcast({"type": "reset"})
                next_t = time.perf_counter()
                continue
            next_t += self.env.dt / self.speed
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
                if delay < -0.5:                       # fell behind (heavy speed-up): do not spiral
                    next_t = time.perf_counter()


async def ws_handler(request):
    sim: Sim = request.app["sim"]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    sim.clients.add(ws)
    await ws.send_str(json.dumps(_round(sim.hello())))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    await sim.handle(json.loads(msg.data))
                except (KeyError, ValueError, TypeError) as exc:
                    await ws.send_str(json.dumps({"type": "error", "message": f"bad message: {exc}"}))
    finally:
        sim.clients.discard(ws)
    return ws


async def index(request):
    return web.FileResponse(os.path.join(UI_DIR, "index.html"))


async def on_startup(app):
    app["sim_task"] = asyncio.create_task(app["sim"].run())


async def on_cleanup(app):
    app["sim_task"].cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="trained .zip (omit to use the scripted pilot)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args()

    agent, name = load_agent(args.model)
    app = web.Application()
    app["sim"] = Sim(agent, name)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", index)
    app.router.add_static("/", UI_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    url = f"http://{args.host}:{args.port}"
    print(f"Agent: {name}\nUI:    {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
