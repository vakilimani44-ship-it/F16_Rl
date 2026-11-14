"""
Evaluate an agent on the full envelope (10 000-45 000 ft, 0-360 deg, 600-1 200 ft/s).

    python evaluate.py --model runs/f16_ppo/best_model.zip --episodes 200
    python evaluate.py --model heuristic                      # scripted-pilot baseline

An episode succeeds when altitude, heading and speed are all inside the
tolerance band (100 ft, 2.5 deg, 15 ft/s) for 2 continuous seconds.
Also runs a fixed table of hard, named maneuvers.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from agent import load_agent
from f16_env import F16Env

HARD_CASES = [
    # name, init (alt, speed, heading), target (alt, heading, speed)
    ("Max climb 10k -> 45k ft",         (10_000, 800, 0),   (45_000, 0, 800)),
    ("Max dive 45k -> 10k ft",          (45_000, 800, 0),   (10_000, 0, 800)),
    ("Reverse course (180 deg turn)",   (25_000, 800, 0),   (25_000, 180, 800)),
    ("Slow down 1100 -> 600 ft/s",      (25_000, 1100, 90), (25_000, 90, 600)),
    ("Speed up 600 -> 1100 ft/s",       (25_000, 600, 90),  (25_000, 90, 1100)),
    ("Climb + reverse + slow",          (15_000, 1000, 0),  (40_000, 180, 650)),
    ("Dive + 150 deg turn + fast",      (42_000, 650, 270), (12_000, 60, 1100)),
    ("Small correction (+500 ft, +10 deg)", (30_000, 900, 45), (30_500, 55, 900)),
]


def corner_cases():
    """Every combination of extreme start / target altitude and speed, with and without a 180 deg turn."""
    from f16_env import vmax_for_alt
    cases = []
    for a0 in (10_000, 45_000):
        for v0 in (600, 1_100):
            for a1 in (10_000, 45_000):
                for v1 in (600, 1_100):
                    for dpsi in (0, 180):
                        if v0 > vmax_for_alt(a0) or v1 > vmax_for_alt(a1):
                            continue
                        cases.append((f"{a0//1000}k/{v0} -> {a1//1000}k/{v1}" + (" +180deg" if dpsi else ""),
                                      (a0, v0, 0), (a1, dpsi, v1)))
    return cases


def rollout(env, agent, options=None, seed=None):
    obs, info = env.reset(seed=seed, options=options)
    done = False
    while not done:
        act, _ = agent.predict(obs, deterministic=True)
        obs, _, te, tr, info = env.step(act)
        done = te or tr
    ok = info["targets_reached"] >= 1 and not info["crash"]
    return ok, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/f16_ppo/best_model.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=50_000)
    ap.add_argument("--json", default=None, help="write results to this file")
    ap.add_argument("--timeout", type=float, default=None, help="per-target time limit in s (default 180)")
    ap.add_argument("--corners-only", action="store_true", help="only run the extreme-corner suite")
    args = ap.parse_args()

    agent, name = load_agent(args.model)
    env = F16Env(difficulty=1.0, max_chain=0, retarget_prob=0.0, timeout_s=args.timeout)

    ok_n, crash_n, times, peaks, mins, finals = 0, 0, [], [], [], []
    if args.corners_only:
        args.episodes = 0
    for ep in range(args.episodes):
        ok, info = rollout(env, agent, seed=args.seed + ep)
        ok_n += ok
        crash_n += info["crash"]
        peaks.append(info["peak_g"]); mins.append(info["min_g"])
        if ok:
            times.append(info["time"])
        else:
            finals.append((abs(info["err_alt"]), abs(info["err_hdg"]), abs(info["err_spd"])))

    n = max(1, args.episodes)
    res = {
        "agent": name, "episodes": args.episodes,
        "success_rate": ok_n / n,
        "crash_rate": crash_n / n,
        "mean_time_to_target_s": float(np.mean(times)) if times else None,
        "median_time_to_target_s": float(np.median(times)) if times else None,
        "mean_peak_g": float(np.mean(peaks)) if peaks else None,
        "max_peak_g": float(np.max(peaks)) if peaks else None,
        "min_g_seen": float(np.min(mins)) if mins else None,
    }
    print(f"\nAgent: {name}")
    if args.episodes:
        print(f"Random full-envelope targets ({args.episodes} episodes)")
        print(f"  success      {res['success_rate']:.1%}")
        print(f"  crash        {res['crash_rate']:.1%}")
    if times:
        print(f"  time to hit  mean {res['mean_time_to_target_s']:.1f} s, median {res['median_time_to_target_s']:.1f} s")
    if args.episodes:
        print(f"  load factor  mean peak {res['mean_peak_g']:.1f} G, max {res['max_peak_g']:.1f} G, lowest {res['min_g_seen']:.1f} G")
    if finals:
        f = np.array(finals)
        print(f"  failures     median residual error: {np.median(f[:, 0]):.0f} ft, {np.median(f[:, 1]):.1f} deg, {np.median(f[:, 2]):.0f} ft/s")

    print("\nHard maneuvers")
    print(f"  {'case':<38}{'result':<10}{'time':>7}{'peak G':>8}")
    res["hard_cases"] = []
    for name_c, (alt, spd, hdg), (talt, thdg, tspd) in HARD_CASES:
        ok, info = rollout(env, agent, options={
            "init": {"alt_ft": alt, "speed_fps": spd, "heading_deg": hdg},
            "target": {"alt_ft": talt, "heading_deg": thdg, "speed_fps": tspd}})
        t = f"{info['time']:.0f} s" if ok else "-"
        print(f"  {name_c:<38}{'reached' if ok else ('crashed' if info['crash'] else 'timeout'):<10}{t:>7}{info['peak_g']:>8.1f}")
        res["hard_cases"].append({"case": name_c, "reached": bool(ok), "time_s": info["time"] if ok else None, "peak_g": info["peak_g"]})

    print("\nExtreme corners of the envelope")
    corner = corner_cases()
    n_ok = 0
    res["corners"] = []
    for name_c, (alt, spd, hdg), (talt, thdg, tspd) in corner:
        ok, info = rollout(env, agent, options={
            "init": {"alt_ft": alt, "speed_fps": spd, "heading_deg": hdg},
            "target": {"alt_ft": talt, "heading_deg": thdg, "speed_fps": tspd}})
        n_ok += ok
        res["corners"].append({"case": name_c, "reached": bool(ok), "time_s": info["time"] if ok else None, "peak_g": info["peak_g"], "crash": bool(info["crash"])})
        print(f"  {name_c:<34}{'reached' if ok else ('crashed' if info['crash'] else 'timeout'):<9}{(str(int(info['time'])) + ' s') if ok else '-':>7}{info['peak_g']:>7.1f} G")
    res["corner_success"] = n_ok / max(1, len(corner))
    print(f"  corner success: {n_ok}/{len(corner)}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
