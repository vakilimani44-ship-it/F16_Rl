"""
Train a PPO agent (Stable-Baselines3) on the JSBSim F-16 target-reaching task.

    python train.py --timesteps 10_000_000 --n-envs 4

A curriculum starts with targets close to the initial state and widens them
(altitude change, heading change, speed change) towards the full envelope
10 000-45 000 ft / 0-360 deg / 600-1 200 ft/s whenever the agent reaches
~75 % of its targets.

Outputs in --run-dir (default runs/f16_ppo):
    best_model.zip      best deterministic score on the full-envelope eval
    final_model.zip     last weights
    ckpt_*.zip          periodic checkpoints
    vecnormalize.pkl    reward-normalisation stats (only needed to resume)
    tb/                 TensorBoard logs   (tensorboard --logdir runs)
"""
from __future__ import annotations

import argparse
import collections
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from f16_env import F16Env


def make_env(rank: int, seed: int, difficulty: float, corner_prob: float = 0.0):
    def _init():
        env = Monitor(F16Env(difficulty=difficulty, corner_prob=corner_prob))
        env.reset(seed=seed + rank)
        return env
    return _init


def corner_eval(model):
    """Fraction of the 32 extreme-corner maneuvers (see evaluate.py) that the policy completes."""
    from evaluate import corner_cases
    env = F16Env(difficulty=1.0, max_chain=0, retarget_prob=0.0)
    cases = corner_cases()
    ok = 0
    for _, (alt, spd, hdg), (talt, thdg, tspd) in cases:
        obs, _ = env.reset(seed=0, options={"init": {"alt_ft": alt, "speed_fps": spd, "heading_deg": hdg},
                                            "target": {"alt_ft": talt, "heading_deg": thdg, "speed_fps": tspd}})
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, te, tr, info = env.step(act)
            done = te or tr
        ok += info["targets_reached"] >= 1 and not info["crash"]
    return ok / len(cases)


def quick_eval(model, n_episodes: int = 20, difficulty: float = 1.0, seed: int = 10_000):
    """Deterministic rollouts on single-target tasks across the full envelope."""
    env = F16Env(difficulty=difficulty, max_chain=0, retarget_prob=0.0)
    ok, times, peaks, crashes = 0, [], [], 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, te, tr, info = env.step(act)
            done = te or tr
        reached = info["targets_reached"] >= 1 and not info["crash"]
        ok += reached
        crashes += info["crash"]
        if reached:
            times.append(info["time"])
        peaks.append(info["peak_g"])
    return {
        "success": ok / n_episodes,
        "crash": crashes / n_episodes,
        "mean_time": float(np.mean(times)) if times else float("nan"),
        "mean_peak_g": float(np.mean(peaks)),
    }


class CurriculumCallback(BaseCallback):
    def __init__(self, run_dir, start_difficulty=0.1, step=0.1, threshold=0.75, window=100,
                 eval_freq=250_000, eval_episodes=20, verbose=1):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.difficulty = start_difficulty
        self.step_size = step
        self.threshold = threshold
        self.window = collections.deque(maxlen=window)
        self.window_size = window
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.best = -1.0
        self.next_eval = eval_freq
        self.t0 = time.time()

    def _on_training_start(self):
        # when resuming, num_timesteps does not start at 0
        self.next_eval = (self.num_timesteps // self.eval_freq + 1) * self.eval_freq
        self.training_env.env_method("set_difficulty", self.difficulty)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            self.window.append(info["targets_reached"] / max(1, info["targets_started"]))
            if (len(self.window) == self.window_size and np.mean(self.window) >= self.threshold
                    and self.difficulty < 1.0):
                self.difficulty = min(1.0, round(self.difficulty + self.step_size, 3))
                self.training_env.env_method("set_difficulty", self.difficulty)
                self.window.clear()
                if self.verbose:
                    print(f"[curriculum] step {self.num_timesteps:,}: difficulty -> {self.difficulty:.2f}", flush=True)
        if len(self.window) > 10:
            self.logger.record("curriculum/difficulty", self.difficulty)
            self.logger.record("curriculum/target_success", float(np.mean(self.window)))

        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_freq
            res = quick_eval(self.model, self.eval_episodes)
            res["corner_success"] = corner_eval(self.model)
            for k, v in res.items():
                self.logger.record(f"eval_full_envelope/{k}", v)
            mt = res["mean_time"] if res["mean_time"] == res["mean_time"] else 200.0   # nan -> bad
            score = (res["success"] + res["corner_success"] - 0.5 * res["crash"]
                     - 0.002 * mt)                                          # success first, then speed
            if self.verbose:
                print(f"[eval] step {self.num_timesteps:,} ({(time.time() - self.t0) / 60:.1f} min) "
                      f"difficulty {self.difficulty:.2f} | full-envelope: success {res['success']:.0%} "
                      f"crash {res['crash']:.0%} time {res['mean_time']:.0f}s peakG {res['mean_peak_g']:.1f} | "
                      f"corners {res['corner_success']:.0%}", flush=True)
            if score > self.best:
                self.best = score
                self.model.save(os.path.join(self.run_dir, "best_model"))
            self.model.save(os.path.join(self.run_dir, f"ckpt_{self.num_timesteps // 1000}k"))
        return True


def linear_schedule(start: float, end: float):
    return lambda progress_remaining: end + (start - end) * progress_remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--run-dir", default="runs/f16_ppo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="path to a .zip model to continue from")
    ap.add_argument("--start-difficulty", type=float, default=0.1)
    ap.add_argument("--eval-freq", type=int, default=250_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--corner-prob", type=float, default=0.0,
                    help="share of episodes that start/end at the extreme corners of the envelope")
    ap.add_argument("--lr-start", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=5e-5)
    ap.add_argument("--reinit-std", type=float, default=None,
                    help="on --resume, reset the policy's action noise (std) to this value to re-open exploration")
    ap.add_argument("--dummy", action="store_true", help="single-process envs (debugging / Windows)")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    torch.set_num_threads(2)

    vec_cls = DummyVecEnv if args.dummy else SubprocVecEnv
    venv = vec_cls([make_env(i, args.seed, args.start_difficulty, args.corner_prob) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995)

    try:                                   # TensorBoard is optional
        import tensorboard  # noqa: F401
        tb_dir = os.path.join(args.run_dir, "tb")
    except ImportError:
        tb_dir = None

    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu")
        model.ent_coef = args.ent_coef
        model.learning_rate = linear_schedule(args.lr_start, args.lr_end)
        model._setup_lr_schedule()
        if args.reinit_std:
            model.policy.log_std.data.fill_(float(np.log(args.reinit_std)))
    else:
        model = PPO(
            "MlpPolicy", venv,
            learning_rate=linear_schedule(args.lr_start, args.lr_end),
            n_steps=1024, batch_size=1024, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
            tensorboard_log=tb_dir,
            seed=args.seed, device="cpu", verbose=1,
        )

    cb = CurriculumCallback(args.run_dir, start_difficulty=args.start_difficulty, eval_freq=args.eval_freq, eval_episodes=args.eval_episodes)
    try:
        model.learn(total_timesteps=args.timesteps, callback=cb, reset_num_timesteps=not args.resume)
    finally:
        model.save(os.path.join(args.run_dir, "final_model"))
        venv.save(os.path.join(args.run_dir, "vecnormalize.pkl"))
        venv.close()


if __name__ == "__main__":
    main()
