"""
Loading a trained agent, plus a simple hand-written pilot.

The scripted pilot is not part of the RL solution. It exists so that the UI
can be tried before a model is trained and so the learned policy has a
baseline to be compared against (see evaluate.py --baseline).
"""
from __future__ import annotations

import math

import numpy as np


class HeuristicPilot:
    """Crude PD autopilot working on the same observation vector as the RL agent."""

    def predict(self, obs, deterministic=True):
        dh = obs[0] * 10_000.0                      # ft to climb
        dpsi = obs[2] * 180.0                       # deg to turn (signed)
        dv = obs[6] * 300.0                         # ft/s to gain
        phi = math.atan2(obs[10], obs[11])
        gamma = obs[13] * 90.0                      # deg
        nz = obs[20] * 9.0

        bank_des = max(-75.0, min(75.0, dpsi * 1.5))
        if abs(dpsi) < 3.0:
            bank_des = dpsi * 4.0
        roll = max(-1.0, min(1.0, (bank_des - math.degrees(phi)) * 0.05))

        gamma_des = max(-25.0, min(25.0, dh * 0.006))
        cosphi = max(0.35, math.cos(phi))
        nz_des = 1.0 / cosphi + (gamma_des - gamma) * 0.35
        pitch = max(-1.0, min(1.0, (nz_des - 1.0) / 8.0))
        pitch -= 0.01 * (nz - nz_des)

        throttle = max(-1.0, min(1.0, 0.0 + dv / 120.0))
        brake = 1.0 if dv < -150.0 else -1.0
        return np.array([pitch, roll, 0.0, throttle, brake], dtype=np.float32), None


def load_agent(path: str | None):
    """path: a Stable-Baselines3 .zip, or None / 'heuristic' for the scripted pilot."""
    if path in (None, "", "heuristic"):
        return HeuristicPilot(), "scripted pilot"
    from stable_baselines3 import PPO
    # The saved model contains a pickled learning-rate lambda that only matters for training.
    # Its bytecode is tied to the Python version that trained it (and crashes other versions),
    # so replace it with harmless constants when just running the agent.
    custom = {"learning_rate": 0.0, "lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0}
    return PPO.load(path, device="cpu", custom_objects=custom), path
