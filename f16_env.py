"""
F-16 (JSBSim) goal-conditioned Gymnasium environment.

The agent flies the JSBSim F-16 and must reach a commanded
(target altitude, target heading, target true airspeed) as fast as it can.
Reaching a far-away target quickly means pulling high G, so the reward
combines dense progress shaping with a time pressure and a "reached early"
bonus.

Operating envelope (as requested)
    altitude : 10 000 .. 45 000 ft
    heading  : 0 .. 360 deg
    speed    : 600 .. 1 200 ft/s   (true airspeed)

Action space  (Box[-1, 1]^5)
    0  pitch stick   +1 = full back stick (flight computer limits to ~9 G),
                     -1 = full forward stick (limited to ~-4 G)
    1  roll stick    +1 = roll right
    2  rudder
    3  throttle      -1 = idle ... +1 = full
    4  speed brake   > 0 = deployed

Observation space  (Box, 27 dims) - see _obs()
"""
from __future__ import annotations

import math
import os

import gymnasium as gym
import jsbsim
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ALT_MIN, ALT_MAX = 10_000.0, 45_000.0      # ft MSL
SPD_MIN, SPD_MAX = 600.0, 1_200.0          # ft/s true airspeed
R_EARTH_FT = 20_925_646.0
SIM_HZ = 120                               # JSBSim f16 model integrates at 120 Hz
G0 = 32.174

# Crash / departure limits (wider than the operating envelope on purpose)
ALT_FLOOR = 5_000.0
ALT_CEIL = 55_000.0
SPD_STALL = 300.0
SPD_OVER = 1_500.0

OBS_DIM = 27
ACT_DIM = 5


def vmax_for_alt(alt_ft: float) -> float:
    """Highest speed the F-16 model can hold in level flight with some thrust
    margin (measured with JSBSim trim). Below ~18 000 ft the engine, not the
    1 200 ft/s envelope limit, is what stops us."""
    return float(np.interp(alt_ft, [10_000, 14_000, 18_000, 45_000],
                           [1_130, 1_160, 1_200, 1_200]))


def wrap180(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def _clip(x: float, lo: float, hi: float) -> float:
    """Scalar clip - ~50x faster than np.clip on a python float."""
    return lo if x < lo else hi if x > hi else x


class F16Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty: float = 1.0,
        agent_hz: int = 10,
        terminate_on_success: bool = True,
        max_chain: int = 2,
        retarget_prob: float = 0.3,
        tol_alt: float = 100.0,
        tol_hdg: float = 2.5,
        tol_spd: float = 15.0,
        hold_s: float = 2.0,
        max_episode_s: float = 600.0,
        time_penalty: float = 0.05,
        crash_penalty: float = 150.0,
        corner_prob: float = 0.0,
        timeout_s: float | None = None,
    ):
        super().__init__()
        self.difficulty = float(difficulty)
        self.agent_hz = agent_hz
        self.substeps = SIM_HZ // agent_hz
        self.dt = 1.0 / agent_hz
        self.terminate_on_success = terminate_on_success
        self.max_chain = max_chain
        self.retarget_prob = retarget_prob
        self._tol_alt, self._tol_hdg, self._tol_spd = float(tol_alt), float(tol_hdg), float(tol_spd)
        self.hold_steps_needed = int(round(hold_s * agent_hz))
        self.max_episode_s = max_episode_s
        self.time_penalty = float(time_penalty)
        self.timeout_s = timeout_s              # per-target time limit; None -> 30 s + 150 s * difficulty
        self.corner_prob = float(corner_prob)   # share of training episodes drawn from the extreme corners
        self._corner = False
        self.crash_penalty = float(crash_penalty)

        root = os.path.dirname(jsbsim.__file__)
        self.fdm = jsbsim.FGFDMExec(root)
        self.fdm.set_debug_level(0)
        self.fdm.load_model("f16")

        self.observation_space = spaces.Box(-10.0, 10.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

        self.target = np.array([20_000.0, 0.0, 800.0])   # alt, hdg, speed
        self._prev_action = np.zeros(ACT_DIM)
        self._reset_counters()

    # ------------------------------------------------------------------
    # public helpers (used by the UI server / evaluation)
    # ------------------------------------------------------------------
    def set_difficulty(self, d: float) -> None:
        self.difficulty = float(np.clip(d, 0.02, 1.0))

    def set_target(self, alt_ft: float, hdg_deg: float, spd_fps: float) -> None:
        self.target = np.array([float(alt_ft), float(hdg_deg) % 360.0, float(spd_fps)])
        self._hold = 0
        self._target_t0 = self.t
        self._prev_pot = self._potential()

    def target_is_feasible(self) -> bool:
        return bool(self.target[2] <= vmax_for_alt(self.target[0]) + 1.0)

    def get_state(self) -> dict:
        """Everything the UI needs, in plain python types."""
        f = self.fdm
        lat = math.radians(f["position/lat-gc-deg"])
        lon = math.radians(f["position/long-gc-deg"])
        dh, dpsi, dv = self._errors()
        return {
            "t": self.t,
            "north_ft": lat * R_EARTH_FT,
            "east_ft": lon * R_EARTH_FT * math.cos(lat),
            "alt_ft": f["position/h-sl-ft"],
            "roll_deg": f["attitude/phi-deg"],
            "pitch_deg": f["attitude/theta-deg"],
            "heading_deg": f["attitude/psi-deg"] % 360.0,
            "vt_fps": f["velocities/vt-fps"],
            "mach": f["velocities/mach"],
            "alpha_deg": f["aero/alpha-deg"],
            "beta_deg": f["aero/beta-deg"],
            "nz_g": f["accelerations/Nz"],
            "climb_fps": f["velocities/h-dot-fps"],
            "gamma_deg": math.degrees(f["flight-path/gamma-rad"]),
            "p_dps": math.degrees(f["velocities/p-rad_sec"]),
            "q_dps": math.degrees(f["velocities/q-rad_sec"]),
            "r_dps": math.degrees(f["velocities/r-rad_sec"]),
            "throttle": float(f["fcs/throttle-cmd-norm"]),
            "thrust_lbs": f["propulsion/engine/thrust-lbs"],
            "speedbrake": float(f["fcs/speedbrake-pos-norm"]),
            "elevator_cmd": f["fcs/elevator-cmd-norm"],
            "aileron_cmd": f["fcs/aileron-cmd-norm"],
            "rudder_cmd": f["fcs/rudder-cmd-norm"],
            "target": {"alt_ft": float(self.target[0]), "heading_deg": float(self.target[1]),
                       "speed_fps": float(self.target[2])},
            "err": {"alt_ft": dh, "heading_deg": dpsi, "speed_fps": dv},
            "on_target": bool(self._within(1.0)),
            "peak_g": self._peak_g, "min_g": self._min_g,
            "feasible": self.target_is_feasible(),
        }

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        init = options.get("init")
        for _ in range(30):
            alt, vt, psi = self._sample_init(init)
            if self._apply_ic(alt, vt, psi):
                break
        else:                                   # extremely unlikely
            alt, vt, psi = 25_000.0, 800.0, 0.0
            self._apply_ic(alt, vt, psi)

        self._reset_counters()
        tgt = options.get("target")
        if tgt is not None:
            self.target = np.array([tgt["alt_ft"], tgt["heading_deg"] % 360.0, tgt["speed_fps"]], dtype=np.float64)
        else:
            self.target = self._sample_target(alt, psi, vt, corner=self._corner)
        self._n_chain = int(self.np_random.integers(0, self.max_chain + 1)) if self.terminate_on_success else 0
        self._retarget_at = None
        if self.terminate_on_success and self.np_random.random() < self.retarget_prob:
            self._retarget_at = float(self.np_random.uniform(5.0, 40.0))
        self._targets_started = 1
        self._timeout_s = float(self.timeout_s) if self.timeout_s else 30.0 + 150.0 * self.difficulty
        self._prev_action = np.array([0, 0, 0, 2 * float(self.fdm["fcs/throttle-cmd-norm"]) - 1, -1.0])
        self._prev_pot = self._potential()
        return self._obs(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._apply_action(a)
        f = self.fdm
        for _ in range(self.substeps):
            f.run()
            nz = f["accelerations/Nz"]
            if nz > self._peak_g:
                self._peak_g = nz
            if nz < self._min_g:
                self._min_g = nz
        self.t += self.dt
        self.steps += 1

        alt = f["position/h-sl-ft"]
        vt = f["velocities/vt-fps"]
        terminated = truncated = False
        crashed = False
        reward = -self.time_penalty                              # time pressure -> pull G, arrive sooner
        if not (math.isfinite(alt) and math.isfinite(vt)):
            return np.zeros(OBS_DIM, np.float32), -self.crash_penalty, True, False, self._info(crash=True)

        err = self._errors()

        # --- dense progress shaping --------------------------------------
        pot = self._potential(err)
        reward += pot - self._prev_pot
        self._prev_pot = pot

        # --- closeness / hold bonus --------------------------------------
        if self._within(3.0, err):
            reward += 0.05
        if self._within(1.0, err):
            reward += 0.15
            self._hold += 1
        else:
            self._hold = 0

        # --- smoothness / stability penalties ----------------------------
        da = a - self._prev_action
        reward -= 0.01 * float(np.sum((da / 2.0) ** 2))
        alpha = f["aero/alpha-deg"]
        if abs(alpha) > 25.0:
            reward -= 0.05
        # early warnings before the hard limits, so the agent does not have to
        # learn them from the crash penalty alone
        if alt < ALT_MIN - 500.0:
            reward -= 0.1 * (ALT_MIN - 500.0 - alt) / 1000.0
        if vt < 450.0:
            reward -= 0.1 * (450.0 - vt) / 100.0
        reward -= 0.002 * min(abs(f["aero/beta-deg"]), 10.0)
        self._prev_action = a

        # --- target reached ------------------------------------------------
        success = False
        if self._hold >= self.hold_steps_needed:
            success = True
            self._targets_reached += 1
            frac_left = max(0.0, 1.0 - (self.t - self._target_t0) / self._timeout_s)
            reward += 20.0 + 20.0 * frac_left
            if self.terminate_on_success:
                if self._n_chain > 0:
                    self._n_chain -= 1
                    self._new_target_from_state()
                    self._targets_started += 1
                    err = self._errors()
                else:
                    terminated = True

        # --- scheduled mid-maneuver retarget ----------------------------
        if self._retarget_at is not None and self.t >= self._retarget_at and not terminated:
            self._retarget_at = None
            self._new_target_from_state()
            self._targets_started += 1
            err = self._errors()

        # --- failure conditions ----------------------------------------
        if alt < ALT_FLOOR or alt > ALT_CEIL or vt < SPD_STALL or vt > SPD_OVER:
            crashed = terminated = True
            reward -= self.crash_penalty
        elif self.terminate_on_success:
            if (self.t - self._target_t0) > self._timeout_s or self.t > self.max_episode_s:
                truncated = True

        return (self._obs(err), round(float(reward), 7), terminated, truncated,
                self._info(crash=crashed, success=success, err=err))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _reset_counters(self):
        self.t = 0.0
        self.steps = 0
        self._hold = 0
        self._target_t0 = 0.0
        self._targets_reached = 0
        self._targets_started = 1
        self._peak_g = 1.0
        self._min_g = 1.0
        self._n_chain = 0
        self._retarget_at = None
        self._timeout_s = 180.0
        self._prev_pot = 0.0

    def _apply_ic(self, alt, vt, psi) -> bool:
        f = self.fdm
        f["ic/h-sl-ft"] = alt
        f["ic/vt-fps"] = vt
        f["ic/psi-true-deg"] = psi
        f["ic/theta-deg"] = 0.0
        f["ic/phi-deg"] = 0.0
        f["ic/gamma-deg"] = 0.0
        f["ic/lat-gc-deg"] = 0.0
        f["ic/long-gc-deg"] = 0.0
        for p in ("fcs/elevator-cmd-norm", "fcs/aileron-cmd-norm", "fcs/rudder-cmd-norm",
                  "fcs/speedbrake-cmd-norm", "fcs/pitch-trim-cmd-norm"):
            f[p] = 0.0
        # reset_to_initial_conditions (unlike run_ic) also clears the flight
        # computer's PID integrators and engine state -> reproducible resets.
        f.reset_to_initial_conditions(0)
        f["propulsion/set-running"] = -1
        f["fcs/throttle-cmd-norm"] = 0.7
        try:
            f["simulation/do_simple_trim"] = 1
        except Exception:                        # jsbsim.TrimFailureError: pick another start
            return False
        thr = f["fcs/throttle-cmd-norm"]
        for _ in range(SIM_HZ):                  # let the flight computer settle
            f.run()
        return bool(
            math.isfinite(thr) and thr < 0.98
            and abs(f["position/h-sl-ft"] - alt) < 100.0
            and abs(f["velocities/vt-fps"] - vt) < 25.0
        )

    def _apply_action(self, a):
        f = self.fdm
        pitch = a[0]
        f["fcs/elevator-cmd-norm"] = -pitch if pitch >= 0.0 else -0.44 * pitch
        f["fcs/aileron-cmd-norm"] = a[1]
        f["fcs/rudder-cmd-norm"] = a[2]
        f["fcs/throttle-cmd-norm"] = 0.5 * (a[3] + 1.0)
        f["fcs/speedbrake-cmd-norm"] = 1.0 if a[4] > 0.0 else 0.0

    def _errors(self):
        f = self.fdm
        dh = self.target[0] - f["position/h-sl-ft"]
        dpsi = wrap180(self.target[1] - f["attitude/psi-deg"])
        dv = self.target[2] - f["velocities/vt-fps"]
        return dh, dpsi, dv

    def _within(self, k: float, err=None) -> bool:
        dh, dpsi, dv = err if err is not None else self._errors()
        return abs(dh) < k * self._tol_alt and abs(dpsi) < k * self._tol_hdg and abs(dv) < k * self._tol_spd

    def _potential(self, err=None) -> float:
        dh, dpsi, dv = err if err is not None else self._errors()
        dh, dpsi, dv = abs(dh), abs(dpsi), abs(dv)
        coarse = dh / 10_000.0 + dpsi / 180.0 + 0.7 * dv / 300.0
        fine = 0.4 * min(dh / 500.0, 1.0) + 0.4 * min(dpsi / 10.0, 1.0) + 0.3 * min(dv / 30.0, 1.0)
        return -(10.0 * coarse + 3.0 * fine)

    # ---- sampling -----------------------------------------------------
    def _sample_init(self, init):
        self._corner = False
        if init is not None:
            return init["alt_ft"], init["speed_fps"], init["heading_deg"]
        rng = self.np_random
        if self.corner_prob > 0.0 and rng.random() < self.corner_prob:
            # extreme start: very low or very high altitude, very slow or very fast
            self._corner = True
            alt = rng.uniform(10_000, 13_000) if rng.random() < 0.5 else rng.uniform(42_000, 45_000)
            vt = rng.uniform(600, 700) if rng.random() < 0.5 else rng.uniform(1_000, 1_150)
            vt = min(vt, vmax_for_alt(alt) - 15.0)
            return float(alt), float(vt), float(rng.uniform(0.0, 360.0))
        alt = rng.uniform(ALT_MIN, ALT_MAX)
        vt = rng.uniform(SPD_MIN, min(SPD_MAX, vmax_for_alt(alt)))
        psi = rng.uniform(0.0, 360.0)
        return float(alt), float(vt), float(psi)

    def _sample_target(self, alt0, psi0, v0, corner=False):
        rng = self.np_random
        d = self.difficulty
        if corner:
            # extreme target, usually on the opposite side of the envelope
            far_alt = alt0 < 27_500.0
            if rng.random() < 0.8:
                alt = rng.uniform(42_000, 45_000) if far_alt else rng.uniform(10_000, 13_000)
            else:
                alt = rng.uniform(42_000, 45_000) if rng.random() < 0.5 else rng.uniform(10_000, 13_000)
            slow = v0 < 850.0
            if rng.random() < 0.7:
                spd = rng.uniform(1_000, 1_150) if slow else rng.uniform(600, 700)
            else:
                spd = rng.uniform(1_000, 1_150) if rng.random() < 0.5 else rng.uniform(600, 700)
            psi = (psi0 + rng.choice([0.0, 180.0, rng.uniform(-180, 180)])) % 360.0
            return np.array([float(alt), float(psi), float(min(spd, vmax_for_alt(alt) - 15.0))])
        active = rng.random(3) < 0.75
        if not active.any():
            active[rng.integers(0, 3)] = True
        alt = alt0
        psi = psi0
        spd = v0
        if active[0]:
            alt = float(np.clip(alt0 + rng.uniform(-1, 1) * d * (ALT_MAX - ALT_MIN), ALT_MIN, ALT_MAX))
        if active[1]:
            psi = (psi0 + rng.uniform(-1, 1) * d * 180.0) % 360.0
        if active[2]:
            spd = float(np.clip(v0 + rng.uniform(-1, 1) * d * (SPD_MAX - SPD_MIN), SPD_MIN, SPD_MAX))
        spd = float(min(spd, vmax_for_alt(alt)))
        return np.array([alt, psi, spd])

    def _new_target_from_state(self):
        f = self.fdm
        self.target = self._sample_target(f["position/h-sl-ft"], f["attitude/psi-deg"], f["velocities/vt-fps"])
        self._hold = 0
        self._target_t0 = self.t
        self._prev_pot = self._potential()

    # ---- observation ----------------------------------------------------
    def _obs(self, err=None):
        f = self.fdm
        dh, dpsi, dv = err if err is not None else self._errors()
        dpsi_r = math.radians(dpsi)
        phi = math.radians(f["attitude/phi-deg"])
        theta = math.radians(f["attitude/theta-deg"])
        o = [
            _clip(dh / 10_000.0, -4.0, 4.0),
            _clip(dh / 500.0, -1.0, 1.0),
            dpsi / 180.0,
            math.sin(dpsi_r),
            math.cos(dpsi_r),
            _clip(dpsi / 10.0, -1.0, 1.0),
            _clip(dv / 300.0, -3.0, 3.0),
            _clip(dv / 30.0, -1.0, 1.0),
            f["position/h-sl-ft"] / 50_000.0,
            f["velocities/vt-fps"] / 1_200.0,
            math.sin(phi),
            math.cos(phi),
            math.sin(theta),
            f["flight-path/gamma-rad"] / (math.pi / 2),
            _clip(f["velocities/h-dot-fps"] / 300.0, -3.0, 3.0),
            f["aero/alpha-deg"] / 30.0,
            _clip(f["aero/beta-deg"] / 10.0, -3.0, 3.0),
            _clip(f["velocities/p-rad_sec"] / 3.0, -3.0, 3.0),
            _clip(f["velocities/q-rad_sec"] / 1.0, -3.0, 3.0),
            _clip(f["velocities/r-rad_sec"] / 0.5, -3.0, 3.0),
            _clip(f["accelerations/Nz"] / 9.0, -2.0, 2.0),
            f["fcs/throttle-cmd-norm"],
        ]
        o.extend(self._prev_action.tolist())
        o = np.array(o, dtype=np.float32)
        if not np.isfinite(o).all():
            np.nan_to_num(o, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
        # rounding removes ~1e-13 trim-solver noise so resets are exactly repeatable
        return np.round(np.clip(o, -10.0, 10.0), 5)

    def _info(self, crash=False, success=False, err=None):
        dh, dpsi, dv = err if err is not None else self._errors()
        return {
            "success": success, "crash": crash, "time": round(self.t, 6),
            "err_alt": round(dh, 5), "err_hdg": round(dpsi, 5), "err_spd": round(dv, 5),
            "peak_g": round(self._peak_g, 5), "min_g": round(self._min_g, 5),
            "targets_reached": self._targets_reached,
            "targets_started": self._targets_started,
        }
