# F-16 target-reaching RL agent (JSBSim + Gymnasium + Stable-Baselines3)

An RL agent flies the JSBSim **F-16** and reaches any commanded
**altitude / heading / airspeed** inside the envelope

| quantity | range |
|---|---|
| altitude | 10 000 - 45 000 ft |
| heading | 0 - 360 deg |
| true airspeed | 600 - 1 200 ft/s |

and pulls high G to get there quickly (the F-16 flight computer limits the
stick to about +9 G / -4 G; the trained agent regularly commands 6-9 G).
A 3D web console shows the aircraft and lets you set the target.

## Quick start

```bash
pip install -r requirements.txt

# 1. run the console with the trained agent  ->  http://127.0.0.1:8000
python server.py --model models/f16_agent.zip --open

# 2. numbers: success rate, time to target, peak G, hard maneuvers
python evaluate.py --model models/f16_agent.zip --episodes 100

# 3. (re)train from scratch (about 1 000 steps/s on 2 cores; 3-4 M steps gives a competent agent)
python train.py --timesteps 6000000 --n-envs 4 --run-dir runs/my_run
```

`python server.py` with no `--model` flies a simple scripted pilot, which
is useful to try the UI and as a baseline (`python evaluate.py --model heuristic`).

## Files

| file | what it is |
|---|---|
| `f16_env.py` | Gymnasium environment around JSBSim's `f16` model |
| `train.py` | PPO training with an automatic curriculum |
| `evaluate.py` | success / time / G statistics and a table of named hard maneuvers |
| `agent.py` | loads a trained model; also the scripted baseline pilot |
| `server.py` | runs the simulation in real time and streams it over a WebSocket |
| `ui/` | browser console (Three.js, vendored, no CDN needed) |
| `models/f16_agent.zip` | trained policy |
| `test_env.py` | environment sanity tests (Gymnasium `check_env`, determinism, throughput) |
| `models/` | the trained agent and the raw evaluation results |
| `runs/logs/` | training logs |

## The environment (`f16_env.py`)

**Observation (27 numbers).** Errors to the target (altitude in two scales,
heading as angle/sin/cos, speed in two scales: a coarse one for far-away
targets and a fine one for the last stretch), the aircraft's own state
(altitude, speed, bank, pitch, flight-path angle, climb rate, angle of attack,
sideslip, body rates, load factor, throttle) and the previous action.

**Action (5 numbers in [-1, 1]).** Pitch stick, roll stick, rudder, throttle,
speed brake. The F-16's built-in flight computer stays active, so the pitch
stick is a *G command* (full back stick = about 9 G) and the roll stick is a
roll-rate command. The agent decides how hard to pull; the flight computer
keeps the airplane inside its structural and angle-of-attack limits.

**Reward.**

* progress toward the target (change of a potential built from the three
  errors, so it is path-independent and cannot be gamed),
* a small constant time penalty and a bonus for arriving early. Together they
  are what makes the agent pull high G instead of cruising to the target,
* closeness / hold bonus near the target and a large success bonus,
* small penalties for jerky commands, large angle of attack and sideslip,
  and early warnings for getting too low or too slow,
* a large penalty for leaving the safe region (below 5 000 ft, above 55 000 ft,
  under 300 ft/s or over 1 500 ft/s).

**Success** = altitude within 100 ft, heading within 2.5 deg and speed within
15 ft/s, held for 2 s.

**Training episodes** start from a trimmed, level flight at a random point of
the envelope with a random target, sometimes followed by a second target or
a target change in the middle of the maneuver, so the agent can also handle
retargeting from the UI at any moment.

**Engine limits.** The F-16 model cannot hold every speed at every altitude:
below about 18 000 ft the engine, not the 1 200 ft/s limit, caps level-flight
speed (about 1 130 ft/s at 10 000 ft). The environment and the UI clip
targets to what the model can sustain and the UI warns when a requested
combination is beyond it.

## Training (`train.py`)

PPO (Stable-Baselines3), 256x256 MLP policy/value networks, 10 Hz control,
JSBSim at 120 Hz. A curriculum starts with targets close to the initial state
and widens the altitude / heading / speed changes toward the full envelope
whenever the agent reaches about 75 % of its targets.

The reported agent (7.5 M steps in total) was trained in stages, continuing the same
network each time (logs in `runs/logs/train_*.log`):

1. about 0.9 M steps of curriculum learning with a weak time penalty,
2. about 2 M more at full difficulty: the policy became safe but gentle, turning at
   about 1.4 G and taking over a minute to reverse course,
3. a stronger time penalty, early warnings for low altitude / low speed and re-opened
   exploration (`--reinit-std`, `--ent-coef`). This is what produced the high-G turns and dives,
4. 30 % of the episodes drawn from the extreme corners of the envelope (`--corner-prob 0.3`),
   because the agent used to stall in zoom climbs from slow, low starts. Checkpoints were
   chosen by random-target success plus corner-suite success.

```bash
python train.py --resume <previous.zip> --timesteps 2500000 --n-envs 2 --start-difficulty 1.0 \
    --corner-prob 0.3 --lr-start 1.5e-4 --lr-end 2e-5 --ent-coef 0.001 --reinit-std 0.3 \
    --run-dir runs/f16_ppo_v4 --eval-freq 250000 --eval-episodes 40
```

## Results

All numbers: `python evaluate.py --model models/f16_agent.zip --episodes 200` (raw output in
`models/eval_results.json`), 200 random start/target pairs drawn uniformly over the full envelope,
time limit 180 s per target, deterministic policy. The scripted pilot is a hand-written
PD autopilot given the same 200 episodes, to show the task is not trivial.

| | RL agent | scripted pilot |
|---|---|---|
| reached the target | **97.0%** | 6.5% |
| crashed / left the envelope | **0.0%** | 0.0% |
| time to target, median | 42 s | 68 s |
| peak load factor per maneuver, mean | 6.1 G | 4.0 G |

The misses of the RL agent are near-misses (median residual 49 ft, 2.8 deg,
4 ft/s: it ended just outside the 2.5 deg heading band), not crashes.
Peak load factor is the simulator's body-axis Nz and includes short transients; the
flight computer limits the *stick* to about 9 G. The agent regularly commands 8-11 G
on dives and tight turns and 5-6 G on moderate turns.

**Named hard maneuvers** (time to reach / peak G):

| maneuver | time | peak G |
|---|---|---|
| Max climb 10k -> 45k ft | 97 s | 7.7 |
| Max dive 45k -> 10k ft | 68 s | 11.1 |
| Reverse course (180 deg turn) | 32 s | 6.4 |
| Slow down 1100 -> 600 ft/s | 34 s | 4.9 |
| Speed up 600 -> 1100 ft/s | 34 s | 2.4 |
| Climb + reverse + slow | 80 s | 8.1 |
| Dive + 150 deg turn + fast | 60 s | 8.6 |
| Small correction (+500 ft, +10 deg) | 13 s | 5.9 |

**Extreme corners.** Every combination of start/target altitude (10k, 45k ft) and
start/target speed (600, 1 100 ft/s), each with and without a 180 deg turn
(`python evaluate.py --corners-only`): **32/32 reached**, none crashed.
The slowest is starting slow at 10 000 ft and ending fast at 45 000 ft, which needs about
135-150 s: the aircraft has to build a lot of energy and the thin air at altitude leaves little
excess thrust. Climbs, dives and 180 deg turns each take tens of seconds, the speed-up from
600 to 1 100 ft/s about 35 s.

**Known limitations.** Steady-state hold is a little jittery (the policy is deterministic
but reacts with fairly large stick motions; the load-factor strip in the UI shows +-0.5 G of
hunting). Targets right at the edge of what the engine can sustain at a given altitude
(see *Engine limits*) are approached but may not be held.


## The UI

`python server.py --model ...` then open the printed address.

* **Target panel (left).** Altitude and speed by slider or number, heading by
  dragging the dial or typing. Nothing is sent until you press *Send to
  agent*, so you can compose a target; the agent can be retargeted at any
  time, including mid-maneuver. Quick maneuvers set up common cases (reverse
  course, max climb, max dive, ...).
* **3D view.** Chase, orbit (drag to rotate, wheel to zoom) and cockpit
  cameras (keys 1/2/3). The orange ring is the target altitude, the orange
  arrow the target heading; both turn green when the aircraft is on target.
  The flight path is colored by load factor (teal = 1 G, orange = high G,
  yellow = negative G). The speed brakes and afterburner animate.
* **Telemetry (right).** Altitude, speed, heading and load factor with
  distance-to-go for each target; pitch, bank, angle of attack, climb rate,
  Mach, thrust; the agent's current stick/throttle/brake commands; and the
  time and peak G for the current target.
* **Bottom strip.** Load factor over the last 60 s, with a marker where the
  target changed.
* Space pauses, and the simulation can run at 0.5x to 4x.

## Notes and limits

* Everything is the JSBSim F-16 (a fixed-weight, clean configuration with
  full flight-control system). The 3D model in the UI is a simple stand-in
  shape and the ground is a flat grid; the world is a flat plane around the
  start point.
* `--n-envs` should match your CPU cores. About 900-1 100 steps/s per 2 cores;
  3-4 M steps already give a competent agent, 8 M+ a polished one.
* A trained model is tied to this observation/action layout; if you change
  `f16_env.py`, retrain.

## Requirements

Tested with Python 3.12 (Windows) and Python 3.13 (Linux). `agent.py` loads the saved model with
replacement constants for the pickled learning-rate lambda, because that lambda's bytecode depends on the
Python version that trained the model.
