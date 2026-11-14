import time, math, numpy as np
from gymnasium.utils.env_checker import check_env
from f16_env import F16Env, vmax_for_alt

env = F16Env()
check_env(env, skip_render_check=True)
print('check_env OK')

# 1) zero action: does it hold trim for 60 s?
obs, _ = env.reset(seed=1, options={'init': dict(alt_ft=30000, speed_fps=900, heading_deg=45),
                                    'target': dict(alt_ft=30000, heading_deg=45, speed_fps=900)})
env.terminate_on_success = False
a = np.array([0,0,0, obs[-2], -1.0], dtype=np.float32)   # keep trimmed throttle
for i in range(600):
    obs, r, te, tr, info = env.step(a)
s = env.get_state()
print('zero-action 60 s: alt %.0f  v %.0f  hdg %.1f  nz %.2f  crash=%s' % (s['alt_ft'], s['vt_fps'], s['heading_deg'], s['nz_g'], info['crash']))

# 2) crude manual maneuver: 180 deg turn at 20k/800 with full pull + full roll to see G and turn rate
env.reset(seed=2, options={'init': dict(alt_ft=20000, speed_fps=900, heading_deg=0),
                           'target': dict(alt_ft=20000, heading_deg=180, speed_fps=900)})
env.terminate_on_success = False
peak = 0; t_turn = None
for i in range(400):
    s = env.get_state()
    dpsi = ((180 - s['heading_deg'] + 180) % 360) - 180
    phi_cmd = np.clip(dpsi*2, -75, 75)
    roll = np.clip((phi_cmd - s['roll_deg'])*0.05, -1, 1)
    pull = 0.9 if abs(s['roll_deg']) > 55 else 0.0
    a = np.array([pull, roll, 0, 1.0, -1.0], dtype=np.float32)
    obs, r, te, tr, info = env.step(a)
    peak = max(peak, s['nz_g'])
    if abs(dpsi) < 5 and t_turn is None: t_turn = env.t
print('hand 180 turn: peak Nz %.2f, t_turn %s, v end %.0f, alt end %.0f' % (peak, t_turn, s['vt_fps'], s['alt_ft']))

# 3) speedbrake effect
env.reset(seed=3, options={'init': dict(alt_ft=25000, speed_fps=1000, heading_deg=0),
                           'target': dict(alt_ft=25000, heading_deg=0, speed_fps=600)})
env.terminate_on_success = False
for sb in (-1.0, 1.0):
    env.reset(seed=3, options={'init': dict(alt_ft=25000, speed_fps=1000, heading_deg=0),
                               'target': dict(alt_ft=25000, heading_deg=0, speed_fps=600)})
    env.terminate_on_success = False
    for i in range(200):
        s = env.get_state()
        pitch = np.clip((25000 - s['alt_ft'])*0.0005 - s['climb_fps']*0.003, -0.5, 0.5)
        obs, *_ = env.step(np.array([pitch, 0, 0, -1.0, sb], dtype=np.float32))
    print('idle thrust, speedbrake=%s : v %.0f after 20 s, alt %.0f, sb pos %.2f' % (sb>0, s['vt_fps'], s['alt_ft'], s['speedbrake']))

# 4) random policy + throughput
env = F16Env()
t = time.time(); n = 0; eps = 0
obs, _ = env.reset(seed=0)
while time.time() - t < 8:
    obs, r, te, tr, info = env.step(env.action_space.sample()); n += 1
    if te or tr:
        eps += 1; obs, _ = env.reset()
print('random policy: %.0f steps/s, %d episodes in 8 s, nan obs: %s' % (n/8, eps, bool(np.isnan(obs).any())))
