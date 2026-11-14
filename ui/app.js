import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

/* =====================================================================
   helpers
   ===================================================================== */
const $ = (s) => document.querySelector(s);
const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const sgn = (n, d = 0) => (n >= 0 ? '+' : '−') + fmt(Math.abs(n), d);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const D2R = Math.PI / 180;
const wrap180 = (a) => ((a + 180) % 360 + 360) % 360 - 180;
const lerp = (a, b, t) => a + (b - a) * t;
const lerpAngle = (a, b, t) => a + wrap180(b - a) * t;

const COLORS = {
  ink: '#10202c', ink2: '#4a5c69', teal: '#0b7f86', orange: '#ff6a2b', green: '#17895c',
  line: '#c3cfd5',
};

/* =====================================================================
   shared state
   ===================================================================== */
const S = {
  limits: { alt: [10000, 45000], speed: [600, 1200] },
  vmax: [[10000, 1130], [14000, 1160], [18000, 1200], [45000, 1200]],
  tol: { alt_ft: 100, heading_deg: 2.5, speed_fps: 15 },
  speedMult: 1,
  paused: false,
  last: null,                 // latest state message
  target: { alt: 25000, hdg: 0, spd: 800 },     // what the agent is flying to
  pending: { alt: 25000, hdg: 0, spd: 800 },    // what the sliders show
  dirty: false,
  targetSetT: 0, reachedT: null, onSince: null, maneuverPeak: 1,
  gHist: [],
};

function vmaxFor(alt) {
  const t = S.vmax;
  if (alt <= t[0][0]) return t[0][1];
  for (let i = 1; i < t.length; i++) {
    if (alt <= t[i][0]) return lerp(t[i - 1][1], t[i][1], (alt - t[i - 1][0]) / (t[i][0] - t[i - 1][0]));
  }
  return t[t.length - 1][1];
}

/* =====================================================================
   three.js scene
   ===================================================================== */
const canvas = $('#scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const HORIZON = new THREE.Color('#a9c3d1');
scene.fog = new THREE.Fog(HORIZON, 120000, 800000);

const camera = new THREE.PerspectiveCamera(55, 1, 20, 4e6);
camera.position.set(0, 25000, 900);

// sky dome follows the camera
const sky = new THREE.Mesh(
  new THREE.SphereGeometry(2.0e6, 32, 16),
  new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, depthTest: false, fog: false,
    uniforms: {
      top: { value: new THREE.Color('#0f2c45') },
      horizon: { value: HORIZON.clone() },
      bottom: { value: new THREE.Color('#5f7f8c') },
    },
    vertexShader: 'varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 horizon; uniform vec3 bottom; varying vec3 vDir;' +
      'void main(){ float h = vDir.y; vec3 c = h > 0.0 ? mix(horizon, top, pow(h, 0.45)) : mix(horizon, bottom, pow(-h, 0.35)); gl_FragColor = vec4(c, 1.0); }',
  }));
sky.renderOrder = -10;
sky.frustumCulled = false;
scene.add(sky);

scene.add(new THREE.HemisphereLight(0xdfeaf2, 0x31423c, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(0.6, 1.0, 0.35);
scene.add(sun);

// ground + range grid (cell = 10 000 ft, snaps to the camera so it scrolls like terrain)
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(3e6, 3e6).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x3d5348 }));
scene.add(ground);
const CELL = 10000;
const grid = new THREE.GridHelper(CELL * 100, 100, 0x6f8f80, 0x557163);
grid.position.y = 2;
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);

/* ---------- aircraft model (procedural, nose toward −Z, 1 unit = 1 ft) ---------- */
function buildF16() {
  const g = new THREE.Group();
  const hull = new THREE.MeshStandardMaterial({ color: 0x9aa7b0, roughness: 0.55, metalness: 0.2, flatShading: true, side: THREE.DoubleSide });
  const dark = new THREE.MeshStandardMaterial({ color: 0x2b363e, roughness: 0.7, flatShading: true });
  const glass = new THREE.MeshStandardMaterial({ color: 0x173a55, roughness: 0.15, metalness: 0.6 });
  const light = new THREE.MeshStandardMaterial({ color: 0xd8dee2, roughness: 0.6, flatShading: true });

  const prof = [[2.5, -24], [2.8, -18], [3.0, -8], [3.1, 2], [2.8, 10], [2.2, 16], [1.4, 21], [0.5, 24.5], [0.01, 25.5]]
    .map(([r, y]) => new THREE.Vector2(r, y));
  const fus = new THREE.Mesh(new THREE.LatheGeometry(prof, 18), hull);
  fus.rotation.x = -Math.PI / 2;
  g.add(fus);

  const flat = (pts, depth, mat, mirror) => {
    const shape = new THREE.Shape();
    pts.forEach(([x, y], i) => (i ? shape.lineTo(x, y) : shape.moveTo(x, y)));
    shape.closePath();
    const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false });
    geo.rotateX(Math.PI / 2);
    const m = new THREE.Mesh(geo, mat);
    g.add(m);
    if (mirror) { const m2 = m.clone(); m2.scale.x = -1; g.add(m2); }
    return m;
  };
  flat([[2.5, -2], [16, 7], [16, 10.5], [2.5, 14]], 0.6, hull, true);          // wings
  flat([[2.8, 13.5], [9.5, 19], [9.5, 22], [2.8, 23.5]], 0.5, hull, true);     // stabilisers

  const finShape = new THREE.Shape();
  [[10, 2.2], [20.5, 9.5], [23.5, 9.5], [23.5, 2.2]].forEach(([x, y], i) => (i ? finShape.lineTo(x, y) : finShape.moveTo(x, y)));
  const finGeo = new THREE.ExtrudeGeometry(finShape, { depth: 0.5, bevelEnabled: false });
  finGeo.translate(0, 0, -0.25);
  finGeo.rotateY(-Math.PI / 2);
  g.add(new THREE.Mesh(finGeo, hull));

  const canopy = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), glass);
  canopy.scale.set(1.15, 1.0, 4.2);
  canopy.position.set(0, 2.6, -7);
  g.add(canopy);

  const intake = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 2.0, 6, 14), dark);
  intake.rotation.x = Math.PI / 2;
  intake.scale.set(1.15, 1, 0.8);
  intake.position.set(0, -2.7, -2);
  g.add(intake);

  [-1, 1].forEach((s) => {
    const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.38, 0.38, 9, 8), light);
    rail.rotation.x = Math.PI / 2;
    rail.position.set(16.4 * s, -0.1, 3.5);
    g.add(rail);
  });

  const brakes = [-1, 1].map((s) => {
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.2, 3.6, 4.5), dark);
    b.position.set(2.7 * s, 0.8, 21);
    b.userData.side = s;
    g.add(b);
    return b;
  });

  const flame = new THREE.Mesh(new THREE.ConeGeometry(2.0, 1, 16), new THREE.MeshBasicMaterial({ color: 0xff8a3d, transparent: true, opacity: 0.8 }));
  flame.rotation.x = Math.PI / 2;
  flame.position.set(0, 0, 26);
  g.add(flame);

  g.userData = { brakes, flame };
  return g;
}

const MODEL_SCALE = 3.2;
const plane = new THREE.Group();
const model = buildF16();
model.scale.setScalar(MODEL_SCALE);
plane.add(model);
scene.add(plane);

/* ---------- trail coloured by load factor ---------- */
const TRAIL_MAX = 8000;
const trailPos = new Float32Array(TRAIL_MAX * 3);
const trailCol = new Float32Array(TRAIL_MAX * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
trailGeo.setDrawRange(0, 0);
const trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, fog: false }));
const trailDots = new THREE.Points(trailGeo, new THREE.PointsMaterial({ vertexColors: true, size: 5, sizeAttenuation: false, fog: false }));
trailLine.frustumCulled = trailDots.frustumCulled = false;
scene.add(trailLine, trailDots);
let trailN = 0, trailLastT = -1;
const cTeal = new THREE.Color('#2fc4cd'), cHot = new THREE.Color('#ff6a2b'), cNeg = new THREE.Color('#f2c230');
function trailPush(p, nz) {
  if (trailN >= TRAIL_MAX) {                       // drop the oldest quarter
    const keep = Math.floor(TRAIL_MAX * 0.75), from = TRAIL_MAX - keep;
    trailPos.copyWithin(0, from * 3, TRAIL_MAX * 3);
    trailCol.copyWithin(0, from * 3, TRAIL_MAX * 3);
    trailN = keep;
  }
  const c = nz >= 1 ? cTeal.clone().lerp(cHot, clamp((nz - 1) / 6, 0, 1)) : cTeal.clone().lerp(cNeg, clamp((1 - nz) / 3, 0, 1));
  trailPos.set([p.x, p.y, p.z], trailN * 3);
  trailCol.set([c.r, c.g, c.b], trailN * 3);
  trailN++;
  trailGeo.setDrawRange(0, trailN);
  trailGeo.attributes.position.needsUpdate = true;
  trailGeo.attributes.color.needsUpdate = true;
}
function trailClear() { trailN = 0; trailLastT = -1; trailGeo.setDrawRange(0, 0); }

/* ---------- target markers ---------- */
function labelSprite(text) {
  const c = document.createElement('canvas');
  c.width = 512; c.height = 96;
  const tex = new THREE.CanvasTexture(c);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, fog: false }));
  sp.renderOrder = 20;
  sp.userData = { c, tex, text: '' };
  sp.setText = (t, color) => {
    if (sp.userData.text === t + color) return;
    sp.userData.text = t + color;
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, 512, 96);
    ctx.font = '600 44px "Segoe UI", Arial, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.lineWidth = 8; ctx.strokeStyle = 'rgba(10,25,38,.85)'; ctx.strokeText(t, 256, 48);
    ctx.fillStyle = color; ctx.fillText(t, 256, 48);
    tex.needsUpdate = true;
  };
  sp.setText(text, '#ff6a2b');
  return sp;
}

const RING_R = 3500;
const altMarker = new THREE.Group();
const altMat = new THREE.MeshBasicMaterial({ color: 0xff6a2b, transparent: true, opacity: 0.95, side: THREE.DoubleSide, fog: false });
const altRing = new THREE.Mesh(new THREE.RingGeometry(RING_R - 45, RING_R, 96).rotateX(-Math.PI / 2), altMat);
const altDisc = new THREE.Mesh(new THREE.CircleGeometry(RING_R, 64).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0xff6a2b, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false, fog: false }));
const altSpokes = new THREE.LineSegments(
  new THREE.BufferGeometry().setFromPoints([[-1, 0, 0], [1, 0, 0], [0, 0, -1], [0, 0, 1]].map(([x, y, z]) => new THREE.Vector3(x * RING_R, y, z * RING_R))),
  new THREE.LineBasicMaterial({ color: 0xff6a2b, transparent: true, opacity: 0.35, fog: false }));
const altLabel = labelSprite('');
altLabel.position.set(0, 0, -RING_R);
altLabel.scale.set(5200, 975, 1);
altMarker.add(altRing, altDisc, altSpokes, altLabel);
scene.add(altMarker);

const dropLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, -1, 0)]),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, fog: false }));
dropLine.frustumCulled = false;
scene.add(dropLine);
const groundMark = new THREE.Mesh(new THREE.RingGeometry(260, 320, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6, fog: false }));
groundMark.position.y = 4;
scene.add(groundMark);

const hdgMarker = new THREE.Group();
const hdgMat = new THREE.MeshBasicMaterial({ color: 0xff6a2b, fog: false });
const HDG_LEN = 2200;
const hdgBar = new THREE.Mesh(new THREE.BoxGeometry(7, 7, HDG_LEN), hdgMat);
hdgBar.position.z = -HDG_LEN / 2 - 300;
const hdgHead = new THREE.Mesh(new THREE.ConeGeometry(45, 200, 16).rotateX(-Math.PI / 2), hdgMat);
hdgHead.position.z = -HDG_LEN - 300 - 90;
const hdgLabel = labelSprite('');
hdgLabel.position.set(0, 150, -HDG_LEN - 800);
hdgLabel.scale.set(1500, 280, 1);
hdgMarker.add(hdgBar, hdgHead, hdgLabel);
scene.add(hdgMarker);

/* ---------- interpolation buffer ---------- */
const buf = [];
let playT = null;
const _e = new THREE.Euler();

function pushState(st) {
  if (buf.length && st.t < buf[buf.length - 1].t - 1e-6) { buf.length = 0; playT = null; trailClear(); S.gHist.length = 0; }
  const pos = new THREE.Vector3(st.east_ft, st.alt_ft, -st.north_ft);
  _e.set(st.pitch_deg * D2R, -st.heading_deg * D2R, -st.roll_deg * D2R, 'YXZ');
  const q = new THREE.Quaternion().setFromEuler(_e);
  buf.push({ t: st.t, pos, q, st });
  if (buf.length > 60) buf.shift();
}

function sampleAt(t) {
  if (!buf.length) return null;
  if (t <= buf[0].t) return { ...buf[0], a: 0, b: buf[0] };
  for (let i = buf.length - 1; i > 0; i--) {
    if (buf[i - 1].t <= t) {
      const A = buf[i - 1], B = buf[i], k = clamp((t - A.t) / Math.max(1e-6, B.t - A.t), 0, 1);
      return {
        pos: A.pos.clone().lerp(B.pos, k),
        q: A.q.clone().slerp(B.q, k),
        st: {
          ...B.st,
          heading_deg: lerpAngle(A.st.heading_deg, B.st.heading_deg, k),
          pitch_deg: lerp(A.st.pitch_deg, B.st.pitch_deg, k),
          roll_deg: lerpAngle(A.st.roll_deg, B.st.roll_deg, k),
          alt_ft: lerp(A.st.alt_ft, B.st.alt_ft, k),
        },
        A, B,
      };
    }
  }
  return { ...buf[buf.length - 1] };
}

/* ---------- cameras ---------- */
let camMode = 'chase';
const orbit = new OrbitControls(camera, canvas);
orbit.enabled = false;
orbit.enableDamping = true;
orbit.minDistance = 150;
orbit.maxDistance = 30000;
let camYaw = 0, camPitch = 0, camInit = false;
const planeLast = new THREE.Vector3();

function setCamMode(m) {
  camMode = m;
  document.querySelectorAll('[data-cam]').forEach((b) => b.classList.toggle('on', b.dataset.cam === m));
  orbit.enabled = m === 'orbit';
  model.visible = m !== 'cockpit';
  camera.fov = m === 'cockpit' ? 75 : 55;
  camera.updateProjectionMatrix();
  if (m === 'orbit') {
    orbit.target.copy(plane.position);
    camera.position.copy(plane.position).add(new THREE.Vector3(900, 350, 900));
  }
  camInit = false;
}

function updateCamera(dt, s) {
  const p = plane.position;
  if (camMode === 'chase') {
    const hdg = s.st.heading_deg * D2R, pit = s.st.pitch_deg * D2R;
    if (!camInit) { camYaw = hdg; camPitch = pit; }
    const k = 1 - Math.exp(-dt * 3.0);
    camYaw += wrap180((hdg - camYaw) / D2R) * D2R * k;
    camPitch += (pit - camPitch) * k;
    const f = new THREE.Vector3(Math.sin(camYaw) * Math.cos(camPitch * 0.6), Math.sin(camPitch * 0.6), -Math.cos(camYaw) * Math.cos(camPitch * 0.6));
    const want = p.clone().addScaledVector(f, -820).add(new THREE.Vector3(0, 190, 0));
    if (!camInit) { camera.position.copy(want); camInit = true; } else camera.position.lerp(want, 1 - Math.exp(-dt * 6));
    camera.up.set(0, 1, 0);
    camera.lookAt(p.clone().addScaledVector(f, 300));
  } else if (camMode === 'orbit') {
    const d = p.clone().sub(planeLast);
    camera.position.add(d);
    orbit.target.copy(p);
    orbit.update();
  } else {           // cockpit
    const fwd = new THREE.Vector3(0, 1.1 * MODEL_SCALE * 0.9, -9 * MODEL_SCALE).applyQuaternion(plane.quaternion);
    camera.position.copy(p).add(fwd);
    camera.quaternion.copy(plane.quaternion);
  }
  planeLast.copy(p);
}

/* =====================================================================
   per-frame render
   ===================================================================== */
const compass = $('#compass');
const cctx = compass.getContext('2d');
let lastFrame = performance.now();

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

function drawCompass(hdg, tgt) {
  const W = compass.width, H = compass.height, c = cctx;
  c.clearRect(0, 0, W, H);
  c.fillStyle = 'rgba(237,241,242,0.92)';
  c.beginPath(); c.roundRect(0, 0, W, H, 18); c.fill();
  const span = 90, pxPer = W / span;
  c.font = '600 24px "Segoe UI", Arial, sans-serif';
  c.textAlign = 'center'; c.textBaseline = 'alphabetic';
  c.fillStyle = COLORS.ink; c.strokeStyle = COLORS.ink2; c.lineWidth = 2;
  for (let d = Math.ceil((hdg - span / 2) / 5) * 5; d <= hdg + span / 2; d += 5) {
    const x = W / 2 + (d - hdg) * pxPer;
    const deg = ((d % 360) + 360) % 360;
    const major = deg % 10 === 0;
    c.beginPath(); c.moveTo(x, H); c.lineTo(x, H - (major ? 24 : 12)); c.stroke();
    if (major) {
      const lab = { 0: 'N', 90: 'E', 180: 'S', 270: 'W' }[deg];
      c.fillText(lab || String(deg / 10).padStart(2, '0'), x, H - 32);
    }
  }
  // target bug
  const dd = wrap180(tgt - hdg);
  const edge = Math.abs(dd) > span / 2 - 4;
  const bx = clamp(W / 2 + dd * pxPer, 18, W - 18);
  c.fillStyle = COLORS.orange;
  c.beginPath();
  if (edge) { const s = dd > 0 ? 1 : -1; c.moveTo(bx + 12 * s, 14); c.lineTo(bx - 8 * s, 4); c.lineTo(bx - 8 * s, 24); }
  else { c.moveTo(bx, 28); c.lineTo(bx - 11, 4); c.lineTo(bx + 11, 4); }
  c.closePath(); c.fill();
  // aircraft index
  c.fillStyle = COLORS.teal;
  c.beginPath(); c.moveTo(W / 2, H - 2); c.lineTo(W / 2 - 9, H + 0); c.lineTo(W / 2, H - 14); c.lineTo(W / 2 + 9, H + 0); c.closePath(); c.fill();
}

function frame(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  const latest = buf.length ? buf[buf.length - 1].t : 0;
  if (buf.length) {
    if (playT === null) playT = Math.max(buf[0].t, latest - 0.2);
    if (!S.paused) {
      const lag = latest - playT;
      const rate = S.speedMult * (1 + clamp((lag - 0.15 * S.speedMult) / 0.5, -0.35, 0.8));
      playT = Math.min(latest, playT + dt * rate);
    }
    const s = sampleAt(playT);
    if (s) {
      plane.position.copy(s.pos);
      plane.quaternion.copy(s.q);
      // trail grows as the displayed aircraft passes each sample
      for (const e of buf) {
        if (e.t > trailLastT && e.t <= playT + 1e-6) { trailPush(e.pos, e.st.nz_g); trailLastT = e.t; }
      }
      const st = s.st;
      const tg = S.last ? S.last.target : { alt_ft: st.alt_ft, heading_deg: st.heading_deg };
      // markers
      altMarker.position.set(s.pos.x, tg.alt_ft, s.pos.z);
      const on = S.last && S.last.on_target;
      const col = on ? 0x17c47f : 0xff6a2b;
      altMat.color.setHex(col); altDisc.material.color.setHex(col); altSpokes.material.color.setHex(col);
      altLabel.setText(`Target ${fmt(tg.alt_ft)} ft`, on ? '#5df0b0' : '#ff9a6b');
      dropLine.position.copy(s.pos);
      dropLine.scale.y = s.pos.y;
      groundMark.position.set(s.pos.x, 4, s.pos.z);
      hdgMarker.position.copy(s.pos);
      hdgMarker.rotation.y = -tg.heading_deg * D2R;
      hdgMat.color.setHex(col);
      hdgLabel.setText(`Heading ${String(Math.round(tg.heading_deg) % 360).padStart(3, '0')}°`, on ? '#5df0b0' : '#ff9a6b');
      // exhaust and speed brake
      const thr = clamp(S.last ? S.last.throttle : 0.5, 0, 1);
      model.userData.flame.visible = thr > 0.7;
      model.userData.flame.scale.set(1, 3 + thr * 14 * (thr - 0.7) / 0.3, 1);
      const sb = S.last ? S.last.speedbrake : 0;
      model.userData.brakes.forEach((b) => { b.rotation.z = b.userData.side * sb * 0.9; });
      // sky and grid follow the camera
      updateCamera(dt, s);
      sky.position.copy(camera.position);
      grid.position.x = Math.round(camera.position.x / CELL) * CELL;
      grid.position.z = Math.round(camera.position.z / CELL) * CELL;
      ground.position.x = camera.position.x; ground.position.z = camera.position.z;
      drawCompass(st.heading_deg, tg.heading_deg);
    }
  }
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

/* =====================================================================
   telemetry panel
   ===================================================================== */
function setMeter(el, v, bipolar) {
  if (bipolar) {
    const w = Math.abs(v) * 50;
    el.style.left = v >= 0 ? '50%' : `${50 - w}%`;
    el.style.width = `${w}%`;
  } else {
    el.style.left = '0';
    el.style.width = `${clamp(v, 0, 1) * 100}%`;
  }
}

function deltaText(el, text, closeness) {
  el.textContent = text;
  el.className = 'sub ' + closeness;
}

function updateTelemetry(st) {
  $('#vAlt').textContent = `${fmt(st.alt_ft)} ft`;
  $('#vSpd').textContent = `${fmt(st.vt_fps)} ft/s`;
  $('#vHdg').textContent = `${String(Math.round(st.heading_deg) % 360).padStart(3, '0')}°`;
  $('#vG').textContent = `${fmt(st.nz_g, 1)} G`;
  $('#vPeak').textContent = `Peak ${fmt(S.maneuverPeak, 1)} G on this target`;
  const e = st.err;
  deltaText($('#dAlt'), `${sgn(e.alt_ft)} ft to go`, Math.abs(e.alt_ft) < S.tol.alt_ft ? 'close' : 'far');
  deltaText($('#dSpd'), `${sgn(e.speed_fps)} ft/s to go`, Math.abs(e.speed_fps) < S.tol.speed_fps ? 'close' : 'far');
  deltaText($('#dHdg'), `${sgn(e.heading_deg, 1)}° to go`, Math.abs(e.heading_deg) < S.tol.heading_deg ? 'close' : 'far');
  $('#vPitch').textContent = `${sgn(st.pitch_deg, 1)}°`;
  $('#vRoll').textContent = `${sgn(st.roll_deg, 0)}°`;
  $('#vAoa').textContent = `${fmt(st.alpha_deg, 1)}°`;
  $('#vClimb').textContent = `${sgn(st.climb_fps * 60, 0)} ft/min`;
  $('#vMach').textContent = fmt(st.mach, 2);
  $('#vKts').textContent = `${fmt(st.vt_fps * 0.592484)} kt`;
  $('#vThrust').textContent = `${fmt(Math.max(0, st.thrust_lbs))} lbf`;
  const a = st.action || [0, 0, 0, 0, 0];
  setMeter($('#mPitch'), a[0], true);
  setMeter($('#mRoll'), a[1], true);
  setMeter($('#mRud'), a[2], true);
  setMeter($('#mThr'), (a[3] + 1) / 2, false);
  setMeter($('#mSb'), st.speedbrake, false);

  // time on this target
  const elapsed = st.t - S.targetSetT;
  let clock = `${fmt(elapsed, 1)} s`;
  if (S.reachedT !== null) clock = `Reached in ${fmt(S.reachedT, 1)} s`;
  $('#vClock').textContent = clock;

  const box = $('#status');
  const b = box.querySelector('b'), sp = box.querySelector('span');
  if (S.paused) { box.className = 'status'; b.textContent = 'Paused'; sp.textContent = 'The simulation is stopped.'; }
  else if (!st.feasible) { box.className = 'status warn'; b.textContent = 'Target is beyond engine limits'; sp.textContent = `At ${fmt(st.target.alt_ft)} ft the F-16 holds about ${fmt(vmaxFor(st.target.alt_ft))} ft/s at most. The agent will get as close as it can.`; }
  else if (st.on_target) {
    box.className = 'status on'; b.textContent = 'On target';
    sp.textContent = S.reachedT !== null ? `Reached in ${fmt(S.reachedT, 1)} s, peak ${fmt(S.maneuverPeak, 1)} G. Holding.` : 'Holding within tolerance.';
  } else {
    box.className = 'status moving'; b.textContent = 'Maneuvering';
    sp.textContent = `${fmt(elapsed, 1)} s since the target changed.`;
  }
}

/* load factor strip chart */
const gcv = $('#gchart');
const gctx = gcv.getContext('2d');
function drawG() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = gcv.clientWidth, h = gcv.clientHeight;
  if (!w || !h) return;
  if (gcv.width !== w * dpr || gcv.height !== h * dpr) { gcv.width = w * dpr; gcv.height = h * dpr; }
  const c = gctx;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  const lo = -5, hi = 10.5, padL = 34;
  const y = (g) => h - ((g - lo) / (hi - lo)) * h;
  c.font = '11px "Segoe UI", Arial, sans-serif';
  c.textBaseline = 'middle';
  [-4, 0, 3, 6, 9].forEach((g) => {
    c.strokeStyle = g === 0 ? '#9fb0b9' : '#d3dce0';
    c.lineWidth = 1;
    c.beginPath(); c.moveTo(padL, y(g)); c.lineTo(w, y(g)); c.stroke();
    c.fillStyle = COLORS.ink2; c.textAlign = 'right';
    c.fillText(`${g} G`, padL - 6, y(g));
  });
  const H = S.gHist;
  if (H.length < 2) return;
  const tEnd = H[H.length - 1].t, t0 = tEnd - 60;
  const x = (t) => padL + ((t - t0) / 60) * (w - padL);
  if (S.targetSetT > t0) { c.strokeStyle = COLORS.orange; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(x(S.targetSetT), 0); c.lineTo(x(S.targetSetT), h); c.stroke(); c.setLineDash([]); }
  c.strokeStyle = COLORS.teal; c.lineWidth = 2; c.lineJoin = 'round';
  c.beginPath();
  H.forEach((p, i) => { const px = x(p.t), py = y(clamp(p.g, lo, hi)); i ? c.lineTo(px, py) : c.moveTo(px, py); });
  c.stroke();
}
window.addEventListener('resize', drawG);

/* =====================================================================
   target controls
   ===================================================================== */
const el = {
  altRange: $('#altRange'), altIn: $('#altIn'), hdgIn: $('#hdgIn'),
  spdRange: $('#spdRange'), spdIn: $('#spdIn'), send: $('#sendBtn'), tag: $('#pendingTag'), spdNote: $('#spdNote'),
};

function renderControls() {
  const p = S.pending;
  el.altRange.value = el.altIn.value = Math.round(p.alt);
  el.spdRange.value = el.spdIn.value = Math.round(p.spd);
  el.hdgIn.value = Math.round(p.hdg) % 360;
  const vm = vmaxFor(p.alt);
  const over = p.spd > vm + 1;
  el.spdNote.textContent = over
    ? `At ${fmt(p.alt)} ft the engine holds about ${fmt(vm)} ft/s at most.`
    : `Sustainable here up to about ${fmt(vm)} ft/s.`;
  el.spdNote.className = 'note' + (over ? ' warn' : '');
  const T = S.target;
  S.dirty = Math.abs(p.alt - T.alt) > 0.5 || Math.abs(wrap180(p.hdg - T.hdg)) > 0.5 || Math.abs(p.spd - T.spd) > 0.5;
  el.tag.hidden = !S.dirty;
  drawDial();
}

function edit(field, value) {
  const [lo, hi] = field === 'alt' ? S.limits.alt : field === 'spd' ? S.limits.speed : [0, 360];
  if (!Number.isFinite(value)) return;
  S.pending[field] = field === 'hdg' ? ((value % 360) + 360) % 360 : clamp(value, lo, hi);
  renderControls();
}
el.altRange.addEventListener('input', () => edit('alt', +el.altRange.value));
el.altIn.addEventListener('change', () => edit('alt', +el.altIn.value));
el.spdRange.addEventListener('input', () => edit('spd', +el.spdRange.value));
el.spdIn.addEventListener('change', () => edit('spd', +el.spdIn.value));
el.hdgIn.addEventListener('change', () => edit('hdg', +el.hdgIn.value));
[el.altIn, el.spdIn, el.hdgIn].forEach((i) => i.addEventListener('keydown', (e) => { if (e.key === 'Enter') { i.dispatchEvent(new Event('change')); sendTarget(); } }));

function sendTarget() {
  const p = S.pending;
  send({ type: 'target', alt_ft: p.alt, heading_deg: p.hdg, speed_fps: p.spd });
  S.target = { ...p };
  S.targetSetT = S.last ? S.last.t : 0;
  S.reachedT = null; S.onSince = null; S.maneuverPeak = S.last ? S.last.nz_g : 1;
  renderControls();
  toast('Target sent to the agent');
}
el.send.addEventListener('click', sendTarget);

/* heading dial (SVG) */
const dial = $('#dial');
const NS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs, parent) { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); (parent || dial).appendChild(n); return n; }
function buildDial() {
  svg('circle', { r: 100, fill: '#fff', stroke: COLORS.line, 'stroke-width': 2 });
  for (let d = 0; d < 360; d += 10) {
    const a = d * D2R, major = d % 30 === 0, r1 = major ? 84 : 90;
    svg('line', { x1: Math.sin(a) * r1, y1: -Math.cos(a) * r1, x2: Math.sin(a) * 97, y2: -Math.cos(a) * 97, stroke: COLORS.ink2, 'stroke-width': major ? 2 : 1 });
  }
  [['N', 0], ['E', 90], ['S', 180], ['W', 270]].forEach(([t, d]) => {
    const a = d * D2R;
    const n = svg('text', { x: Math.sin(a) * 68, y: -Math.cos(a) * 68 + 5, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 650, fill: COLORS.ink });
    n.textContent = t;
  });
  dial.aircraft = svg('g', {});
  svg('path', { d: 'M0,-34 L9,10 L0,2 L-9,10 Z', fill: COLORS.teal }, dial.aircraft);
  dial.bug = svg('g', {});
  svg('line', { x1: 0, y1: -22, x2: 0, y2: -80, stroke: COLORS.orange, 'stroke-width': 3, 'stroke-linecap': 'round' }, dial.bug);
  svg('circle', { cx: 0, cy: -88, r: 11, fill: COLORS.orange, stroke: '#fff', 'stroke-width': 2 }, dial.bug);
  dial.sent = svg('path', { d: 'M0,-100 L-6,-109 L6,-109 Z', fill: COLORS.ink2 });
}
function drawDial() {
  if (!dial.bug) return;
  dial.bug.setAttribute('transform', `rotate(${S.pending.hdg})`);
  dial.sent.setAttribute('transform', `rotate(${S.target.hdg})`);
  if (S.last) dial.aircraft.setAttribute('transform', `rotate(${S.last.heading_deg})`);
}
function dialPointer(e) {
  const r = dial.getBoundingClientRect();
  const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
  let deg = Math.atan2(dx, -dy) / D2R;
  deg = Math.round((deg + 360) % 360);
  edit('hdg', deg);
}
buildDial();
let dragging = false;
dial.addEventListener('pointerdown', (e) => { dragging = true; dial.setPointerCapture(e.pointerId); dialPointer(e); });
dial.addEventListener('pointermove', (e) => { if (dragging) dialPointer(e); });
dial.addEventListener('pointerup', () => { dragging = false; });

/* quick maneuvers */
const PROGRAMS = {
  reverse: (s) => ({ hdg: (s.heading_deg + 180) % 360 }),
  climb: () => ({ alt: 45000 }),
  dive: () => ({ alt: 10000 }),
  slow: () => ({ spd: 600 }),
  fast: (s) => ({ spd: Math.min(1100, vmaxFor(S.pending.alt) - 30) }),
  hard: (s) => ({ alt: s.alt_ft > 27500 ? 10000 : 45000, hdg: (s.heading_deg + 150) % 360, spd: s.alt_ft > 27500 ? 1100 : 650 }),
};
document.querySelectorAll('[data-prog]').forEach((b) => b.addEventListener('click', () => {
  if (!S.last) return;
  const upd = PROGRAMS[b.dataset.prog](S.last);
  Object.assign(S.pending, S.target, upd);
  renderControls();
  sendTarget();
}));
document.querySelectorAll('[data-reset]').forEach((b) => b.addEventListener('click', () => send({ type: 'reset', mode: b.dataset.reset })));

/* top bar */
document.querySelectorAll('[data-cam]').forEach((b) => b.addEventListener('click', () => setCamMode(b.dataset.cam)));
$('#simSpeed').addEventListener('change', (e) => { S.speedMult = +e.target.value; send({ type: 'speed', value: S.speedMult }); });
function setPaused(v) { S.paused = v; $('#pauseBtn').textContent = v ? 'Resume' : 'Pause'; $('#pauseBtn').classList.toggle('on', v); send({ type: 'pause', value: v }); }
$('#pauseBtn').addEventListener('click', () => setPaused(!S.paused));
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); setPaused(!S.paused); }
  if (e.key === '1') setCamMode('chase');
  if (e.key === '2') setCamMode('orbit');
  if (e.key === '3') setCamMode('cockpit');
});

let toastTimer;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
}

/* =====================================================================
   websocket
   ===================================================================== */
let ws;
function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function setConn(on) {
  $('#conn').classList.toggle('on', on);
  $('#conn span').textContent = on ? 'Live' : 'Offline';
}

function adoptTarget(st) {
  const t = st.target;
  const changed = Math.abs(t.alt_ft - S.target.alt) > 0.5 || Math.abs(wrap180(t.heading_deg - S.target.hdg)) > 0.5 || Math.abs(t.speed_fps - S.target.spd) > 0.5;
  if (changed) {
    S.target = { alt: t.alt_ft, hdg: t.heading_deg, spd: t.speed_fps };
    if (!S.dirty || !S.last) S.pending = { ...S.target };
    S.targetSetT = st.t; S.reachedT = null; S.onSince = null; S.maneuverPeak = st.nz_g;
    renderControls();
  }
}

function onState(st) {
  if (S.last && st.t < S.last.t - 1e-6) { S.gHist.length = 0; S.targetSetT = 0; S.reachedT = null; S.onSince = null; S.maneuverPeak = 1; }
  adoptTarget(st);
  S.last = st;
  S.maneuverPeak = Math.max(S.maneuverPeak, st.nz_g);
  if (st.on_target) {
    if (S.onSince === null) S.onSince = st.t;
    if (S.reachedT === null && st.t - S.onSince >= 2.0) S.reachedT = S.onSince - S.targetSetT + 2.0;
  } else S.onSince = null;
  S.gHist.push({ t: st.t, g: st.nz_g });
  if (S.gHist.length > 700) S.gHist.shift();
  pushState(st);
  updateTelemetry(st);
  drawG();
  drawDial();
}

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'hello') {
      S.limits = m.limits; S.vmax = m.vmax_table; S.tol = m.tolerance;
      S.speedMult = m.speed_mult; $('#simSpeed').value = String(m.speed_mult);
      S.paused = m.paused; $('#pauseBtn').textContent = m.paused ? 'Resume' : 'Pause';
      $('#agentName').textContent = `Agent: ${m.agent.split('/').slice(-2).join('/')}`;
      const t = m.state.target;
      S.target = { alt: t.alt_ft, hdg: t.heading_deg, spd: t.speed_fps };
      S.pending = { ...S.target };
      renderControls();
      onState({ ...m.state, action: [0, 0, 0, 0, 0] });
    } else if (m.type === 'state') onState(m);
    else if (m.type === 'reset') { buf.length = 0; playT = null; trailClear(); S.gHist.length = 0; toast('Flight reset'); }
    else if (m.type === 'event' && m.kind === 'crash') toast('The aircraft left the flight envelope. Resetting.');
    else if (m.type === 'error') toast(m.message);
  };
}
renderControls();
connect();
