// Ungroup web renderer: the look of the original SFML client (src/client/rendering) on a canvas.
//
//   import { createRenderer } from './render.js';
//   const r = createRenderer(canvas, {
//     fontUrl: 'assets/monogram.ttf',
//     dottedBackgroundUrl: 'assets/dotted_background.png',
//     minePatternUrl: 'assets/mine_pattern.png',
//     sparkUrl: 'assets/spark.png',
//     letterUrls: ['assets/a_letter.png', 'assets/m_letter.png', 'assets/e_letter.png', 'assets/n_letter.png'],
//   });
//   await r.ready;               // textures + font loaded (draw() before that skips what is missing)
//   r.resize();                  // on layout change (also called on the first draw)
//   r.draw({ frame, meta, cfg, me, camera, time, names });
//
// draw() arguments
//   frame   engine frame (see the frame schema: t, R, bodies, mines, alive, picks, players, events)
//   meta    engine meta (needs, pads, mine_pos, mine_type, cfg array)
//   cfg     config object (mine_radius, solo_radius, pad_radius, time_limit, crown, brand, mine_cap, bloom_cap ...)
//           or the meta.cfg array (mapped with CFG_FIELDS); optional when meta.cfg is present
//   me      local player index, or -1 / null for spectator mode (camera frames the whole arena)
//   camera  optional: { x, y, zoom } pins the camera in world units (no chase); { zoom } only scales the
//           chase camera; { mode: 'arena' } forces the whole-arena view
//   time    seconds, drives the cell animation and the sparks (defaults to performance.now() / 1000)
//   names   optional array of player names (the local name goes in the HUD)
//
// Font: the HUD uses the monogram pixel font. The renderer registers it through the FontFace API from
// assets.fontUrl; if you prefer CSS, add this to the page and omit fontUrl:
//   @font-face { font-family: 'monogram'; src: url('assets/monogram.ttf') format('truetype'); }
//
// Pipeline (WebGL2): the world is rendered into a low resolution framebuffer (PIXEL_SCALE css px per
// buffer pixel; the original draws at 1x and upscales the buffer 3x with nearest sampling) and blitted
// up with nearest sampling so the cells, circles and dots stay chunky. Pass 1 is one fullscreen quad for
// the dark arena disc, the out-of-bounds grey, the shrink ring and the two parallax dot layers (texelFetch,
// integer scaling). Pass 2 draws one quad per circle (mines, bodies, pads) with the voronoi_counts shader
// ported to GLSL ES 3.0. Pass 3 batches the flat geometry (pickups, arrows, crown notches, brand
// segments, leaver pulses, dead mine rings) into one triangle buffer, then the spark sprites. The HUD
// (resource letters + count/goal, round timer, name) is drawn on a 2D canvas at full resolution and
// uploaded as a texture only when its content changes; off-screen mine indicators are letter sprites.
// Without WebGL2 a 2D canvas fallback draws the same scene into a low resolution offscreen canvas; the
// cells are approximated by soft blobs at the animated cell centres (no true Voronoi partition).

export const PALETTE = [[159, 224, 246], [243, 229, 154], [243, 181, 155], [243, 156, 156]];
export const PALETTE_CSS = PALETTE.map(c => `rgb(${c[0]},${c[1]},${c[2]})`);
export const BACKGROUND_COLOR = [34, 32, 52];
export const OUT_OF_BOUNDS_COLOR = [120, 120, 120];
export const GOLD = [255, 208, 80];
export const WORLD_PX = 450;        // css px per world unit at zoom 1 (unit disc = 900 css px across)
export const PIXEL_SCALE = 2;       // css px per world buffer pixel
export const CAMERA_CHASE = 0.5;    // lerp factor per 8 ms step, like the original (CAMERA_CHASE * MIN_TIME_STEP_SEC * dt)
export const MAX_CELLS = 30;
export const RESOURCE_LETTERS = ['a', 'm', 'e', 'n'];

// Config array layout of the C++ core / Python Config dataclass (meta.cfg).
export const CFG_FIELDS = ['n_players', 'n_mines', 'dt', 'time_limit', 'base_speed', 'vel_lerp', 'solo_radius', 'mine_radius',
  'mine_cap', 'mine_regen', 'mine_rate', 'mine_exp', 'pad_radius', 'need_primary', 'need_secondary', 'leave_time',
  'spill_min_speed', 'spill_k', 'spill_max', 'pickup_ttl', 'max_pickups', 'shrink_start', 'final_radius', 'restitution',
  'max_group', 'join_cooldown', 'partner_cooldown', 'leave_forfeit', 'intent_weight', 'stun_time', 'leave_hold',
  'group_bank_bonus', 'rammer_stun_mult', 'carried_shaping', 'win_bonus', 'lose_penalty', 'relative_reward', 'crown',
  'head_vest', 'brand', 'brand_min', 'brand_base', 'brand_per_unit', 'brand_max', 'bloom_rate', 'seed_rate', 'seed_floor',
  'seed_range', 'bloom_cap', 'persist', 'ledger_decay', 'grudge_window', 'obs_legacy'];

const CFG_DEFAULTS = { solo_radius: 0.045, mine_radius: 0.08, pad_radius: 0.06, time_limit: 240, mine_cap: 30, bloom_cap: 0, crown: 0, brand: 0, stun_time: 1 };

export function cfgFromArray(arr) {
  const o = {};
  for (let i = 0; i < CFG_FIELDS.length && i < arr.length; i++) o[CFG_FIELDS[i]] = arr[i];
  return o;
}

// The colour-fill rule of voronoi_counts.frag, in JS (used by the 2D fallback and by tests):
// the first `cells` cells cycle over the four types round-robin while a type still has count left.
export function cellTypes(counts, cells) {
  const cc = [counts[0], counts[1], counts[2], counts[3]];
  const out = new Array(cells).fill(-1);
  let k = 0;
  for (let i = 0; i < MAX_CELLS && k < cells; i++) {
    for (let c = 0; c < 4 && k < cells; c++) {
      const ci = (i + c) % 4;
      if (cc[ci] > 0.5) { out[k++] = ci; cc[ci] -= 1; }
    }
  }
  return out;
}

// random2 of the shader (fract(sin(dot)*43758.5453)) in JS; used for the fallback's cell centres.
function random2(x, y) {
  const a = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  const b = Math.sin(x * 269.5 + y * 183.3) * 43758.5453;
  return [a - Math.floor(a), b - Math.floor(b)];
}
const CELL_SEEDS = [];
for (let i = 0; i < MAX_CELLS; i++) CELL_SEEDS.push(random2(i, 1.0));
export function cellPoint(i, time) {
  const s = CELL_SEEDS[i];
  return [0.5 + 0.5 * Math.sin(time + 6.2831 * s[0]), 0.5 + 0.5 * Math.sin(time + 6.2831 * s[1])];
}

// ---------------------------------------------------------------------------------------------------
// GLSL
// ---------------------------------------------------------------------------------------------------

const VS_QUAD = `#version 300 es
in vec2 a_pos;      // clip space
in vec2 a_uv;
out vec2 v_uv;
void main() { v_uv = a_uv; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FS_BLIT = `#version 300 es
precision mediump float;
uniform sampler2D u_tex;
uniform vec4 u_tint;
in vec2 v_uv;
out vec4 o;
void main() { o = texture(u_tex, v_uv) * u_tint; }`;

// Pass 1: ground disc + out of bounds + shrink ring + two parallax dot layers. Coordinates: buffer px, y down.
const FS_BG = `#version 300 es
precision highp float;
uniform vec2 u_buf;        // buffer size
uniform vec2 u_origin;     // buffer px of world (0,0)
uniform float u_scale;     // buffer px per world unit
uniform float u_R;         // shrink radius
uniform vec3 u_bg;
uniform vec3 u_oob;
uniform sampler2D u_dots;  // 200x200
uniform ivec2 u_dotSize;
uniform ivec2 u_off1;      // parallax offsets in buffer px (integer)
uniform ivec2 u_off2;
uniform int u_k1;          // integer texel scale of layer 1 (2)
uniform int u_k2;          // layer 2 (1)
uniform int u_hasDots;
out vec4 o;
ivec2 wrap(ivec2 p) { return ((p % u_dotSize) + u_dotSize) % u_dotSize; }
void main() {
  vec2 fc = vec2(gl_FragCoord.x, u_buf.y - gl_FragCoord.y);   // y down
  vec2 w = (fc - u_origin) / u_scale;
  float d = length(w);
  vec3 col = d <= 1.0 ? u_bg : u_oob;
  if (u_R < 0.9999) {
    if (d > u_R && d <= 1.0) col = mix(col, u_oob, 0.18);
    float ring = abs(d - u_R) * u_scale;           // px from the ring
    if (ring < 1.0) col = mix(col, vec3(0.80, 0.80, 0.88), 0.55);
  }
  if (u_hasDots == 1) {
    ivec2 p2 = ivec2(floor((fc + vec2(u_off2)) / float(u_k2)));
    vec4 t2 = texelFetch(u_dots, wrap(p2 + u_dotSize / 2), 0);
    col = mix(col, t2.rgb, t2.a * 0.7);
    ivec2 p1 = ivec2(floor((fc + vec2(u_off1)) / float(u_k1)));
    vec4 t1 = texelFetch(u_dots, wrap(p1), 0);
    col = mix(col, t1.rgb, t1.a * 0.8);
  }
  o = vec4(col, 1.0);
}`;

const VS_CIRCLE = `#version 300 es
in vec2 a_pos;              // unit quad [-1,1]
uniform vec2 u_buf;
uniform vec2 u_center;      // buffer px, y down
uniform float u_ext;        // half size of the quad in px
void main() {
  vec2 p = u_center + a_pos * u_ext;
  gl_Position = vec4(p.x / u_buf.x * 2.0 - 1.0, 1.0 - p.y / u_buf.y * 2.0, 0.0, 1.0);
}`;

// Pass 2: port of resources/shaders/voronoi_counts.frag. The original computes st = fract(coord / (2r))
// with coord relative to the circle's top-left corner; the same is done here in buffer pixels.
const FS_CIRCLE = `#version 300 es
precision highp float;
const int MAX_CELL_COUNT = ${MAX_CELLS};
const int COLOR_COUNT = 4;
uniform vec2 u_buf;
uniform vec2 u_center;
uniform float u_radius;
uniform int u_mode;                 // 0 flat, 1 voronoi cells, 2 voronoi cells * mine texture
uniform float u_time;
uniform int u_maxResources;         // cell count K
uniform float u_resourceCounts[COLOR_COUNT];
uniform vec4 u_fill;                // flat fill (mode 0)
uniform vec4 u_ring;                // ring colour
uniform float u_ringIn;             // ring drawn for u_ringIn <= d < u_ringOut
uniform float u_ringOut;
uniform sampler2D u_tex;
out vec4 o;

const vec4 soft_a = vec4(159. / 255., 224. / 255., 246. / 255., 1.);
const vec4 soft_b = vec4(243. / 255., 229. / 255., 154. / 255., 1.);
const vec4 soft_c = vec4(243. / 255., 181. / 255., 155. / 255., 1.);
const vec4 soft_d = vec4(243. / 255., 156. / 255., 156. / 255., 1.);

vec2 random2(vec2 p) {
  return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}

vec4 voronoi(vec2 st) {
  vec4 color_pool[COLOR_COUNT];
  color_pool[0] = soft_a; color_pool[1] = soft_b; color_pool[2] = soft_c; color_pool[3] = soft_d;
  int CELL_COUNT = int(min(float(MAX_CELL_COUNT), float(u_maxResources)));
  vec4 point_color[MAX_CELL_COUNT];
  for (int i = 0; i < MAX_CELL_COUNT; i++) point_color[i] = vec4(1., 1., 1., .05);
  float color_counts[COLOR_COUNT];
  for (int i = 0; i < COLOR_COUNT; i++) color_counts[i] = u_resourceCounts[i];
  // Fill in colours evenly (round-robin over the types while a type still has count left)
  int counter = 0;
  for (int i = 0; i < MAX_CELL_COUNT; i++) {
    if (counter >= CELL_COUNT) break;
    for (int c = 0; c < COLOR_COUNT; c++) {
      int color_index = (i + c) % COLOR_COUNT;
      if (color_counts[color_index] > 0.5 && counter < CELL_COUNT) {
        point_color[counter] = color_pool[color_index];
        counter += 1;
        color_counts[color_index] -= 1.;
      }
    }
  }
  float m_dist = 20.;
  vec4 closest = vec4(1., 1., 1., 0.05);
  for (int i = 0; i < CELL_COUNT; i++) {
    vec2 pt = random2(vec2(float(i), 1.0));
    pt = 0.5 + 0.5 * sin(u_time + 6.2831 * pt);
    float dist = distance(st, pt);
    if (dist < m_dist) { m_dist = dist; closest = point_color[i]; }
  }
  return closest;
}

void main() {
  vec2 fc = vec2(gl_FragCoord.x, u_buf.y - gl_FragCoord.y);
  float d = distance(fc, u_center);
  if (d <= u_radius) {
    if (u_mode == 0) { o = u_fill; return; }
    vec2 st = fract((fc - (u_center - vec2(u_radius))) / (u_radius * 2.));
    vec4 col = voronoi(st);
    if (u_mode == 2) {
      vec4 t = texture(u_tex, st);
      col.rgb *= mix(vec3(1.0), t.rgb, 0.55);
    }
    o = col;
    return;
  }
  if (d >= u_ringIn && d < u_ringOut) { o = u_ring; return; }
  discard;
}`;

const VS_SHAPES = `#version 300 es
in vec2 a_pos;      // buffer px, y down
in vec4 a_color;
uniform vec2 u_buf;
out vec4 v_color;
void main() {
  v_color = a_color;
  gl_Position = vec4(a_pos.x / u_buf.x * 2.0 - 1.0, 1.0 - a_pos.y / u_buf.y * 2.0, 0.0, 1.0);
}`;
const FS_SHAPES = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 o;
void main() { o = v_color; }`;

// ---------------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------------

function isImage(x) {
  return x && (typeof HTMLImageElement !== 'undefined' && x instanceof HTMLImageElement ||
    typeof ImageBitmap !== 'undefined' && x instanceof ImageBitmap ||
    typeof HTMLCanvasElement !== 'undefined' && x instanceof HTMLCanvasElement);
}

function loadImage(src) {
  if (!src) return Promise.resolve(null);
  if (isImage(src)) return Promise.resolve(src);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => { console.warn('render: could not load', src); resolve(null); };
    img.src = src;
  });
}

async function loadFont(fontUrl) {
  if (!fontUrl || typeof FontFace === 'undefined' || !document.fonts) return false;
  try {
    const face = new FontFace('monogram', `url(${fontUrl})`);
    await face.load();
    document.fonts.add(face);
    return true;
  } catch (e) { console.warn('render: font', e); return false; }
}

function rgba(c, a = 1) { return [c[0] / 255, c[1] / 255, c[2] / 255, a]; }
function css(c, a = 1) { return `rgba(${c[0]},${c[1]},${c[2]},${a})`; }
function pad2(n) { n = Math.max(0, Math.floor(n)); return n < 10 ? '0' + n : String(n); }

// Batched flat geometry (triangles), positions in buffer px.
class ShapeBatch {
  constructor() { this.data = new Float32Array(6 * 6 * 512); this.n = 0; }
  _grow() { const d = new Float32Array(this.data.length * 2); d.set(this.data); this.data = d; }
  vert(x, y, c) {
    if (this.n + 6 > this.data.length) this._grow();
    const d = this.data, n = this.n;
    d[n] = x; d[n + 1] = y; d[n + 2] = c[0]; d[n + 3] = c[1]; d[n + 4] = c[2]; d[n + 5] = c[3];
    this.n += 6;
  }
  tri(ax, ay, bx, by, cx, cy, c) { this.vert(ax, ay, c); this.vert(bx, by, c); this.vert(cx, cy, c); }
  quad(ax, ay, bx, by, cx, cy, dx, dy, c) { this.tri(ax, ay, bx, by, cx, cy, c); this.tri(ax, ay, cx, cy, dx, dy, c); }
  poly(cx, cy, r, n, c, rot = 0) {
    for (let i = 0; i < n; i++) {
      const a0 = rot + i / n * Math.PI * 2, a1 = rot + (i + 1) / n * Math.PI * 2;
      this.tri(cx, cy, cx + r * Math.cos(a0), cy + r * Math.sin(a0), cx + r * Math.cos(a1), cy + r * Math.sin(a1), c);
    }
  }
  ring(cx, cy, rIn, rOut, c, a0 = 0, a1 = Math.PI * 2, seg = 0) {
    if (!seg) seg = Math.max(8, Math.ceil(Math.abs(a1 - a0) / (Math.PI * 2) * Math.max(12, rOut * 1.2)));
    for (let i = 0; i < seg; i++) {
      const t0 = a0 + (a1 - a0) * i / seg, t1 = a0 + (a1 - a0) * (i + 1) / seg;
      const c0 = Math.cos(t0), s0 = Math.sin(t0), c1 = Math.cos(t1), s1 = Math.sin(t1);
      this.quad(cx + rIn * c0, cy + rIn * s0, cx + rOut * c0, cy + rOut * s0, cx + rOut * c1, cy + rOut * s1, cx + rIn * c1, cy + rIn * s1, c);
    }
  }
  // Triangle "arrow" like sf::CircleShape(size/2, 3) rotated to point along (dx,dy): tip at the far end.
  arrow(cx, cy, dx, dy, size, c) {
    const r = size / 2, ang = Math.atan2(dy, dx), pts = [];
    for (let k = 0; k < 3; k++) { const a = ang + k * Math.PI * 2 / 3; pts.push(cx + r * Math.cos(a), cy + r * Math.sin(a)); }
    this.tri(pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], c);
  }
  clear() { this.n = 0; }
}

// ---------------------------------------------------------------------------------------------------
// WebGL2 backend
// ---------------------------------------------------------------------------------------------------

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s) + '\n' + src);
  return s;
}
function program(gl, vs, fs, attribs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  attribs.forEach((a, i) => gl.bindAttribLocation(p, i, a));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(p));
  const u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name.replace('[0]', '')] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}

function makeGL(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) return null;
  const B = {};
  B.gl = gl;
  B.mode = 'webgl2';
  B.blit = program(gl, VS_QUAD, FS_BLIT, ['a_pos', 'a_uv']);
  B.bg = program(gl, VS_QUAD, FS_BG, ['a_pos', 'a_uv']);
  B.circle = program(gl, VS_CIRCLE, FS_CIRCLE, ['a_pos']);
  B.shapes = program(gl, VS_SHAPES, FS_SHAPES, ['a_pos', 'a_color']);

  // unit quad (pos in [-1,1], uv in [0,1] with v flipped so image row 0 is at the top)
  B.quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, B.quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 0, 1, 1, -1, 1, 1, -1, 1, 0, 0, 1, 1, 1, 0]), gl.STATIC_DRAW);
  B.quadVao = gl.createVertexArray();
  gl.bindVertexArray(B.quadVao);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
  B.dynQuad = gl.createBuffer();       // per-sprite quad (pos clip, uv)
  B.dynVao = gl.createVertexArray();
  gl.bindVertexArray(B.dynVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, B.dynQuad);
  gl.bufferData(gl.ARRAY_BUFFER, 16 * 4, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
  B.shapeBuf = gl.createBuffer();
  B.shapeVao = gl.createVertexArray();
  gl.bindVertexArray(B.shapeVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, B.shapeBuf);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 24, 8);
  gl.bindVertexArray(null);

  B.tex = {};
  B.texture = (name, img, nearest = true) => {
    const t = B.tex[name] || gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, nearest ? gl.NEAREST : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, nearest ? gl.NEAREST : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    B.tex[name] = t;
    return t;
  };
  // 1x1 white fallback texture
  B.texture('white', new ImageData(new Uint8ClampedArray([255, 255, 255, 255]), 1, 1));

  // world framebuffer
  B.fbo = gl.createFramebuffer();
  B.fboTex = gl.createTexture();
  B.bufW = 0; B.bufH = 0;
  B.ensureBuffer = (w, h) => {
    if (w === B.bufW && h === B.bufH) return;
    B.bufW = w; B.bufH = h;
    gl.bindTexture(gl.TEXTURE_2D, B.fboTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, B.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, B.fboTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  };

  B.begin = (w, h) => {
    B.ensureBuffer(w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, B.fbo);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  };

  B.background = (p) => {
    const P = B.bg;
    gl.useProgram(P.p); B._prog = P;
    gl.uniform2f(P.u.u_buf, B.bufW, B.bufH);
    gl.uniform2f(P.u.u_origin, p.ox, p.oy);
    gl.uniform1f(P.u.u_scale, p.scale);
    gl.uniform1f(P.u.u_R, p.R);
    gl.uniform3f(P.u.u_bg, BACKGROUND_COLOR[0] / 255, BACKGROUND_COLOR[1] / 255, BACKGROUND_COLOR[2] / 255);
    gl.uniform3f(P.u.u_oob, OUT_OF_BOUNDS_COLOR[0] / 255, OUT_OF_BOUNDS_COLOR[1] / 255, OUT_OF_BOUNDS_COLOR[2] / 255);
    gl.uniform2i(P.u.u_dotSize, p.dotW, p.dotH);
    gl.uniform2i(P.u.u_off1, p.off1x, p.off1y);
    gl.uniform2i(P.u.u_off2, p.off2x, p.off2y);
    gl.uniform1i(P.u.u_k1, 2);
    gl.uniform1i(P.u.u_k2, 1);
    gl.uniform1i(P.u.u_hasDots, B.tex.dots ? 1 : 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, B.tex.dots || B.tex.white);
    gl.uniform1i(P.u.u_dots, 0);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(B.quadVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.enable(gl.BLEND);
  };

  B.drawCircle = (c) => {
    const P = B.circle;
    if (B._prog !== P) { gl.useProgram(P.p); B._prog = P; gl.bindVertexArray(B.quadVao); gl.uniform2f(P.u.u_buf, B.bufW, B.bufH); }
    const ext = Math.max(c.r, c.ringOut || 0) + 1;
    gl.uniform2f(P.u.u_center, c.x, c.y);
    gl.uniform1f(P.u.u_ext, ext);
    gl.uniform1f(P.u.u_radius, c.r);
    gl.uniform1i(P.u.u_mode, c.mode);
    gl.uniform1f(P.u.u_time, c.time || 0);
    gl.uniform1i(P.u.u_maxResources, c.cells | 0);
    const cnt = c.counts || [0, 0, 0, 0];
    gl.uniform1fv(P.u.u_resourceCounts, cnt);
    const f = c.fill || [0, 0, 0, 0];
    gl.uniform4f(P.u.u_fill, f[0], f[1], f[2], f[3]);
    const rg = c.ring || [0, 0, 0, 0];
    gl.uniform4f(P.u.u_ring, rg[0], rg[1], rg[2], rg[3]);
    gl.uniform1f(P.u.u_ringIn, c.ringIn || 0);
    gl.uniform1f(P.u.u_ringOut, c.ringOut || 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, (c.mode === 2 && B.tex.mine) || B.tex.white);
    gl.uniform1i(P.u.u_tex, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };

  B.flushShapes = (batch) => {
    if (!batch.n) return;
    const P = B.shapes;
    gl.useProgram(P.p); B._prog = P;
    gl.uniform2f(P.u.u_buf, B.bufW, B.bufH);
    gl.bindVertexArray(B.shapeVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, B.shapeBuf);
    gl.bufferData(gl.ARRAY_BUFFER, batch.data.subarray(0, batch.n), gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.TRIANGLES, 0, batch.n / 6);
    batch.clear();
  };

  // textured quad; x,y,w,h in the current target's pixels (y down), uv rect in texture space
  B._sprite = (tex, x, y, w, h, u0, v0, u1, v1, tint, W, H) => {
    const P = B.blit;
    gl.useProgram(P.p); B._prog = P;
    const cx0 = x / W * 2 - 1, cx1 = (x + w) / W * 2 - 1, cy0 = 1 - y / H * 2, cy1 = 1 - (y + h) / H * 2;
    gl.bindVertexArray(B.dynVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, B.dynQuad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([cx0, cy1, u0, v1, cx1, cy1, u1, v1, cx0, cy0, u0, v0, cx1, cy0, u1, v0]), gl.DYNAMIC_DRAW);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(P.u.u_tex, 0);
    gl.uniform4f(P.u.u_tint, tint[0], tint[1], tint[2], tint[3]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };
  B.sprite = (name, x, y, w, h, u0, v0, u1, v1, tint) => {
    if (!B.tex[name]) return;
    B._sprite(B.tex[name], x, y, w, h, u0, v0, u1, v1, tint, B.bufW, B.bufH);
  };

  B.end = () => {
    // blit the world buffer to the screen with nearest sampling, PIXEL_SCALE*dpr device px per buffer px
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.disable(gl.BLEND);
    B._sprite(B.fboTex, 0, 0, B.bufW * B.pix, B.bufH * B.pix, 0, 1, 1, 0, [1, 1, 1, 1], canvas.width, canvas.height);
    gl.enable(gl.BLEND);
  };
  B.screenSprite = (name, x, y, w, h, u0, v0, u1, v1, tint) => {
    if (!B.tex[name]) return;
    B._sprite(B.tex[name], x, y, w, h, u0, v0, u1, v1, tint, canvas.width, canvas.height);
  };
  B.hud = (hudCanvas, dirty) => {
    if (dirty) B.texture('hud', hudCanvas, true);
    if (!B.tex.hud) return;
    B._sprite(B.tex.hud, 0, 0, hudCanvas.width, hudCanvas.height, 0, 0, 1, 1, [1, 1, 1, 1], canvas.width, canvas.height);
  };
  B.destroy = () => { const ext = gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); };
  return B;
}

// ---------------------------------------------------------------------------------------------------
// 2D canvas backend (fallback). Cells are approximated by blobs at the animated cell centres.
// ---------------------------------------------------------------------------------------------------

function make2D(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const B = { mode: 'canvas2d', ctx, img: {}, tex: {} };
  const off = document.createElement('canvas');
  const octx = off.getContext('2d');
  let bufW = 0, bufH = 0;
  const dotsCache = {};
  B.texture = (name, img) => { B.img[name] = img; B.tex[name] = true; };
  B.begin = (w, h) => {
    if (w !== bufW || h !== bufH) { off.width = w; off.height = h; bufW = w; bufH = h; }
    B.bufW = w; B.bufH = h;
    octx.setTransform(1, 0, 0, 1, 0, 0);
    octx.imageSmoothingEnabled = false;
    octx.globalAlpha = 1;
  };
  // a scaled copy of the dotted texture (integer scale), cached
  const dotsScaled = (k) => {
    const img = B.img.dots;
    if (!img) return null;
    if (dotsCache[k]) return dotsCache[k];
    const c = document.createElement('canvas');
    c.width = img.width * k; c.height = img.height * k;
    const cx = c.getContext('2d'); cx.imageSmoothingEnabled = false;
    cx.drawImage(img, 0, 0, c.width, c.height);
    dotsCache[k] = c;
    return c;
  };
  B.background = (p) => {
    octx.fillStyle = css(OUT_OF_BOUNDS_COLOR);
    octx.fillRect(0, 0, bufW, bufH);
    octx.fillStyle = css(BACKGROUND_COLOR);
    octx.beginPath(); octx.arc(p.ox, p.oy, p.scale, 0, Math.PI * 2); octx.fill();
    if (p.R < 0.9999) {
      octx.fillStyle = css(OUT_OF_BOUNDS_COLOR, 0.18);
      octx.beginPath(); octx.arc(p.ox, p.oy, p.scale, 0, Math.PI * 2); octx.arc(p.ox, p.oy, p.R * p.scale, 0, Math.PI * 2, true); octx.fill();
      octx.strokeStyle = 'rgba(204,204,224,0.55)'; octx.lineWidth = 1.5;
      octx.beginPath(); octx.arc(p.ox, p.oy, p.R * p.scale, 0, Math.PI * 2); octx.stroke();
    }
    const layer = (k, offx, offy, alpha, half) => {
      const img = dotsScaled(k);
      if (!img) return;
      const tw = img.width, th = img.height;
      let sx = ((-offx - (half ? tw / 2 : 0)) % tw + tw) % tw, sy = ((-offy - (half ? th / 2 : 0)) % th + th) % th;
      octx.globalAlpha = alpha;
      for (let y = -sy; y < bufH; y += th) for (let x = -sx; x < bufW; x += tw) octx.drawImage(img, x, y);
      octx.globalAlpha = 1;
    };
    layer(1, p.off2x, p.off2y, 0.7, true);
    layer(2, p.off1x, p.off1y, 0.8, false);
  };
  B.drawCircle = (c) => {
    octx.save();
    octx.beginPath(); octx.arc(c.x, c.y, c.r, 0, Math.PI * 2); octx.closePath();
    if (c.mode === 0) {
      octx.fillStyle = css([c.fill[0] * 255, c.fill[1] * 255, c.fill[2] * 255], c.fill[3]);
      octx.fill();
    } else {
      octx.clip();
      octx.fillStyle = 'rgba(255,255,255,0.05)';
      octx.fillRect(c.x - c.r, c.y - c.r, 2 * c.r, 2 * c.r);
      if (c.mode === 2 && B.img.mine) {
        octx.globalAlpha = 0.25;
        octx.drawImage(B.img.mine, c.x - c.r, c.y - c.r, 2 * c.r, 2 * c.r);
        octx.globalAlpha = 1;
      }
      const cells = Math.min(MAX_CELLS, c.cells | 0);
      if (cells > 0) {
        const types = cellTypes(c.counts, cells);
        const br = c.r * Math.max(0.55, 1.5 / Math.sqrt(cells));
        for (let i = 0; i < cells; i++) {
          if (types[i] < 0) continue;
          const [px, py] = cellPoint(i, c.time);
          octx.fillStyle = PALETTE_CSS[types[i]];
          octx.beginPath(); octx.arc(c.x - c.r + px * 2 * c.r, c.y - c.r + py * 2 * c.r, br, 0, Math.PI * 2); octx.fill();
        }
      }
    }
    octx.restore();
    if (c.ringOut > (c.ringIn || 0)) {
      octx.strokeStyle = css([c.ring[0] * 255, c.ring[1] * 255, c.ring[2] * 255], c.ring[3]);
      octx.lineWidth = c.ringOut - c.ringIn;
      octx.beginPath(); octx.arc(c.x, c.y, (c.ringOut + c.ringIn) / 2, 0, Math.PI * 2); octx.stroke();
    }
  };
  B.flushShapes = (batch) => {
    const d = batch.data;
    for (let i = 0; i < batch.n; i += 18) {
      octx.fillStyle = `rgba(${(d[i + 2] * 255) | 0},${(d[i + 3] * 255) | 0},${(d[i + 4] * 255) | 0},${d[i + 5]})`;
      octx.beginPath(); octx.moveTo(d[i], d[i + 1]); octx.lineTo(d[i + 6], d[i + 7]); octx.lineTo(d[i + 12], d[i + 13]); octx.closePath(); octx.fill();
    }
    batch.clear();
  };
  const spr = (target, name, x, y, w, h, u0, v0, u1, v1, tint) => {
    const img = B.img[name];
    if (!img) return;
    const sx = u0 * img.width, sy = v0 * img.height, sw = (u1 - u0) * img.width, sh = (v1 - v0) * img.height;
    target.imageSmoothingEnabled = false;
    target.globalAlpha = tint[3];
    target.drawImage(img, sx, sy, sw, sh, x, y, w, h);
    target.globalAlpha = 1;
  };
  B.sprite = (name, x, y, w, h, u0, v0, u1, v1, tint) => spr(octx, name, x, y, w, h, u0, v0, u1, v1, tint);
  B.end = () => {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, bufW * B.pix, bufH * B.pix);
  };
  // tinted letter sprites need a tint pass on 2D: draw into a tiny canvas with multiply
  const tintCache = {};
  B.screenSprite = (name, x, y, w, h, u0, v0, u1, v1, tint) => {
    const img = B.img[name];
    if (!img) return;
    const key = name + '|' + tint.map(v => Math.round(v * 255)).join(',');
    let t = tintCache[key];
    if (!t) {
      t = document.createElement('canvas'); t.width = img.width; t.height = img.height;
      const tc = t.getContext('2d');
      tc.drawImage(img, 0, 0);
      tc.globalCompositeOperation = 'multiply';
      tc.fillStyle = css([tint[0] * 255, tint[1] * 255, tint[2] * 255]); tc.fillRect(0, 0, t.width, t.height);
      tc.globalCompositeOperation = 'destination-in';
      tc.drawImage(img, 0, 0);
      tintCache[key] = t;
    }
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = tint[3];
    ctx.drawImage(t, u0 * t.width, v0 * t.height, (u1 - u0) * t.width, (v1 - v0) * t.height, x, y, w, h);
    ctx.globalAlpha = 1;
  };
  B.hud = (hudCanvas) => { ctx.imageSmoothingEnabled = false; ctx.drawImage(hudCanvas, 0, 0); };
  B.destroy = () => {};
  return B;
}

// ---------------------------------------------------------------------------------------------------
// the renderer
// ---------------------------------------------------------------------------------------------------

export function createRenderer(canvas, assets = {}, options = {}) {
  const opt = { worldPx: WORLD_PX, pixelScale: PIXEL_SCALE, chase: CAMERA_CHASE, hud: true, ...options };
  let B = null;
  if (opt.mode !== 'canvas2d') { try { B = makeGL(canvas); } catch (e) { console.warn('render: WebGL2 failed, using 2D', e); B = null; } }
  if (!B) B = make2D(canvas);
  if (!B) throw new Error('render: no canvas context available');

  const hudCanvas = document.createElement('canvas');
  const hctx = hudCanvas.getContext('2d');
  const letters = [null, null, null, null];
  let sparkImg = null;

  const state = {
    cam: { x: 0, y: 0, zoom: 1, init: false, lastMs: 0 },
    sparks: [],            // { x, y, t0 } world units
    stunSeen: new Map(),   // body key -> stun value at the previous draw
    hudKey: '',
    dpr: 1, cssW: 0, cssH: 0,
    stats: { frameMs: 0, avgMs: 0, frames: 0 },
    fontReady: false,
  };

  const letterUrls = Array.isArray(assets.letterUrls) ? assets.letterUrls
    : assets.letterUrls ? RESOURCE_LETTERS.map(k => assets.letterUrls[k]) : [];
  const ready = Promise.all([
    loadImage(assets.dottedBackgroundUrl).then(img => { if (img) B.texture('dots', img); state.dots = img; }),
    loadImage(assets.minePatternUrl).then(img => { if (img) B.texture('mine', img); }),
    loadImage(assets.sparkUrl).then(img => { if (img) { B.texture('spark', img); sparkImg = img; } }),
    ...letterUrls.map((u, i) => loadImage(u).then(img => { if (img) { B.texture('letter' + i, img); letters[i] = img; } })),
    loadFont(assets.fontUrl).then(ok => { state.fontReady = ok || (document.fonts && document.fonts.check && document.fonts.check('16px monogram')); }),
  ]).then(() => { state.hudKey = ''; return renderer; });

  function resize() {
    const dpr = Math.max(1, Math.min(4, window.devicePixelRatio || 1));
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(1, Math.round(rect.width || canvas.clientWidth || canvas.width));
    const cssH = Math.max(1, Math.round(rect.height || canvas.clientHeight || canvas.height));
    const w = Math.round(cssW * dpr), h = Math.round(cssH * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    state.dpr = dpr; state.cssW = cssW; state.cssH = cssH;
    if (hudCanvas.width !== w || hudCanvas.height !== h) { hudCanvas.width = w; hudCanvas.height = h; state.hudKey = ''; }
  }

  const batch = new ShapeBatch();

  function findBody(frame, me) {
    if (me == null || me < 0) return null;
    for (const b of frame.bodies) if (b.m.indexOf(me) >= 0) return b;
    return null;
  }

  function draw(args) {
    const t0 = performance.now();
    const { frame, meta } = args;
    if (!frame) return;
    if (!state.cssW || canvas.width !== Math.round(state.cssW * state.dpr)) resize();
    const cfg = args.cfg ? (Array.isArray(args.cfg) ? cfgFromArray(args.cfg) : args.cfg)
      : (meta && meta.cfg ? cfgFromArray(meta.cfg) : CFG_DEFAULTS);
    const me = (args.me == null) ? -1 : args.me;
    const time = args.time != null ? args.time : performance.now() / 1000;
    const names = args.names || [];
    const camArg = args.camera || {};
    const dpr = state.dpr;
    const pix = Math.max(1, Math.round(opt.pixelScale * dpr));   // device px per buffer px
    B.pix = pix;
    const bufW = Math.ceil(canvas.width / pix), bufH = Math.ceil(canvas.height / pix);
    const R = frame.R != null ? frame.R : 1;

    // ---- camera (world units) ----
    const cam = state.cam;
    const myBody = findBody(frame, me);
    const nowMs = performance.now();
    const dtMs = cam.lastMs ? Math.min(100, nowMs - cam.lastMs) : 16;
    cam.lastMs = nowMs;
    let zoom = camArg.zoom || 1;
    let scale;  // buffer px per world unit
    const arenaMode = camArg.mode === 'arena' || (me < 0 && camArg.x == null);
    if (camArg.x != null && camArg.y != null) {
      cam.x = camArg.x; cam.y = camArg.y; cam.init = true;
      scale = opt.worldPx * dpr / pix * zoom;
    } else if (arenaMode) {
      const tx = 0, ty = 0;
      if (!cam.init) { cam.x = tx; cam.y = ty; cam.init = true; }
      const a = Math.min(1, opt.chase * 0.008 * dtMs);
      cam.x += (tx - cam.x) * a; cam.y += (ty - cam.y) * a;
      scale = Math.min(bufW, bufH) / 2.12 * zoom;
    } else {
      const tx = myBody ? myBody.x : 0, ty = myBody ? myBody.y : 0;
      if (!cam.init) { cam.x = tx; cam.y = ty; cam.init = true; }
      const a = Math.min(1, opt.chase * 0.008 * dtMs);
      cam.x += (tx - cam.x) * a; cam.y += (ty - cam.y) * a;
      scale = opt.worldPx * dpr / pix * zoom;
    }
    // buffer px of the world origin (y down); snapped to whole buffer pixels so the dots stay aligned
    const ox = Math.round(bufW / 2 - cam.x * scale), oy = Math.round(bufH / 2 - cam.y * scale);
    const X = (wx) => ox + wx * scale, Y = (wy) => oy + wy * scale;
    const soloR = (cfg.solo_radius || 0.045) * scale;                 // buffer px of a solo body
    const unit = Math.max(1, Math.round(soloR / 10));                   // 1 buffer px at zoom 1 (the original's game unit)

    // ---- pass 1: ground ----
    B.begin(bufW, bufH);
    const camPx = cam.x * scale, camPy = cam.y * scale;
    B.background({
      ox, oy, scale, R,
      dotW: state.dots ? state.dots.width : 200, dotH: state.dots ? state.dots.height : 200,
      off1x: Math.round(camPx / 10), off1y: Math.round(camPy / 10),
      off2x: Math.round(camPx / 5), off2y: Math.round(camPy / 5),
    });

    // ---- pads ----
    const padR = (cfg.pad_radius || 0.06);
    const padRing = Math.max(R - padR - 0.02, 0.1);
    const crownOn = !!cfg.crown;
    let headOfMine = -1;
    if (myBody && myBody.m.length > 1 && crownOn) headOfMine = myBody.head;
    if (meta && meta.pads) {
      for (let i = 0; i < meta.pads.length; i++) {
        const a = meta.pads[i];
        const px = X(padRing * Math.cos(a)), py = Y(padRing * Math.sin(a));
        const mine = i === me, head = i === headOfMine && i !== me;
        B.drawCircle({
          x: px, y: py, r: padR * scale, mode: 0, cells: 0,
          fill: mine ? [1, 1, 1, 0.13] : head ? rgba(GOLD, 0.12) : [1, 1, 1, 0.06],
          ring: mine ? [1, 1, 1, 0.75] : head ? rgba(GOLD, 0.8) : [1, 1, 1, 0.22],
          ringIn: padR * scale, ringOut: padR * scale + unit,
        });
      }
    }

    // ---- pickups (small cells of their type colour) ----
    const pickR = Math.max(2, 0.012 * scale);
    for (const p of frame.picks || []) {
      const c = PALETTE[p[2] & 3];
      batch.poly(X(p[0]), Y(p[1]), pickR, 6, rgba(c, 0.95), time * 2 + p[0] * 7);
    }
    B.flushShapes(batch);

    // ---- mines ----
    const mineR = (cfg.mine_radius || 0.08) * scale;
    const K = cfg.bloom_cap > 0 ? cfg.bloom_cap : (cfg.mine_cap || 30);
    const cells = Math.max(1, Math.round(K));
    if (meta && meta.mine_pos) {
      for (let m = 0; m < meta.mine_pos.length; m++) {
        const [mx, my] = meta.mine_pos[m];
        const type = meta.mine_type ? meta.mine_type[m] & 3 : m & 3;
        const alive = frame.alive ? frame.alive[m] : true;
        const cx = X(mx), cy = Y(my);
        if (!alive) {
          B.drawCircle({ x: cx, y: cy, r: mineR, mode: 0, cells: 0, fill: [0, 0, 0, 0.18], ring: [0.45, 0.44, 0.52, 0.5], ringIn: mineR, ringOut: mineR + unit });
          continue;
        }
        const stock = frame.mines ? frame.mines[m] : K;
        const counts = [0, 0, 0, 0];
        counts[type] = Math.min(cells, Math.round(stock));
        B.drawCircle({ x: cx, y: cy, r: mineR, mode: 2, cells, counts, time, ring: rgba(PALETTE[type], 0.75), ringIn: mineR, ringOut: mineR + unit });
      }
    }

    // ---- bodies ----
    const players = frame.players || [];
    const bodyOf = [];
    for (const b of frame.bodies) {
      const n = b.m.length;
      const r = (cfg.solo_radius || 0.045) * Math.sqrt(n) * scale;
      const cx = X(b.x), cy = Y(b.y);
      const counts = b.pool.map(v => Math.round(v));
      const total = counts[0] + counts[1] + counts[2] + counts[3];
      let joinable = n > 0;
      for (const i of b.m) { const p = players[i]; if (!p || !p.join) joinable = false; }
      B.drawCircle({
        x: cx, y: cy, r, mode: 1, cells: total, counts, time,
        ring: [1, 1, 1, 1], ringIn: joinable ? r + unit : 0, ringOut: joinable ? r + 2 * unit : 0,
      });
      bodyOf.push({ b, cx, cy, r, n });
    }

    // ---- arrows, crown notch, brand rim, leaver pulse (one batch) ----
    const arrowSize = 6 * unit, edgeDist = 3 * unit, TWO_MINUS_SQRT3 = 0.2679;
    for (const { b, cx, cy, r, n } of bodyOf) {
      for (let k = 0; k < b.m.length; k++) {
        const i = b.m[k];
        const p = players[i];
        if (!p) continue;
        let dx = p.dir ? p.dir[0] : 0, dy = p.dir ? p.dir[1] : 0;
        const len = Math.hypot(dx, dy);
        const hasDir = len > 1e-6;
        if (hasDir) { dx /= len; dy /= len; }
        const ang = hasDir ? Math.atan2(dy, dx) : (k / b.m.length) * Math.PI * 2 - Math.PI / 2;
        const color = rgba(PALETTE[(p.intent | 0) & 3], i === me ? 1 : 0.2);
        const dist = r + TWO_MINUS_SQRT3 * arrowSize + edgeDist;
        if (hasDir) {
          batch.arrow(cx + dx * dist, cy + dy * dist, dx, dy, arrowSize, color);
          if (crownOn && n > 1 && b.head === i) {
            // small gold crown notch beyond the arrow tip
            const nd = dist + arrowSize * 0.9;
            batch.poly(cx + dx * nd, cy + dy * nd, 1.6 * unit, 4, rgba(GOLD, i === me ? 1 : 0.85), ang);
          }
        } else if (crownOn && n > 1 && b.head === i) {
          batch.poly(cx + Math.cos(ang) * (r + 2.5 * unit), cy + Math.sin(ang) * (r + 2.5 * unit), 1.6 * unit, 4, rgba(GOLD, 0.85), ang);
        }
        if (p.brand > 0) {
          // red rim segment centred on the member's direction
          batch.ring(cx, cy, r + 0.5 * unit, r + 2 * unit, rgba(PALETTE[3], 0.95), ang - 0.45, ang + 0.45, 6);
        }
        if (p.leaving != null && p.leaving >= 0) {
          const ph = 0.5 + 0.5 * Math.sin(time * 10);
          const rr = r + 2 * unit + 3 * unit * ph;
          batch.ring(cx, cy, rr, rr + unit, [1, 1, 1, 0.35 + 0.5 * ph]);
        }
      }
    }
    B.flushShapes(batch);

    // ---- sparks (stun / spill) ----
    const sparks = state.sparks;
    for (const e of frame.events || []) {
      if (e.kind === 'spill' && e.x != null) sparks.push({ x: e.x, y: e.y, t0: time, key: 'ev' + e.t });
    }
    const seen = new Set();
    for (const b of frame.bodies) {
      const key = b.m.join(',');
      seen.add(key);
      const prev = state.stunSeen.get(key) || 0;
      if (b.stun > 0 && prev <= 0) {
        // a spill event at the same tick already made a spark near this body
        const near = sparks.some(s => time - s.t0 < 0.05 && Math.hypot(s.x - b.x, s.y - b.y) < 0.2);
        if (!near) sparks.push({ x: b.x, y: b.y, t0: time });
      }
      state.stunSeen.set(key, b.stun);
    }
    for (const key of Array.from(state.stunSeen.keys())) if (!seen.has(key)) state.stunSeen.delete(key);
    const SPARK_DUR = 0.24, SPARK_FRAMES = 6;
    for (let s = sparks.length - 1; s >= 0; s--) {
      const sp = sparks[s];
      const age = time - sp.t0;
      if (age < 0 || age >= SPARK_DUR) { sparks.splice(s, 1); continue; }
      const f = Math.min(SPARK_FRAMES - 1, Math.floor(age / SPARK_DUR * SPARK_FRAMES));
      const size = 48 * unit;
      B.sprite('spark', X(sp.x) - size / 2, Y(sp.y) - size / 2, size, size, f / SPARK_FRAMES, 0, (f + 1) / SPARK_FRAMES, 1, [1, 1, 1, 1]);
    }
    if (sparks.length > 64) sparks.splice(0, sparks.length - 64);

    // ---- present ----
    B.end();

    // ---- HUD (full resolution) ----
    if (opt.hud) drawHUD({ frame, meta, cfg, me, names, myBody, X, Y, scale, ox, oy, bufW, bufH, pix, time });

    const ms = performance.now() - t0;
    state.stats.frameMs = ms;
    state.stats.frames++;
    state.stats.avgMs = state.stats.avgMs ? state.stats.avgMs * 0.95 + ms * 0.05 : ms;
  }

  function drawHUD(ctx) {
    const { frame, meta, cfg, me, names, myBody } = ctx;
    const dpr = state.dpr;
    const W = hudCanvas.width, H = hudCanvas.height;
    const banked = me >= 0 && frame.players[me] ? frame.players[me].banked : null;
    const needs = me >= 0 && meta && meta.needs ? meta.needs[me] : null;
    const timeLimit = cfg.time_limit || 240;
    const left = Math.max(0, Math.ceil(timeLimit - frame.t));
    const name = me >= 0 ? (names[me] || ('player ' + me)) : 'spectator';
    const key = [W, H, name, left, banked ? banked.map(v => Math.floor(v)).join(',') : '-', needs ? needs.join(',') : '-',
      frame.R, state.fontReady, letters.map(l => !!l).join('')].join('|');
    const dirty = key !== state.hudKey;
    if (dirty) {
      state.hudKey = key;
      hctx.setTransform(1, 0, 0, 1, 0, 0);
      hctx.clearRect(0, 0, W, H);
      hctx.imageSmoothingEnabled = false;
      const fpx = Math.round(16 * dpr) * 3;    // monogram renders crisp at multiples of its pixel grid
      hctx.font = `${fpx}px monogram, "Courier New", monospace`;
      hctx.textBaseline = 'middle';
      hctx.fillStyle = '#fff';
      // resources, top right: tinted letter + count/goal
      const rowH = fpx * 0.95, padR = 24 * dpr, padT = 22 * dpr;
      const letterPx = 3 * Math.round(dpr) * 10;
      if (banked && needs) {
        for (let i = 0; i < 4; i++) {
          const text = pad2(banked[i]) + '/' + pad2(needs[i]);
          const tw = hctx.measureText('00/00').width;
          const y = padT + rowH * i + rowH / 2;
          hctx.textAlign = 'left';
          const tx = W - padR - tw;
          hctx.fillStyle = banked[i] >= needs[i] ? PALETTE_CSS[i] : '#fff';
          hctx.fillText(text, tx, y);
          drawLetter(hctx, i, tx - letterPx - 12 * dpr, y - letterPx / 2, letterPx);
        }
      }
      // round timer, top left
      const tr = 22 * dpr, tx = 24 * dpr + tr, ty = 24 * dpr + tr;
      hctx.lineWidth = 3 * dpr;
      hctx.strokeStyle = 'rgba(255,255,255,0.18)';
      hctx.beginPath(); hctx.arc(tx, ty, tr, 0, Math.PI * 2); hctx.stroke();
      const frac = Math.max(0, Math.min(1, left / timeLimit));
      hctx.strokeStyle = frac < 0.15 ? PALETTE_CSS[3] : '#fff';
      hctx.beginPath(); hctx.arc(tx, ty, tr, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * frac); hctx.stroke();
      hctx.font = `${Math.round(16 * dpr) * 2}px monogram, "Courier New", monospace`;
      hctx.textAlign = 'center';
      hctx.fillStyle = '#fff';
      hctx.fillText(String(left), tx, ty + 1 * dpr);
      // name
      hctx.textAlign = 'left';
      hctx.font = `${Math.round(16 * dpr) * 2}px monogram, "Courier New", monospace`;
      hctx.fillStyle = me >= 0 ? PALETTE_CSS[(frame.players[me] && frame.players[me].intent | 0) & 3] : 'rgba(255,255,255,0.7)';
      hctx.fillText(name, tx + tr + 12 * dpr, ty);
    }
    B.hud(hudCanvas, dirty);

    // off-screen mines of the local player's intent type: letter at the screen edge (original behaviour)
    if (me >= 0 && meta && meta.mine_pos && frame.players[me]) {
      const intent = (frame.players[me].intent | 0) & 3;
      const pad = 8 * dpr, L = 3 * Math.round(dpr) * 10;
      const { X, Y, pix, scale } = ctx;
      const mineR = (cfg.mine_radius || 0.08) * scale * pix;
      for (let m = 0; m < meta.mine_pos.length; m++) {
        if ((meta.mine_type ? meta.mine_type[m] : m) !== intent) continue;
        if (frame.alive && !frame.alive[m]) continue;
        const sx = X(meta.mine_pos[m][0]) * pix, sy = Y(meta.mine_pos[m][1]) * pix;   // device px
        if (sx + mineR > 0 && sx - mineR < W && sy + mineR > 0 && sy - mineR < H) continue;
        const cx = Math.min(W - pad - L / 2, Math.max(pad + L / 2, sx));
        const cy = Math.min(H - pad - L / 2, Math.max(pad + L / 2, sy));
        B.screenSprite('letter' + intent, cx - L / 2, cy - L / 2, L, L, 0, 0, 1, 1, rgba(PALETTE[intent], 0.9));
      }
    }
  }

  const tintedLetters = {};
  function drawLetter(c, i, x, y, size) {
    const img = letters[i];
    if (!img) {
      c.save(); c.fillStyle = PALETTE_CSS[i]; c.textAlign = 'center';
      c.fillText(RESOURCE_LETTERS[i].toUpperCase(), x + size / 2, y + size / 2); c.restore();
      return;
    }
    let t = tintedLetters[i];
    if (!t) {
      t = document.createElement('canvas'); t.width = img.width; t.height = img.height;
      const tc = t.getContext('2d');
      tc.drawImage(img, 0, 0);
      tc.globalCompositeOperation = 'multiply';
      tc.fillStyle = PALETTE_CSS[i]; tc.fillRect(0, 0, t.width, t.height);
      tc.globalCompositeOperation = 'destination-in';
      tc.drawImage(img, 0, 0);
      tintedLetters[i] = t;
    }
    c.imageSmoothingEnabled = false;
    c.drawImage(t, x, y, size, size);
  }

  const renderer = {
    resize, draw, ready,
    get mode() { return B.mode; },
    stats: state.stats,
    camera: state.cam,
    hudCanvas,
    destroy() { B.destroy(); },
  };
  return renderer;
}
