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
// Pipeline, after RenderingController.cpp: the original draws the world at one pixel per game unit into
// a render texture (GAME_SIZE = 1200 px for the 600 unit arena radius) and blits that buffer to the window
// scaled 3x with nearest sampling, so everything is chunky 3x3 pixel blocks. Here the buffer covers the
// visible part of the world at UNIT_PX buffer px per world unit (R = 1) and is blitted up by PIXEL_SCALE
// css px per buffer px with nearest sampling. Pass 1 is one fullscreen quad for the dark arena disc, the
// out-of-bounds grey and the two dot layers of dotted_background.png (static in world space, like the
// shipped client whose parallax update is disabled). Pass 2 draws one quad per circle (pads, mines,
// bodies) with voronoi_counts.frag ported to GLSL ES 3.0 (cells tiled over the circle's bounding box,
// K = capacity cells for mines, K = units carried for bodies, uncoloured cells white at 5%). Pass 3
// batches the flat geometry (direction arrows, crown notches, brand rims, pickups) into one triangle
// buffer, then the spark frames (spark.png, 6 frames, 2x, 240 ms). The HUD is the original's
// ResourceUIElement: tinted 10 px letter sprites at 2.5x and "NN/NN" in monogram at 55 px, top right,
// plus off-screen mine letters at the view edge; it is drawn on a 2D canvas at full resolution and
// uploaded as a texture only when its content changes. Without WebGL2 a 2D canvas fallback rasterises the
// same scene by hand into a low resolution offscreen canvas (per-pixel cells, no anti-aliasing).

export const PALETTE = [[159, 224, 246], [243, 229, 154], [243, 181, 155], [243, 156, 156]];
export const PALETTE_CSS = PALETTE.map(c => `rgb(${c[0]},${c[1]},${c[2]})`);
export const BACKGROUND_COLOR = [34, 32, 52];
export const OUT_OF_BOUNDS_COLOR = [120, 120, 120];   // the original's grey beyond the arena edge
export const UNGROUP_COLOR = [0, 146, 199];             // the original's ring on a group that is being left
export const GOLD = [255, 208, 80];
export const UNIT_PX = 450;         // buffer px per world unit (R = 1) at zoom 1: a mine (r 0.08) is 72 px across, near the original's 80
export const PIXEL_SCALE = 3;       // css px per buffer px (the original upscales its 1x buffer 3x)
export const ARENA_PIXEL_SCALE = 2; // css px per buffer px in the whole-arena view (a map: finer blocks on purpose)
export const WORLD_PX = UNIT_PX * PIXEL_SCALE;   // css px per world unit at zoom 1 (1500: the arena is 3000 px across)
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

// Pass 1: the arena disc (BACKGROUND_COLOR inside the current radius R, OUT_OF_BOUNDS grey beyond, so a
// shrinking arena is the grey closing in) and the two dot layers of BackgroundController: sprite 2 at 1x
// with the texture rect offset by half a tile, sprite 1 at 2x, alphas 0.7 / 0.8, both anchored to the
// arena's top-left corner in world space. Coordinates: buffer px, y down.
const FS_BG = `#version 300 es
precision highp float;
uniform vec2 u_buf;        // buffer size
uniform vec2 u_origin;     // buffer px of world (0,0)
uniform float u_scale;     // buffer px per world unit
uniform float u_R;         // arena radius (world units)
uniform vec3 u_bg;
uniform vec3 u_oob;
uniform sampler2D u_dots;  // 200x200
uniform ivec2 u_dotSize;
uniform ivec2 u_off1;      // buffer px added to the fragment before the texel lookup (integer)
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
  vec3 col = d <= u_R ? u_bg : u_oob;
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
uniform int u_mode;                 // 0 flat fill, 1 voronoi cells
uniform float u_time;
uniform int u_maxResources;         // cell count K
uniform float u_resourceCounts[COLOR_COUNT];
uniform vec4 u_fill;                // flat fill (mode 0)
uniform vec4 u_ring;                // ring colour
uniform float u_ringIn;             // ring drawn for u_ringIn <= d < u_ringOut
uniform float u_ringOut;
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
    // voronoi_counts.frag: st = fract((coord - top_left) / (2 r)), one tile over the circle's bounding box
    vec2 st = fract((fc - (u_center - vec2(u_radius))) / (u_radius * 2.));
    o = voronoi(st);
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
    const oob = p.oob;
    const P = B.bg;
    gl.useProgram(P.p); B._prog = P;
    gl.uniform2f(P.u.u_buf, B.bufW, B.bufH);
    gl.uniform2f(P.u.u_origin, p.ox, p.oy);
    gl.uniform1f(P.u.u_scale, p.scale);
    gl.uniform1f(P.u.u_R, p.R);
    gl.uniform3f(P.u.u_bg, BACKGROUND_COLOR[0] / 255, BACKGROUND_COLOR[1] / 255, BACKGROUND_COLOR[2] / 255);
    gl.uniform3f(P.u.u_oob, oob[0] / 255, oob[1] / 255, oob[2] / 255);
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
    const sh = B.shift || [0, 0];
    B._sprite(B.fboTex, sh[0], sh[1], B.bufW * B.pix, B.bufH * B.pix, 0, 1, 1, 0, [1, 1, 1, 1], canvas.width, canvas.height);
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
  B.finish = () => gl.finish();
  B.destroy = () => { const ext = gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); };
  return B;
}

// ---------------------------------------------------------------------------------------------------
// 2D canvas backend (fallback): the same scene rasterised by hand on the buffer grid (pixel-centre tests,
// per-pixel Voronoi cells, no anti-aliasing) so it looks like the WebGL path, only slower.
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
    octx.fillStyle = css(p.oob);
    octx.fillRect(0, 0, bufW, bufH);
    octx.fillStyle = css(BACKGROUND_COLOR);
    const rr = p.R * p.scale;
    for (let y = Math.max(0, Math.floor(p.oy - rr)); y <= Math.min(bufH - 1, Math.ceil(p.oy + rr)); y++) {
      const dy = y + 0.5 - p.oy;
      if (Math.abs(dy) > rr) continue;
      const hw = Math.sqrt(rr * rr - dy * dy);
      const xa = Math.ceil(p.ox - hw - 0.5), xb = Math.floor(p.ox + hw - 0.5);
      if (xb >= xa) octx.fillRect(xa, y, xb - xa + 1, 1);
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
  // A circle rasterised with the pixel-centre test of the WebGL path (d <= r), the cells a true Voronoi
  // partition per pixel, rings for ringIn <= d < ringOut; alpha is blended by hand (no anti-aliasing).
  B.drawCircle = (c) => {
    const r = c.r, cx = c.x, cy = c.y;
    const rOut = Math.max(r, c.ringOut || 0);
    const x0 = Math.max(0, Math.floor(cx - rOut - 1)), x1 = Math.min(bufW - 1, Math.ceil(cx + rOut + 1));
    const y0 = Math.max(0, Math.floor(cy - rOut - 1)), y1 = Math.min(bufH - 1, Math.ceil(cy + rOut + 1));
    if (x1 < x0 || y1 < y0) return;
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    const img = octx.getImageData(x0, y0, w, h);
    const d = img.data;
    const cells = c.mode === 1 ? Math.min(MAX_CELLS, c.cells | 0) : 0;
    const types = cells > 0 ? cellTypes(c.counts, cells) : null;
    const pts = [];
    for (let i = 0; i < cells; i++) pts.push(cellPoint(i, c.time));
    const fill = c.fill || [0, 0, 0, 0], ring = c.ring || [0, 0, 0, 0];
    const rIn = c.ringIn || 0, rO = c.ringOut || 0;
    const blend = (o, col, a) => { d[o] = d[o] + (col[0] - d[o]) * a; d[o + 1] = d[o + 1] + (col[1] - d[o + 1]) * a; d[o + 2] = d[o + 2] + (col[2] - d[o + 2]) * a; d[o + 3] = 255; };
    const white = [255, 255, 255];
    for (let py = 0; py < h; py++) {
      const fy = y0 + py + 0.5 - cy;
      for (let px = 0; px < w; px++) {
        const fx = x0 + px + 0.5 - cx;
        const dist = Math.sqrt(fx * fx + fy * fy);
        const o = (py * w + px) * 4;
        if (dist <= r) {
          if (c.mode === 0) { blend(o, [fill[0] * 255, fill[1] * 255, fill[2] * 255], fill[3]); continue; }
          let best = -1, bd = 1e9;
          if (cells > 0) {
            const sx = ((fx + r) / (2 * r)) % 1, sy = ((fy + r) / (2 * r)) % 1;
            for (let i = 0; i < cells; i++) { const ddx = sx - pts[i][0], ddy = sy - pts[i][1]; const dd = ddx * ddx + ddy * ddy; if (dd < bd) { bd = dd; best = i; } }
          }
          if (best >= 0 && types[best] >= 0) blend(o, PALETTE[types[best]], 1); else blend(o, white, 0.05);
        } else if (dist >= rIn && dist < rO) blend(o, [ring[0] * 255, ring[1] * 255, ring[2] * 255], ring[3]);
      }
    }
    octx.putImageData(img, x0, y0);
  };
  // triangles on the pixel grid (edge functions on pixel centres, no anti-aliasing)
  B.flushShapes = (batch) => {
    const d = batch.data;
    for (let i = 0; i < batch.n; i += 18) {
      const ax = d[i], ay = d[i + 1], bx = d[i + 6], by = d[i + 7], qx = d[i + 12], qy = d[i + 13];
      const col = `rgba(${(d[i + 2] * 255) | 0},${(d[i + 3] * 255) | 0},${(d[i + 4] * 255) | 0},${d[i + 5]})`;
      const x0 = Math.max(0, Math.floor(Math.min(ax, bx, qx))), x1 = Math.min(bufW - 1, Math.ceil(Math.max(ax, bx, qx)));
      const y0 = Math.max(0, Math.floor(Math.min(ay, by, qy))), y1 = Math.min(bufH - 1, Math.ceil(Math.max(ay, by, qy)));
      const area = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax);
      if (Math.abs(area) < 1e-9) continue;
      const sgn = area > 0 ? 1 : -1;
      octx.fillStyle = col;
      for (let y = y0; y <= y1; y++) {
        const cy = y + 0.5;
        let start = -1;
        for (let x = x0; x <= x1 + 1; x++) {
          const cx = x + 0.5;
          const inside = x <= x1 && sgn * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) >= 0 && sgn * ((qx - bx) * (cy - by) - (qy - by) * (cx - bx)) >= 0 && sgn * ((ax - qx) * (cy - qy) - (ay - qy) * (cx - qx)) >= 0;
          if (inside && start < 0) start = x;
          if (!inside && start >= 0) { octx.fillRect(start, y, x - start, 1); start = -1; }
        }
      }
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
    const sh = B.shift || [0, 0];
    ctx.drawImage(off, sh[0], sh[1], bufW * B.pix, bufH * B.pix);
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
  B.finish = () => {};
  B.destroy = () => {};
  return B;
}

// ---------------------------------------------------------------------------------------------------
// the renderer
// ---------------------------------------------------------------------------------------------------

export function createRenderer(canvas, assets = {}, options = {}) {
  const opt = { unitPx: UNIT_PX, pixelScale: PIXEL_SCALE, arenaPixelScale: ARENA_PIXEL_SCALE, chase: CAMERA_CHASE, hud: true, outOfBounds: OUT_OF_BOUNDS_COLOR, ...options };
  let B = null;
  if (opt.mode !== 'canvas2d') { try { B = makeGL(canvas); } catch (e) { console.warn('render: WebGL2 failed, using 2D', e); B = null; } }
  if (!B) B = make2D(canvas);
  if (!B) throw new Error('render: no canvas context available');

  const hudCanvas = document.createElement('canvas');
  const hctx = hudCanvas.getContext('2d');
  const letters = [null, null, null, null];

  const state = {
    cam: { x: 0, y: 0, zoom: 1, init: false, lastMs: 0 },
    view: { ox: 0, oy: 0, scale: 1, pix: 1, bufW: 1, bufH: 1 },   // the last draw's transform (project())
    sparks: [],            // { x, y, t0 } world units
    hits: new Map(),       // body key @ mine -> { t: last hit spark, units: whole units carried then, seen }
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
    loadImage(assets.sparkUrl).then(img => { if (img) B.texture('spark', img); }),
    ...letterUrls.map((u, i) => loadImage(u).then(img => { if (img) { B.texture('letter' + i, img); letters[i] = img; } })),
    loadFont(assets.fontUrl).then(ok => { state.fontReady = ok || (document.fonts && document.fonts.check && document.fonts.check('16px monogram')); }),
  ]).then(() => { state.hudKey = ''; return renderer; });

  // a 10x10 ring, the "pad" counterpart of the letter sprites for the off-screen indicator
  {
    const c = document.createElement('canvas'); c.width = 10; c.height = 10;
    const cx = c.getContext('2d');
    const rows = ['..######..', '.########.', '##......##', '##......##', '##......##', '##......##', '##......##', '##......##', '.########.', '..######..'];
    cx.fillStyle = '#fff';
    rows.forEach((row, y) => { for (let x = 0; x < 10; x++) if (row[x] === '#') cx.fillRect(x, y, 1, 1); });
    B.texture('padicon', c, true);
  }

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

  // world -> css px of the canvas, with the transform of the last draw
  function project(wx, wy) {
    const v = state.view;
    return { x: ((v.ox + wx * v.scale) * v.pix + (v.shx || 0)) / state.dpr, y: ((v.oy + wy * v.scale) * v.pix + (v.shy || 0)) / state.dpr };
  }

  function draw(args) {
    const t0 = performance.now();
    const { frame, meta } = args;
    if (!frame) return;
    if (!state.cssW || canvas.clientWidth !== state.cssW || canvas.clientHeight !== state.cssH || canvas.width !== Math.round(state.cssW * state.dpr)) resize();
    const cfg = args.cfg ? (Array.isArray(args.cfg) ? cfgFromArray(args.cfg) : args.cfg)
      : (meta && meta.cfg ? cfgFromArray(meta.cfg) : CFG_DEFAULTS);
    const me = (args.me == null) ? -1 : args.me;
    const time = args.time != null ? args.time : performance.now() / 1000;
    const camArg = args.camera || {};
    const dpr = state.dpr;
    const myBody = findBody(frame, me);
    const arenaMode = camArg.mode === 'arena' || (me < 0 && camArg.x == null);
    // chunky pixels: PIXEL_SCALE css px per buffer px (2 on narrow screens so enough world stays in view)
    const pscale = arenaMode ? opt.arenaPixelScale : (state.cssW < 700 ? Math.min(2, opt.pixelScale) : opt.pixelScale);
    const pix = Math.max(1, Math.round(pscale * dpr));   // device px per buffer px (whole, so blocks stay even; a fractional dpr changes the world's size a little, as a different window size did in the original)
    B.pix = pix;
    const bufW = Math.ceil(canvas.width / pix) + 1, bufH = Math.ceil(canvas.height / pix) + 1;   // a spare column and row for the sub-block shift
    const R = frame.R != null ? frame.R : 1;

    // ---- camera (world units): the original lerps the view centre to the player each frame ----
    const cam = state.cam;
    const nowMs = performance.now();
    const dtMs = cam.lastMs ? Math.min(100, nowMs - cam.lastMs) : 16;
    cam.lastMs = nowMs;
    const zoom = camArg.zoom || 1;
    let scale;  // buffer px per world unit
    if (camArg.x != null && camArg.y != null) {
      cam.x = camArg.x; cam.y = camArg.y; cam.init = true;
      scale = opt.unitPx * zoom;
    } else if (arenaMode) {
      if (!cam.init) { cam.x = 0; cam.y = 0; cam.init = true; }
      const a = Math.min(1, opt.chase * 0.008 * dtMs);
      cam.x += (0 - cam.x) * a; cam.y += (0 - cam.y) * a;
      scale = Math.min(bufW, bufH) / 2.12 * zoom;
    } else {
      const tx = myBody ? myBody.x : 0, ty = myBody ? myBody.y : 0;
      if (!cam.init) { cam.x = tx; cam.y = ty; cam.init = true; }
      const a = Math.min(1, opt.chase * 0.008 * dtMs);
      cam.x += (tx - cam.x) * a; cam.y += (ty - cam.y) * a;
      scale = opt.unitPx * zoom;
    }
    // buffer px of the world origin (y down): a whole buffer pixel so the dots and rings stay on the grid;
    // the sub-pixel remainder shifts the blit by whole device px (the original moves its buffer sprite by
    // fractional window px, so the view scrolls one window pixel at a time, not one block)
    const oxf = bufW / 2 - cam.x * scale, oyf = bufH / 2 - cam.y * scale;
    const ox = Math.ceil(oxf), oy = Math.ceil(oyf);
    const shx = Math.round((oxf - ox) * pix), shy = Math.round((oyf - oy) * pix);   // -(pix-1)..0 device px
    B.shift = [shx, shy];
    const X = (wx) => ox + wx * scale, Y = (wy) => oy + wy * scale;
    state.view = { ox, oy, scale, pix, bufW, bufH, shx, shy };

    // ---- pass 1: ground. The dot layers are anchored to the arena's top-left corner (world -1,-1), the
    // original's texture origin; sprite 2 starts half a tile in. ----
    B.begin(bufW, bufH);
    const dotOx = Math.round(ox - scale), dotOy = Math.round(oy - scale);
    B.background({
      ox, oy, scale, R, oob: opt.outOfBounds,
      dotW: state.dots ? state.dots.width : 200, dotH: state.dots ? state.dots.height : 200,
      off1x: -dotOx, off1y: -dotOy,
      off2x: -dotOx, off2y: -dotOy,
    });

    // ---- pads: a one pixel ring (the own pad bright, the group head's pad gold, the rest faint) ----
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
        const pr = padR * scale;
        B.drawCircle({
          x: px, y: py, r: pr, mode: 0, cells: 0,
          fill: mine ? [1, 1, 1, 0.08] : head ? rgba(GOLD, 0.06) : [1, 1, 1, 0.03],
          ring: mine ? [1, 1, 1, 0.9] : head ? rgba(GOLD, 0.85) : [1, 1, 1, 0.22],
          ringIn: pr, ringOut: pr + 1,
        });
      }
    }

    // ---- bodies: DrawableGroup, K = units carried (an empty body is the 5% white disc), the joinable
    // outline is the original's 1 px white ring one pixel out, a group being left gets the ungroup blue ----
    const players = frame.players || [];
    const bodyOf = [];
    for (const b of frame.bodies) {
      const n = b.m.length;
      const r = (cfg.solo_radius || 0.045) * Math.sqrt(n) * scale;
      const cx = X(b.x), cy = Y(b.y);
      const counts = b.pool.map(v => Math.round(v));
      const total = counts[0] + counts[1] + counts[2] + counts[3];
      let joinable = n > 0, leaving = false;
      for (const i of b.m) { const p = players[i]; if (!p || !p.join) joinable = false; if (p && p.leaving != null && p.leaving >= 0) leaving = true; }
      B.drawCircle({
        x: cx, y: cy, r, mode: 1, cells: total, counts, time,
        ring: leaving ? rgba(UNGROUP_COLOR, 1) : [1, 1, 1, 1], ringIn: joinable || leaving ? r + 1 : 0, ringOut: joinable || leaving ? r + 2 : 0,
      });
      bodyOf.push({ b, cx, cy, r, n });
    }

    // ---- pickups: a spilled unit on the ground, a small disc of its colour ----
    for (const p of frame.picks || []) {
      B.drawCircle({ x: X(p[0]), y: Y(p[1]), r: Math.max(2, 0.008 * scale), mode: 0, cells: 0, fill: rgba(PALETTE[p[2] & 3], 1), ringIn: 0, ringOut: 0 });
    }

    // ---- mines: DrawableMine, K = capacity cells, the stock coloured, the rest white at 5%, no outline; drawn
    // after the groups like GameObjectRenderer::draw (a body at a mine tucks under its edge) ----
    const mineR = (cfg.mine_radius || 0.08) * scale;
    const K = cfg.bloom_cap > 0 ? cfg.bloom_cap : (cfg.mine_cap || 30);
    const cells = Math.max(1, Math.round(K));
    if (meta && meta.mine_pos) {
      for (let m = 0; m < meta.mine_pos.length; m++) {
        const [mx, my] = meta.mine_pos[m];
        const type = meta.mine_type ? meta.mine_type[m] & 3 : m & 3;
        const alive = frame.alive ? frame.alive[m] : true;
        const stock = alive ? (frame.mines ? frame.mines[m] : K) : 0;
        const counts = [0, 0, 0, 0];
        counts[type] = Math.max(0, Math.min(cells, Math.round(stock)));
        B.drawCircle({ x: X(mx), y: Y(my), r: mineR, mode: 1, cells, counts, time, ringIn: 0, ringOut: 0 });
      }
    }

    // ---- direction arrows (DirectionArrows.cpp: a 6 px triangle 3 px off the edge, intent colour, other
    // players' at 20%), the crown notch and the brand rim (one batch) ----
    const arrowSize = 6, edgeDist = 3, TWO_MINUS_SQRT3 = 0.2679;
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
        if (hasDir) batch.arrow(cx + dx * dist, cy + dy * dist, dx, dy, arrowSize, color);
        if (crownOn && n > 1 && b.head === i) {
          // a small gold diamond beyond the arrow (or on the rim when the head stands still)
          const nd = hasDir ? dist + arrowSize * 0.9 : r + 3;
          batch.poly(cx + Math.cos(ang) * nd, cy + Math.sin(ang) * nd, 2, 4, rgba(GOLD, i === me ? 1 : 0.85), ang);
        }
        if (p.brand > 0) batch.ring(cx, cy, r + 1, r + 2.5, rgba(PALETTE[3], 0.95), ang - 0.45, ang + 0.45, 6);   // red rim segment on the leaver's side
      }
    }
    B.flushShapes(batch);

    // ---- sparks: the collision animation (spark.png 6 frames at 2x = 64 px, 240 ms), timed in game seconds ----
    const sparks = state.sparks;
    const gt = frame.t || 0;
    if (state.lastGameT != null && (gt < state.lastGameT - 0.5 || gt > state.lastGameT + 5)) { sparks.length = 0; state.stunSeen.clear(); state.hits.clear(); }
    state.lastGameT = gt;
    for (const e of frame.events || []) {
      if (e.kind === 'spill' && e.x != null && !sparks.some(s => s.ev === e.t + ':' + e.x + ':' + e.y)) sparks.push({ x: e.x, y: e.y, t0: e.t != null ? e.t : gt, ev: e.t + ':' + e.x + ':' + e.y });
    }
    // mine hits: the original's mines were rigid bodies the group kept bumping into, each bump moving a unit
    // and playing a spark at the contact point; here a body gathers continuously while it touches a mine, so
    // a spark is played on the mine's edge at every 0.5 s of gathering and at every whole unit
    if (meta && meta.mine_pos) {
      const mr = cfg.mine_radius || 0.08, sr = cfg.solo_radius || 0.045;
      for (const b of frame.bodies) {
        if (b.stun > 0) continue;
        const key = b.m.join(',');
        const br = sr * Math.sqrt(b.m.length);
        const carried = b.pool[0] + b.pool[1] + b.pool[2] + b.pool[3];
        for (let m = 0; m < meta.mine_pos.length; m++) {
          if (frame.alive && !frame.alive[m]) continue;
          if (frame.mines && !(frame.mines[m] > 0)) continue;
          const dx = b.x - meta.mine_pos[m][0], dy = b.y - meta.mine_pos[m][1];
          const d = Math.hypot(dx, dy);
          if (d >= br + mr + 0.01) continue;
          const hk = key + '@' + m;
          const h = state.hits.get(hk);
          if (!h) { state.hits.set(hk, { t: gt, units: Math.floor(carried), seen: gt }); continue; }
          h.seen = gt;
          const wholeUnit = Math.floor(carried) > h.units;
          if (gt - h.t >= 0.5 || wholeUnit) {
            h.t = gt; h.units = Math.floor(carried);
            const nx = d > 1e-6 ? dx / d : 1, ny = d > 1e-6 ? dy / d : 0;
            sparks.push({ x: meta.mine_pos[m][0] + nx * mr, y: meta.mine_pos[m][1] + ny * mr, t0: gt });
          }
        }
      }
      for (const [hk, h] of state.hits) if (gt - h.seen > 0.3) state.hits.delete(hk);
    }
    const seen = new Set();
    for (const b of frame.bodies) {
      const key = b.m.join(',');
      seen.add(key);
      const prev = state.stunSeen.get(key) || 0;
      if (b.stun > 0 && prev <= 0) {
        // a spill event at the same tick already made a spark near this body
        const near = sparks.some(s => gt - s.t0 < 0.2 && Math.hypot(s.x - b.x, s.y - b.y) < 0.2);
        if (!near) sparks.push({ x: b.x, y: b.y, t0: gt });
      }
      state.stunSeen.set(key, b.stun);
    }
    for (const key of Array.from(state.stunSeen.keys())) if (!seen.has(key)) state.stunSeen.delete(key);
    const SPARK_DUR = 0.24, SPARK_FRAMES = 6, SPARK_PX = 64;
    for (let s = sparks.length - 1; s >= 0; s--) {
      const sp = sparks[s];
      const age = gt - sp.t0;
      if (age < 0 || age >= SPARK_DUR) { sparks.splice(s, 1); continue; }
      const f = Math.min(SPARK_FRAMES - 1, Math.floor(age / SPARK_DUR * SPARK_FRAMES));
      B.sprite('spark', Math.round(X(sp.x) - SPARK_PX / 2), Math.round(Y(sp.y) - SPARK_PX / 2), SPARK_PX, SPARK_PX, f / SPARK_FRAMES, 0, (f + 1) / SPARK_FRAMES, 1, [1, 1, 1, 1]);
    }
    if (sparks.length > 64) sparks.splice(0, sparks.length - 64);

    // ---- present ----
    B.end();

    // ---- HUD (full resolution) ----
    if (opt.hud) drawHUD({ frame, meta, cfg, me, X, Y, scale, pix });

    const ms = performance.now() - t0;
    state.stats.frameMs = ms;
    state.stats.frames++;
    state.stats.avgMs = state.stats.avgMs ? state.stats.avgMs * 0.95 + ms * 0.05 : ms;
  }

  // ResourceUIElement: rows of [tinted letter sprite at 2.5x][16 px]["NN/NN" monogram 55 px], right-aligned
  // 104 px from the right edge (measured on the original: the count's left edge 163 px from the right,
  // cap tops at 59 px then every 40 px), plus the round clock as a fifth row in the same font.
  function drawHUD(ctx) {
    const { frame, meta, cfg, me, X, Y, scale, pix } = ctx;
    const dpr = state.dpr;
    const W = hudCanvas.width, H = hudCanvas.height;
    const banked = me >= 0 && frame.players[me] ? frame.players[me].banked : null;
    const needs = me >= 0 && meta && meta.needs ? meta.needs[me] : null;
    const timeLimit = cfg.time_limit || 240;
    const left = Math.max(0, Math.ceil(timeLimit - frame.t));
    const key = [W, H, left, banked ? banked.map(v => Math.floor(v)).join(',') : '-', needs ? needs.join(',') : '-',
      state.fontReady, letters.map(l => !!l).join('')].join('|');
    const dirty = key !== state.hudKey;
    if (dirty) {
      state.hudKey = key;
      hctx.setTransform(1, 0, 0, 1, 0, 0);
      hctx.clearRect(0, 0, W, H);
      hctx.imageSmoothingEnabled = false;
      // monogram is a 16 px pixel font (crisp at 16, 32, 48 px; anti-aliased at 55): the counts are drawn
      // from a glyph atlas rendered at 16 px and scaled by whole device px, so every stroke is a clean block
      const k = Math.max(1, Math.round(3 * dpr));
      const rowPitch = Math.round(40 * dpr), capTop = Math.round(59 * dpr);
      const letterPx = Math.round(25 * dpr), gap = Math.round(16 * dpr);
      const textLeft = W - Math.round(163 * dpr);
      const atlas = glyphAtlas();
      if (banked && needs) {
        for (let i = 0; i < 4; i++) {
          const top = capTop + rowPitch * i;
          drawText(hctx, atlas, pad2(banked[i]) + '/' + pad2(needs[i]), textLeft, top, k, '#ffffff');
          drawLetter(hctx, i, textLeft - gap - letterPx, top, letterPx);
        }
      }
      // the round clock as a fifth row (the original has no clock; the HUD block is the one place the edge
      // indicators never reach)
      const mm = Math.floor(left / 60), ss = left % 60;
      drawText(hctx, atlas, `${mm}:${ss < 10 ? '0' : ''}${ss}`, textLeft, capTop + rowPitch * 4, k, left <= 30 ? PALETTE_CSS[3] : 'rgba(255,255,255,0.75)');
    }
    B.hud(hudCanvas, dirty);

    // off-screen mines of the local player's intent type: the intent letter at the view edge, 8 units in
    // (GameObjectRenderer::drawMineDirections); the own pad gets a ring icon the same way
    if (me >= 0 && frame.players[me]) {
      const intent = (frame.players[me].intent | 0) & 3;
      const pad = 8 * pix, L = Math.round(25 * dpr);
      // the original clamps the sprite's centre 8 units in from the view edge
      const edge = (sx, sy) => ({ x: Math.min(W - pad, Math.max(pad, sx)), y: Math.min(H - pad, Math.max(pad, sy)) });
      const shx = state.view.shx || 0, shy = state.view.shy || 0;
      if (meta && meta.mine_pos) {
        const mineR = (cfg.mine_radius || 0.08) * scale * pix;
        for (let m = 0; m < meta.mine_pos.length; m++) {
          if ((meta.mine_type ? meta.mine_type[m] : m) !== intent) continue;
          if (frame.alive && !frame.alive[m]) continue;
          const sx = X(meta.mine_pos[m][0]) * pix + shx, sy = Y(meta.mine_pos[m][1]) * pix + shy;   // device px
          if (sx + mineR > 0 && sx - mineR < W && sy + mineR > 0 && sy - mineR < H) continue;
          const e = edge(sx, sy);
          B.screenSprite('letter' + intent, Math.round(e.x - L / 2), Math.round(e.y - L / 2), L, L, 0, 0, 1, 1, rgba(PALETTE[intent], 1));
        }
      }
      if (meta && meta.pads && meta.pads[me] != null) {
        const padR = (cfg.pad_radius || 0.06);
        const R = frame.R != null ? frame.R : 1;
        const padRing = Math.max(R - padR - 0.02, 0.1);
        const a = meta.pads[me];
        const sx = X(padRing * Math.cos(a)) * pix + shx, sy = Y(padRing * Math.sin(a)) * pix + shy;
        const pr = padR * scale * pix;
        if (!(sx + pr > 0 && sx - pr < W && sy + pr > 0 && sy - pr < H)) {
          const e = edge(sx, sy);
          B.screenSprite('padicon', Math.round(e.x - L / 2), Math.round(e.y - L / 2), L, L, 0, 0, 1, 1, [1, 1, 1, 0.9]);
        }
      }
    }
  }

  // The HUD glyphs ("0123456789/:") rendered once at monogram's native 16 px (alpha thresholded so no
  // browser anti-aliasing survives), with the ink top of the digits so text can be placed by its cap top.
  const atlasCache = {};
  const GLYPHS = '0123456789/: ';
  function glyphAtlas() {
    const key = state.fontReady ? 'font' : 'fallback';
    if (atlasCache[key]) return atlasCache[key];
    const c = document.createElement('canvas');
    const cx = c.getContext('2d');
    const font = state.fontReady ? '16px monogram' : '16px "Courier New", monospace';
    cx.font = font;
    const adv = {};
    let total = 0;
    for (const ch of GLYPHS) { adv[ch] = Math.max(1, Math.ceil(cx.measureText(ch).width)); total += adv[ch] + 1; }
    c.width = total; c.height = 20;
    cx.font = font; cx.textBaseline = 'alphabetic'; cx.textAlign = 'left'; cx.fillStyle = '#fff';
    const pos = {};
    let x = 0;
    for (const ch of GLYPHS) { pos[ch] = x; cx.fillText(ch, x, 15); x += adv[ch] + 1; }
    const img = cx.getImageData(0, 0, c.width, c.height);
    const d = img.data;
    let top = 20, bottom = -1;
    for (let i = 3; i < d.length; i += 4) {
      d[i] = d[i] >= 96 ? 255 : 0;
      if (d[i]) { d[i - 3] = 255; d[i - 2] = 255; d[i - 1] = 255; const y = ((i - 3) / 4 / c.width) | 0; if (y < top) top = y; if (y > bottom) bottom = y; }
    }
    cx.putImageData(img, 0, 0);
    const a = { canvas: c, adv, pos, top: bottom >= 0 ? top : 0, h: bottom >= 0 ? bottom - top + 1 : 7 };
    atlasCache[key] = a;
    return a;
  }
  const tintCanvas = document.createElement('canvas');
  function drawText(c, atlas, text, x, capTop, k, color) {
    // tint: draw the white glyphs into a scratch canvas and multiply by the colour
    const w = [...text].reduce((acc, ch) => acc + (atlas.adv[ch] || atlas.adv['0']) * k, 0);
    const h = atlas.canvas.height * k;
    if (tintCanvas.width < w || tintCanvas.height < h) { tintCanvas.width = Math.max(tintCanvas.width, w); tintCanvas.height = Math.max(tintCanvas.height, h); }
    const tc = tintCanvas.getContext('2d');
    tc.setTransform(1, 0, 0, 1, 0, 0);
    tc.globalCompositeOperation = 'source-over';
    tc.clearRect(0, 0, tintCanvas.width, tintCanvas.height);
    tc.imageSmoothingEnabled = false;
    let px = 0;
    for (const ch of text) {
      const g = atlas.pos[ch] != null ? ch : '0';
      tc.drawImage(atlas.canvas, atlas.pos[g], 0, atlas.adv[g], atlas.canvas.height, px, 0, atlas.adv[g] * k, h);
      px += atlas.adv[g] * k;
    }
    tc.globalCompositeOperation = 'source-in';
    tc.fillStyle = color; tc.fillRect(0, 0, w, h);
    tc.globalCompositeOperation = 'source-over';
    c.imageSmoothingEnabled = false;
    c.drawImage(tintCanvas, 0, 0, w, h, x, capTop - atlas.top * k, w, h);
  }

  const tintedLetters = {};
  function drawLetter(c, i, x, y, size) {
    const img = letters[i];
    if (!img) {
      c.save(); c.fillStyle = PALETTE_CSS[i]; c.textAlign = 'center'; c.textBaseline = 'middle';
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
    resize, draw, ready, project,
    get mode() { return B.mode; },
    get view() { return state.view; },
    get sparks() { return state.sparks.length; },   // live spark count (tests)
    stats: state.stats,
    camera: state.cam,
    hudCanvas,
    finish() { B.finish(); },   // gl.finish(): wait for the GPU (benchmarks)
    destroy() { B.destroy(); },
  };
  return renderer;
}
