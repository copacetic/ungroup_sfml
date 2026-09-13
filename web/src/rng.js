// rng.js - std::mt19937_64 plus the libstdc++ (GCC 13) distribution algorithms used by the
// canonical rules core (rl/native/ungroup.cpp), so that Game.reset(seed) in the browser draws the
// same needs, pads, intents and mine rotation as the C++ core seeded with the same value.
//
// Algorithm: the 64-bit Mersenne Twister (Matsumoto & Nishimura, 2000) exactly as specified by
// std::mersenne_twister_engine<uint_fast64_t, 64, 312, 156, 31, 0xb5026f5aa96619e9, 29,
// 0x5555555555555555, 17, 0x71d67fffeda60000, 37, 0xfff7eee000000000, 43, 6364136223846793005>.
// seed(s): mt[0] = s; mt[i] = 6364136223846793005 * (mt[i-1] ^ (mt[i-1] >> 62)) + i.
//
// Implementation notes and per-call cost (Node 22, one core):
// - The state lives in a BigUint64Array(312); the twist and the tempering use BigInt arithmetic
//   with BigInt.asUintN(64, ...) for the wrap-around. A raw draw costs about 0.4 us amortised
//   (the twist of 312 words is roughly 60 us every 312 draws; tempering is five BigInt ops).
//   The rules core draws only at reset (2n + 2 draws), at ejection of a still group (1 draw) and
//   at spills (units + 2 per spilled unit), so BigInt is far from the hot path.
// - canonical(): std::generate_canonical<double, 53>(mt19937_64) makes ONE draw (k = 1 because
//   the engine already provides 64 bits) and returns double(x) / 2^64 (the uint64 -> double
//   conversion rounds to nearest even, which Number(BigInt) also does); a result of exactly 1.0 is
//   clamped to the largest double below 1 as libstdc++ does since GCC 10 (LWG 2524).
// - uniform(a, b): std::uniform_real_distribution<double>(a, b) = canonical() * (b - a) + a.
// - uniformInt(a, b): std::uniform_int_distribution<int>(a, b) for a 64-bit engine, GCC 11+:
//   Lemire's nearly divisionless method with a 128-bit product (_S_nd<unsigned __int128>):
//   product = draw * range; low = product mod 2^64; if low < range, reject while low < (2^64 - range) mod range;
//   result = a + (product >> 64). For range = 4 this is simply the top two bits of one draw.
//   (When range == 2^64 the whole draw is returned; that case never occurs in the core.)
// - discrete(weights): std::discrete_distribution<int>: probabilities normalised by division by
//   their sum, cumulative sums by left fold, last cumulative forced to 1.0, then lower_bound on a
//   single canonical() draw.
// - roundHalfEven(x): std::nearbyint in the default rounding mode.

const N = 312, M = 156;
const MATRIX_A = 0xb5026f5aa96619e9n;
const UPPER_MASK = 0xffffffff80000000n;
const LOWER_MASK = 0x7fffffffn;
const MASK64 = 0xffffffffffffffffn;
const TWO64 = 18446744073709551616; // 2^64 as a double (exact)
const ONE_BELOW = 1 - Number.EPSILON / 2; // nextafter(1.0, 0.0)

export class RNG {
  constructor(seed = 5489n) {
    this.mt = new BigUint64Array(N);
    this.mti = N + 1;
    this.seed(seed);
  }

  // seed: number, bigint or decimal string; interpreted as an unsigned 64-bit value.
  seed(s) {
    let v = typeof s === 'bigint' ? s : BigInt(typeof s === 'string' ? s : Math.trunc(Number(s)));
    v = BigInt.asUintN(64, v);
    const mt = this.mt;
    mt[0] = v;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1];
      mt[i] = BigInt.asUintN(64, 6364136223846793005n * (prev ^ (prev >> 62n)) + BigInt(i));
    }
    this.mti = N;
  }

  _twist() {
    const mt = this.mt;
    let i = 0;
    for (; i < N - M; i++) {
      const x = (mt[i] & UPPER_MASK) | (mt[i + 1] & LOWER_MASK);
      mt[i] = mt[i + M] ^ (x >> 1n) ^ ((x & 1n) ? MATRIX_A : 0n);
    }
    for (; i < N - 1; i++) {
      const x = (mt[i] & UPPER_MASK) | (mt[i + 1] & LOWER_MASK);
      mt[i] = mt[i + (M - N)] ^ (x >> 1n) ^ ((x & 1n) ? MATRIX_A : 0n);
    }
    const x = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (x >> 1n) ^ ((x & 1n) ? MATRIX_A : 0n);
    this.mti = 0;
  }

  // One raw 64-bit draw as a BigInt in [0, 2^64).
  next() {
    if (this.mti >= N) this._twist();
    let y = this.mt[this.mti++];
    y ^= (y >> 29n) & 0x5555555555555555n;
    y ^= (y << 17n) & 0x71d67fffeda60000n;
    y ^= (y << 37n) & 0xfff7eee000000000n;
    y ^= y >> 43n;
    return BigInt.asUintN(64, y);
  }

  // std::generate_canonical<double, 53>: one draw / 2^64 in [0, 1).
  canonical() {
    const r = Number(this.next()) / TWO64;
    return r >= 1 ? ONE_BELOW : r;
  }

  // std::uniform_real_distribution<double>(a, b)
  uniform(a, b) {
    return this.canonical() * (b - a) + a;
  }

  // std::uniform_int_distribution<int>(a, b) with a 64-bit engine (GCC 11+, Lemire with a 128-bit product).
  uniformInt(a, b) {
    const range = BigInt(b) - BigInt(a) + 1n; // > 0, <= 2^32 for int parameters
    if (range > MASK64) return a + Number(this.next()); // range == 2^64 (unreachable for int)
    let product = this.next() * range;
    let low = product & MASK64;
    if (low < range) {
      const threshold = BigInt.asUintN(64, -range) % range;
      while (low < threshold) {
        product = this.next() * range;
        low = product & MASK64;
      }
    }
    return a + Number(product >> 64n);
  }

  // std::discrete_distribution<int>(weights) sampled once. `weights` is an array of non-negative
  // numbers (at least two entries; the core always passes four).
  discrete(weights) {
    const n = weights.length;
    if (n < 2) return 0;
    let sum = 0;
    for (let k = 0; k < n; k++) sum += weights[k];
    // libstdc++: probabilities = w / sum, cumulative by partial_sum, last forced to 1.0
    const cp = new Float64Array(n);
    let acc = 0;
    for (let k = 0; k < n; k++) {
      const p = weights[k] / sum;
      acc = k === 0 ? p : acc + p;
      cp[k] = acc;
    }
    cp[n - 1] = 1.0;
    const p = this.canonical();
    // lower_bound: first index with cp[k] >= p
    let lo = 0, hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cp[mid] < p) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
}

// std::nearbyint in FE_TONEAREST: round half to even.
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return (f % 2 === 0) ? f : f + 1;
}
