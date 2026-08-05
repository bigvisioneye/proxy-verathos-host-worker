// Fused Goldilocks sumcheck-round kernels for the proof-v3 native tier.
// Field: p = 2^64 - 2^32 + 1, elements as uint64 bit patterns in int64.
// Byte-identical semantics to the torch/python reference paths.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#define GLSTREAM at::cuda::getCurrentCUDAStream()

#define GL_P   0xFFFFFFFF00000001ULL
#define GL_EPS 0xFFFFFFFFULL

__device__ __forceinline__ uint64_t gl_reduce128(uint64_t hi, uint64_t lo) {
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo = hi & GL_EPS;
    uint64_t t0 = lo - hi_hi;
    if (lo < hi_hi) t0 -= GL_EPS;
    uint64_t t1 = hi_lo * GL_EPS;
    uint64_t res = t0 + t1;
    if (res < t1) res += GL_EPS;
    if (res >= GL_P) res -= GL_P;
    return res;
}

__device__ __forceinline__ uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return gl_reduce128(hi, lo);
}

__device__ __forceinline__ uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a) s += GL_EPS;
    if (s >= GL_P) s -= GL_P;
    return s;
}

__device__ __forceinline__ uint64_t gl_sub(uint64_t a, uint64_t b) {
    uint64_t d = a - b;
    if (a < b) d -= GL_EPS;
    if (d >= GL_P) d -= GL_P;
    return d;
}

// One fused sumcheck round over pair layout (lo half | hi half):
// g0 += x_lo*f_lo ; g1 += x_hi*f_hi ; g2 += (2x_hi-x_lo)*(2f_hi-f_lo).
// Block-level shared-memory reduction; one partial triple per block.
__global__ void k_round_partials(const uint64_t* x, const uint64_t* f,
                                 uint64_t* partials, int64_t half) {
    __shared__ uint64_t s0[256], s1[256], s2[256];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t g0 = 0, g1 = 0, g2 = 0;
    for (int64_t idx = i; idx < half;
         idx += (int64_t)gridDim.x * blockDim.x) {
        uint64_t xl = x[idx], xh = x[half + idx];
        uint64_t fl = f[idx], fh = f[half + idx];
        g0 = gl_add(g0, gl_mul(xl, fl));
        g1 = gl_add(g1, gl_mul(xh, fh));
        uint64_t x2 = gl_sub(gl_add(xh, xh), xl);
        uint64_t f2 = gl_sub(gl_add(fh, fh), fl);
        g2 = gl_add(g2, gl_mul(x2, f2));
    }
    s0[threadIdx.x] = g0; s1[threadIdx.x] = g1; s2[threadIdx.x] = g2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s0[threadIdx.x] = gl_add(s0[threadIdx.x], s0[threadIdx.x + stride]);
            s1[threadIdx.x] = gl_add(s1[threadIdx.x], s1[threadIdx.x + stride]);
            s2[threadIdx.x] = gl_add(s2[threadIdx.x], s2[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x * 3 + 0] = s0[0];
        partials[blockIdx.x * 3 + 1] = s1[0];
        partials[blockIdx.x * 3 + 2] = s2[0];
    }
}

// Elementwise field ops: one launch instead of composed torch limb ops.
__global__ void k_gl_mul_ew(const uint64_t* a, const uint64_t* b,
                            uint64_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_mul(a[i], b[i]);
}

__global__ void k_gl_add_ew(const uint64_t* a, const uint64_t* b,
                            uint64_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_add(a[i], b[i]);
}

__global__ void k_gl_sub_ew(const uint64_t* a, const uint64_t* b,
                            uint64_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_sub(a[i], b[i]);
}

// RLC join for the batched D-chain: out = acc + rho * src.
__global__ void k_gl_axpy(const uint64_t* acc, const uint64_t* src,
                          uint64_t rho, uint64_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_add(acc[i], gl_mul(rho, src[i]));
}

// One COMPLETE FRI fold step per element (batched D-chain):
//   even_raw = lo + hi
//   odd_raw  = (lo - hi) * (bp[i] * scale)
//   out[i]   = (even_raw + c*(odd_raw - even_raw)) * inv2
// which equals (1-c)*(lo+hi)/2 + c*(lo-hi)/(2*s*g^i) -- the verifier
// fold equation.  Exact modular arithmetic: byte-identical to the
// composed add/sub/mul/lerp/scale sequence in any op order.
__global__ void k_gl_fold_step(const uint64_t* lo, const uint64_t* hi,
                               const uint64_t* bp, uint64_t scale,
                               uint64_t c, uint64_t inv2,
                               uint64_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t l = lo[i], h = hi[i];
    uint64_t even_raw = gl_add(l, h);
    uint64_t odd_raw = gl_mul(gl_sub(l, h), gl_mul(bp[i], scale));
    uint64_t lerp = gl_add(
        even_raw, gl_mul(c, gl_sub(odd_raw, even_raw)));
    out[i] = gl_mul(lerp, inv2);
}

__global__ void k_gl_mul_scalar(const uint64_t* a, uint64_t c,
                                uint64_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_mul(a[i], c);
}

__device__ __constant__ uint32_t FS_K256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ uint32_t fs_rotr(uint32_t x, int n){return (x>>n)|(x<<(32-n));}

// Byte-exact SHA-256 for messages of len <= 200 bytes (multi-block).
__device__ void sha256(const uint8_t* msg, int len, uint8_t* out) {
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    uint8_t block[256]; int total = ((len + 8) / 64 + 1) * 64;
    for (int i=0;i<total;i++) block[i]=0;
    for (int i=0;i<len;i++) block[i]=msg[i];
    block[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for (int i=0;i<8;i++) block[total-1-i]=(bits>>(8*i))&0xff;
    for (int off=0; off<total; off+=64) {
        uint32_t w[64];
        for (int i=0;i<16;i++)
            w[i]=(block[off+4*i]<<24)|(block[off+4*i+1]<<16)|(block[off+4*i+2]<<8)|block[off+4*i+3];
        for (int i=16;i<64;i++){
            uint32_t s0=fs_rotr(w[i-15],7)^fs_rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=fs_rotr(w[i-2],17)^fs_rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i=0;i<64;i++){
            uint32_t S1=fs_rotr(e,6)^fs_rotr(e,11)^fs_rotr(e,25);
            uint32_t ch=(e&f)^((~e)&g);
            uint32_t t1=hh+S1+ch+FS_K256[i]+w[i];
            uint32_t S0=fs_rotr(a,2)^fs_rotr(a,13)^fs_rotr(a,22);
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+maj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    for (int i=0;i<8;i++){out[4*i]=(h[i]>>24)&0xff;out[4*i+1]=(h[i]>>16)&0xff;out[4*i+2]=(h[i]>>8)&0xff;out[4*i+3]=h[i]&0xff;}
}

// ---- cross-argument batched sumcheck: one launch drives round r of
// ALL same-shape arguments (rows of a matrix), each with its OWN
// Fiat-Shamir transcript.  Grid = args x blocks-per-arg.

__global__ void k_round_partials_b(const uint64_t* x, const uint64_t* f,
                                   uint64_t* partials, int64_t half,
                                   int blocks_per_arg) {
    int arg = blockIdx.x / blocks_per_arg;
    int blk = blockIdx.x % blocks_per_arg;
    const uint64_t* xa = x + (int64_t)arg * half * 2;
    const uint64_t* fa = f + (int64_t)arg * half * 2;
    __shared__ uint64_t s0[256], s1[256], s2[256];
    uint64_t g0 = 0, g1 = 0, g2 = 0;
    for (int64_t i = (int64_t)blk * blockDim.x + threadIdx.x; i < half;
         i += (int64_t)blocks_per_arg * blockDim.x) {
        uint64_t xl = xa[i], xh = xa[half + i];
        uint64_t fl = fa[i], fh = fa[half + i];
        uint64_t x2 = gl_add(xh, gl_sub(xh, xl));
        uint64_t f2 = gl_add(fh, gl_sub(fh, fl));
        g0 = gl_add(g0, gl_mul(xl, fl));
        g1 = gl_add(g1, gl_mul(xh, fh));
        g2 = gl_add(g2, gl_mul(x2, f2));
    }
    s0[threadIdx.x] = g0; s1[threadIdx.x] = g1; s2[threadIdx.x] = g2;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s0[threadIdx.x] = gl_add(s0[threadIdx.x], s0[threadIdx.x + stride]);
            s1[threadIdx.x] = gl_add(s1[threadIdx.x], s1[threadIdx.x + stride]);
            s2[threadIdx.x] = gl_add(s2[threadIdx.x], s2[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[(int64_t)blockIdx.x * 3 + 0] = s0[0];
        partials[(int64_t)blockIdx.x * 3 + 1] = s1[0];
        partials[(int64_t)blockIdx.x * 3 + 2] = s2[0];
    }
}

// One thread per ARGUMENT: reduce its block partials, absorb into its
// transcript, derive its challenge.
__global__ void k_fs_round_b(const uint64_t* partials, int blocks_per_arg,
                             uint8_t* transcripts,        // args x 32
                             const uint8_t* dom, int dom_len,
                             const uint8_t* label, int label_len,
                             int index, int n_args,
                             uint64_t* rounds_out,        // args x 4
                             uint64_t* challenges_out) {  // args
    int arg = blockIdx.x * blockDim.x + threadIdx.x;
    if (arg >= n_args) return;
    uint64_t g[4] = {0, 0, 0, 0};
    const uint64_t* p = partials + (int64_t)arg * blocks_per_arg * 3;
    for (int i = 0; i < blocks_per_arg * 3; i++)
        g[i % 3] = gl_add(g[i % 3], p[i]);
    uint64_t three = 3;
    g[3] = gl_add(gl_sub(gl_mul(three, g[2]), gl_mul(three, g[1])), g[0]);
    uint8_t* t = transcripts + arg * 32;
    uint8_t msg[64];
    for (int i = 0; i < 32; i++) msg[i] = t[i];
    for (int e = 0; e < 4; e++) {
        rounds_out[(int64_t)arg * 4 + e] = g[e];
        for (int b = 0; b < 8; b++)
            msg[32 + 8 * e + b] = (uint8_t)(g[e] >> (8 * b));
    }
    sha256(msg, 64, t);
    uint8_t dmsg[160];
    int off = 0;
    for (int i = 0; i < dom_len; i++) dmsg[off++] = dom[i];
    for (int i = 0; i < 32; i++) dmsg[off++] = t[i];
    for (int i = 0; i < label_len; i++) dmsg[off++] = label[i];
    dmsg[off + 0] = (uint8_t)(index);
    dmsg[off + 1] = (uint8_t)(index >> 8);
    dmsg[off + 2] = (uint8_t)(index >> 16);
    dmsg[off + 3] = (uint8_t)(index >> 24);
    uint8_t h[32];
    for (uint32_t counter = 0;; counter++) {
        dmsg[off + 4] = (uint8_t)(counter);
        dmsg[off + 5] = (uint8_t)(counter >> 8);
        dmsg[off + 6] = (uint8_t)(counter >> 16);
        dmsg[off + 7] = (uint8_t)(counter >> 24);
        sha256(dmsg, off + 8, h);
        uint64_t cand = 0;
        for (int b = 0; b < 8; b++) cand |= ((uint64_t)h[b]) << (8 * b);
        if (cand < 0xFFFFFFFF00000001ULL) {
            challenges_out[arg] = cand;
            return;
        }
    }
}

// Fold every row by ITS OWN challenge.
__global__ void k_lerp_fold_b(const uint64_t* x, uint64_t* out,
                              const uint64_t* challenges, int64_t half,
                              int blocks_per_arg) {
    int arg = blockIdx.x / blocks_per_arg;
    int blk = blockIdx.x % blocks_per_arg;
    uint64_t c = challenges[arg];
    const uint64_t* xa = x + (int64_t)arg * half * 2;
    uint64_t* oa = out + (int64_t)arg * half;
    for (int64_t i = (int64_t)blk * blockDim.x + threadIdx.x; i < half;
         i += (int64_t)blocks_per_arg * blockDim.x) {
        uint64_t lo = xa[i], hi = xa[half + i];
        oa[i] = gl_add(lo, gl_mul(c, gl_sub(hi, lo)));
    }
}

// Device Fiat-Shamir round: reduce raw block partials to the 4-eval
// wire, absorb into the SHA-256 transcript, derive the next challenge
// (rejection loop) - all in one 1-thread launch so sumcheck rounds
// enqueue with ZERO host syncs.
__global__ void k_fs_round(const uint64_t* partials_a, int n_a,
                           const uint64_t* partials_b, int n_b,
                           int mode,   // 2=linear 3=deg2 4=deg3 5=etf(a4-b3ext)
                           uint8_t* transcript,  // 32B in/out
                           const uint8_t* dom, int dom_len,
                           const uint8_t* label, int label_len,
                           int absorb_tag,       // 1: absorb label too
                           int index,
                           uint64_t* rounds_out,   // 4 evals
                           uint64_t* challenge_out) {
    if (threadIdx.x || blockIdx.x) return;
    uint64_t g[4] = {0, 0, 0, 0};
    if (mode == 4 || mode == 5) {
        for (int i = 0; i < n_a; i++) g[i % 4] = gl_add(g[i % 4], partials_a[i]);
    } else {
        for (int i = 0; i < n_a; i++) g[i % 3] = gl_add(g[i % 3], partials_a[i]);
        uint64_t three = 3, two = 2;
        if (mode == 2) {
            // linear cells: g2 = 2*g1 - g0, g3 = 3*g1 - 2*g0
            g[2] = gl_sub(gl_mul(two, g[1]), g[0]);
            g[3] = gl_sub(gl_mul(three, g[1]), gl_mul(two, g[0]));
        } else if (mode != 6) {
            // degree-2 wire: g3 = 3*g2 - 3*g1 + g0
            g[3] = gl_add(gl_sub(gl_mul(three, g[2]), gl_mul(three, g[1])), g[0]);
        }
    }
    if (mode == 5) {
        // etf: (deg3 a-part) minus (deg2 b-part extended)
        uint64_t m[4] = {0, 0, 0, 0};
        for (int i = 0; i < n_b; i++) m[i % 3] = gl_add(m[i % 3], partials_b[i]);
        uint64_t three = 3;
        m[3] = gl_add(gl_sub(gl_mul(three, m[2]), gl_mul(three, m[1])), m[0]);
        for (int e = 0; e < 4; e++) g[e] = gl_sub(g[e], m[e]);
    }
    uint8_t msg[112];
    int off = 0;
    int n_evals = (mode == 6) ? 3 : 4;
    for (int i = 0; i < 32; i++) msg[off++] = transcript[i];
    if (absorb_tag) for (int i = 0; i < label_len; i++) msg[off++] = label[i];
    for (int e = 0; e < 4; e++) rounds_out[e] = g[e];
    for (int e = 0; e < n_evals; e++) {
        for (int b = 0; b < 8; b++) msg[off++] = (uint8_t)(g[e] >> (8 * b));
    }
    sha256(msg, off, transcript);
    uint8_t dmsg[160];
    int doff = 0;
    for (int i = 0; i < dom_len; i++) dmsg[doff++] = dom[i];
    for (int i = 0; i < 32; i++) dmsg[doff++] = transcript[i];
    for (int i = 0; i < label_len; i++) dmsg[doff++] = label[i];
    dmsg[doff + 0] = (uint8_t)(index);
    dmsg[doff + 1] = (uint8_t)(index >> 8);
    dmsg[doff + 2] = (uint8_t)(index >> 16);
    dmsg[doff + 3] = (uint8_t)(index >> 24);
    uint8_t h[32];
    for (uint32_t counter = 0;; counter++) {
        dmsg[doff + 4] = (uint8_t)(counter);
        dmsg[doff + 5] = (uint8_t)(counter >> 8);
        dmsg[doff + 6] = (uint8_t)(counter >> 16);
        dmsg[doff + 7] = (uint8_t)(counter >> 24);
        sha256(dmsg, doff + 8, h);
        uint64_t cand = 0;
        for (int b = 0; b < 8; b++) cand |= ((uint64_t)h[b]) << (8 * b);
        if (cand < 0xFFFFFFFF00000001ULL) { *challenge_out = cand; return; }
    }
}

// commitment = sha256(prefix || raw_root); transcript = sha256(transcript || commitment)
__global__ void k_root_commit_absorb(const uint8_t* prefix, int prefix_len,
                                     const uint8_t* raw_root,
                                     uint8_t* commitment_out,
                                     uint8_t* transcript) {
    if (threadIdx.x || blockIdx.x) return;
    uint8_t msg[160];
    for (int i = 0; i < prefix_len; i++) msg[i] = prefix[i];
    for (int i = 0; i < 32; i++) msg[prefix_len + i] = raw_root[i];
    sha256(msg, prefix_len + 32, commitment_out);
    uint8_t tmsg[64];
    for (int i = 0; i < 32; i++) tmsg[i] = transcript[i];
    for (int i = 0; i < 32; i++) tmsg[32 + i] = commitment_out[i];
    sha256(tmsg, 64, transcript);
}

__global__ void k_lerp_fold_ptr(const uint64_t* x, uint64_t* out,
                                const uint64_t* c, int64_t half) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half) {
        uint64_t lo = x[i], hi = x[half + i];
        out[i] = gl_add(lo, gl_mul(*c, gl_sub(hi, lo)));
    }
}

// Fused lerp fold: out[i] = x_lo + c * (x_hi - x_lo), pair layout.
__global__ void k_lerp_fold(const uint64_t* x, uint64_t* out,
                            uint64_t c, int64_t half) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half) {
        uint64_t lo = x[i], hi = x[half + i];
        out[i] = gl_add(lo, gl_mul(c, gl_sub(hi, lo)));
    }
}

static inline uint64_t* uptr(torch::Tensor& t) {
    return reinterpret_cast<uint64_t*>(t.data_ptr<int64_t>());
}
static inline const uint64_t* cuptr(const torch::Tensor& t) {
    return reinterpret_cast<const uint64_t*>(t.data_ptr<int64_t>());
}

torch::Tensor round_partials(torch::Tensor x, torch::Tensor f) {
    int64_t half = x.numel() / 2;
    int blocks = (int)std::min<int64_t>(1024, (half + 255) / 256);
    auto partials = torch::zeros({blocks * 3},
        torch::dtype(torch::kInt64).device(x.device()));
    k_round_partials<<<blocks, 256, 0, GLSTREAM>>>(cuptr(x), cuptr(f), uptr(partials), half);
    return partials;
}

torch::Tensor lerp_fold(torch::Tensor x, int64_t c) {
    int64_t half = x.numel() / 2;
    auto out = torch::empty({half},
        torch::dtype(torch::kInt64).device(x.device()));
    k_lerp_fold<<<(unsigned)((half + 255) / 256), 256, 0, GLSTREAM>>>(
        cuptr(x), uptr(out), (uint64_t)c, half);
    return out;
}



// Degree-3 product-sumcheck round: three tables (a, b, f), partials for
// g(z) at z = 0..3 via linear extension per pair. One triple-quad per block.
__global__ void k_product_round_partials(const uint64_t* a, const uint64_t* b,
                                         const uint64_t* f, uint64_t* partials,
                                         int64_t half) {
    __shared__ uint64_t s[4][256];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t g[4] = {0, 0, 0, 0};
    for (int64_t idx = i; idx < half;
         idx += (int64_t)gridDim.x * blockDim.x) {
        uint64_t lo[3] = {a[idx], b[idx], f[idx]};
        uint64_t hi[3] = {a[half + idx], b[half + idx], f[half + idx]};
        for (int z = 0; z < 4; ++z) {
            uint64_t term = 1;
            for (int t = 0; t < 3; ++t) {
                uint64_t d = gl_sub(hi[t], lo[t]);
                uint64_t folded = lo[t];
                for (int rep = 0; rep < z; ++rep) folded = gl_add(folded, d);
                term = gl_mul(term, folded);
            }
            g[z] = gl_add(g[z], term);
        }
    }
    for (int z = 0; z < 4; ++z) s[z][threadIdx.x] = g[z];
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            for (int z = 0; z < 4; ++z)
                s[z][threadIdx.x] = gl_add(s[z][threadIdx.x],
                                           s[z][threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int z = 0; z < 4; ++z)
            partials[blockIdx.x * 4 + z] = s[z][0];
}

torch::Tensor product_round_partials(torch::Tensor a, torch::Tensor b,
                                     torch::Tensor f) {
    int64_t half = a.numel() / 2;
    int blocks = (int)std::min<int64_t>(1024, (half + 255) / 256);
    auto partials = torch::zeros({blocks * 4},
        torch::dtype(torch::kInt64).device(a.device()));
    k_product_round_partials<<<blocks, 256, 0, GLSTREAM>>>(
        cuptr(a), cuptr(b), cuptr(f), uptr(partials), half);
    return partials;
}



// ── Fused Goldilocks NTT for the commit/LDE path ─────────────────────
// Byte-identical to ntt_goldilocks_reference: coefficient-domain input is
// pre-scaled by shift^j (coset), bit-reversed, then radix-2 DIT butterflies
// with twiddles root^(len/span). root = principal root of unity of `size`.
__global__ void k_bitrev(const uint64_t* in, uint64_t* out, int64_t n, int bits) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // reverse the low `bits` bits of i
    uint64_t r = 0, x = (uint64_t)i;
    for (int b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
    out[r] = in[i];
}

__global__ void k_coset_scale(uint64_t* a, uint64_t shift, int64_t n) {
    // a[j] *= shift^j  (sequential powers via per-thread pow by repeated mul
    // is too slow; instead caller passes precomputed powers). This kernel
    // multiplies elementwise by a provided power table.
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = gl_mul(a[i], shift);  // placeholder; real path uses table
}

__global__ void k_coset_scale_table(uint64_t* a, const uint64_t* powers, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = gl_mul(a[i], powers[i]);
}

// One radix-2 DIT stage on the bit-reversed array. half = span/2.
__global__ void k_ntt_butterfly(uint64_t* a, const uint64_t* stage_tw,
                                int64_t n, int64_t span) {
    int64_t half = span >> 1;
    int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // butterfly idx
    if (k >= n / 2) return;
    int64_t blk = k / half;
    int64_t off = k % half;
    int64_t base = blk * span + off;
    uint64_t w = stage_tw[off];
    uint64_t u = a[base];
    uint64_t t = gl_mul(w, a[base + half]);
    a[base] = gl_add(u, t);
    a[base + half] = gl_sub(u, t);
}

void bitrev(torch::Tensor in, torch::Tensor out, int64_t bits) {
    int64_t n = in.numel();
    k_bitrev<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(cuptr(in), uptr(out), n, (int)bits);
}
void coset_scale_table(torch::Tensor a, torch::Tensor powers) {
    int64_t n = a.numel();
    k_coset_scale_table<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(uptr(a), cuptr(powers), n);
}
void fs_round(torch::Tensor partials, int64_t wire,
              torch::Tensor transcript, torch::Tensor dom_label,
              int64_t dom_len, int64_t label_len, int64_t index,
              torch::Tensor rounds_out, torch::Tensor challenge_out) {
    const uint8_t* dl = dom_label.data_ptr<uint8_t>();
    k_fs_round<<<1, 1, 0, GLSTREAM>>>(
        cuptr(partials), (int)partials.numel(), nullptr, 0, (int)wire,
        transcript.data_ptr<uint8_t>(), dl, (int)dom_len,
        dl + dom_len, (int)label_len, 0, (int)index,
        uptr(rounds_out), uptr(challenge_out));
}

void fs_round_v2(torch::Tensor partials_a, torch::Tensor partials_b,
                 int64_t mode, torch::Tensor transcript,
                 torch::Tensor dom_label, int64_t dom_len,
                 int64_t label_len, int64_t absorb_tag, int64_t index,
                 torch::Tensor rounds_out, torch::Tensor challenge_out) {
    const uint8_t* dl = dom_label.data_ptr<uint8_t>();
    const uint64_t* pb = partials_b.numel() ? cuptr(partials_b) : nullptr;
    k_fs_round<<<1, 1, 0, GLSTREAM>>>(
        cuptr(partials_a), (int)partials_a.numel(),
        pb, (int)partials_b.numel(), (int)mode,
        transcript.data_ptr<uint8_t>(), dl, (int)dom_len,
        dl + dom_len, (int)label_len, (int)absorb_tag, (int)index,
        uptr(rounds_out), uptr(challenge_out));
}

torch::Tensor round_partials_b(torch::Tensor x, torch::Tensor f,
                               int64_t n_args, int64_t blocks_per_arg) {
    int64_t half = x.numel() / n_args / 2;
    auto partials = torch::empty({n_args * blocks_per_arg * 3},
                                 x.options());
    k_round_partials_b<<<(unsigned)(n_args * blocks_per_arg), 256, 0,
                         GLSTREAM>>>(
        cuptr(x), cuptr(f), uptr(partials), half, (int)blocks_per_arg);
    return partials;
}

void fs_round_b(torch::Tensor partials, int64_t blocks_per_arg,
                torch::Tensor transcripts, torch::Tensor dom_label,
                int64_t dom_len, int64_t label_len, int64_t index,
                int64_t n_args, torch::Tensor rounds_out,
                torch::Tensor challenges_out) {
    const uint8_t* dl = dom_label.data_ptr<uint8_t>();
    k_fs_round_b<<<(unsigned)((n_args + 63) / 64), 64, 0, GLSTREAM>>>(
        cuptr(partials), (int)blocks_per_arg,
        transcripts.data_ptr<uint8_t>(), dl, (int)dom_len,
        dl + dom_len, (int)label_len, (int)index, (int)n_args,
        uptr(rounds_out), uptr(challenges_out));
}

torch::Tensor lerp_fold_b(torch::Tensor x, torch::Tensor challenges,
                          int64_t n_args, int64_t blocks_per_arg) {
    int64_t half = x.numel() / n_args / 2;
    auto out = torch::empty({n_args * half}, x.options());
    k_lerp_fold_b<<<(unsigned)(n_args * blocks_per_arg), 256, 0,
                    GLSTREAM>>>(
        cuptr(x), uptr(out), cuptr(challenges), half,
        (int)blocks_per_arg);
    return out;
}

void root_commit_absorb(torch::Tensor prefix, torch::Tensor raw_root,
                        torch::Tensor commitment_out,
                        torch::Tensor transcript) {
    k_root_commit_absorb<<<1, 1, 0, GLSTREAM>>>(
        prefix.data_ptr<uint8_t>(), (int)prefix.numel(),
        raw_root.data_ptr<uint8_t>(),
        commitment_out.data_ptr<uint8_t>(),
        transcript.data_ptr<uint8_t>());
}

torch::Tensor lerp_fold_ptr(torch::Tensor x, torch::Tensor c) {
    int64_t half = x.numel() / 2;
    auto out = torch::empty({half}, x.options());
    k_lerp_fold_ptr<<<(unsigned)((half + 255) / 256), 256, 0, GLSTREAM>>>(
        cuptr(x), uptr(out), cuptr(c), half);
    return out;
}

torch::Tensor gl_mul_ew(torch::Tensor a, torch::Tensor b) {
    auto ac = a.contiguous();
    auto out = torch::empty_like(ac);
    int64_t n = ac.numel();
    if (b.numel() == 1) {
        uint64_t c = (uint64_t)b.item<int64_t>();
        k_gl_mul_scalar<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(
            cuptr(ac), c, uptr(out), n);
    } else {
        auto bc = b.contiguous();
        k_gl_mul_ew<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(
            cuptr(ac), cuptr(bc), uptr(out), n);
    }
    return out;
}

void ntt_butterfly(torch::Tensor a, torch::Tensor stage_tw, int64_t span) {
    int64_t n = a.numel();
    k_ntt_butterfly<<<(unsigned)((n / 2 + 255) / 256), 256, 0, GLSTREAM>>>(
        uptr(a), cuptr(stage_tw), n, span);
}

torch::Tensor gl_add_ew(torch::Tensor a, torch::Tensor b) {
    auto ac = a.contiguous();
    auto bc = b.contiguous();
    auto out = torch::empty_like(ac);
    int64_t n = ac.numel();
    k_gl_add_ew<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(
        cuptr(ac), cuptr(bc), uptr(out), n);
    return out;
}

torch::Tensor gl_sub_ew(torch::Tensor a, torch::Tensor b) {
    auto ac = a.contiguous();
    auto bc = b.contiguous();
    auto out = torch::empty_like(ac);
    int64_t n = ac.numel();
    k_gl_sub_ew<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(
        cuptr(ac), cuptr(bc), uptr(out), n);
    return out;
}

// out (preallocated, contiguous) = acc + rho * src; scalars arrive as
// int64 two's-complement encodings of canonical field elements.
void gl_axpy_out(torch::Tensor acc, torch::Tensor src, int64_t rho,
                 torch::Tensor out) {
    int64_t n = acc.numel();
    k_gl_axpy<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(
        cuptr(acc), cuptr(src), (uint64_t)rho, uptr(out), n);
}

// out (preallocated, contiguous) = fold of (lo|hi) with twiddle table
// bp scaled by `scale`, challenge `c`, times inv2.
void gl_fold_step_out(torch::Tensor lo, torch::Tensor hi,
                      torch::Tensor bp, int64_t scale, int64_t c,
                      int64_t inv2, torch::Tensor out) {
    int64_t n = lo.numel();
    k_gl_fold_step<<<(unsigned)((n + 255) / 256), 256, 0, GLSTREAM>>>(
        cuptr(lo), cuptr(hi), cuptr(bp), (uint64_t)scale, (uint64_t)c,
        (uint64_t)inv2, uptr(out), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("round_partials", &round_partials);
    m.def("lerp_fold", &lerp_fold);
    m.def("product_round_partials", &product_round_partials);
    m.def("bitrev", &bitrev);
    m.def("coset_scale_table", &coset_scale_table);
    m.def("ntt_butterfly", &ntt_butterfly);
    m.def("gl_mul_ew", &gl_mul_ew);
    m.def("gl_add_ew", &gl_add_ew);
    m.def("gl_sub_ew", &gl_sub_ew);
    m.def("gl_axpy_out", &gl_axpy_out);
    m.def("gl_fold_step_out", &gl_fold_step_out);
    m.def("fs_round", &fs_round);
    m.def("fs_round_v2", &fs_round_v2);
    m.def("root_commit_absorb", &root_commit_absorb);
    m.def("round_partials_b", &round_partials_b);
    m.def("fs_round_b", &fs_round_b);
    m.def("lerp_fold_b", &lerp_fold_b);
    m.def("lerp_fold_ptr", &lerp_fold_ptr);
}
