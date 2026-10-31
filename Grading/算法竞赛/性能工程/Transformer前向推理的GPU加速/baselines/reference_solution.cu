// reference_solution.cu: the baseline of the Transformer task.
// FP32 throughout: tiles in shared memory for every matrix product, causal attention fused with an online softmax,
// no tensor core. Very long sequences are split over several launches so no launch comes near the driver's timeout.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct TransformerSpec { int B, S, D, H, F, L; const float* weights; };

namespace {

const int THREADS = 256;
const int SHARED_BYTES = 88 * 1024;
const double LAUNCH_FMA = 2e12;   // about a quarter of a second of attention per launch
constexpr float NEVER = -1e30f;

// Shared-memory tiles have fixed power-of-two sizes, so an item number splits into its two indices by shift and mask;
// integer division in a thread is expensive on the GPU. Threads work on groups of 4 rows and 4 columns: each loaded
// float4 then serves many multiplications, and that is what limits the speed, not the arithmetic.
const int LIN_LOG = 6;            // tokens and input columns of a matrix-product tile
const int LIN_N = 1 << LIN_LOG;
const int COL_LOG = 4;            // output columns of a tile, in float4
const int COL_N = 1 << COL_LOG;

struct LayerWeights {
    size_t g1, b1, wq, wk, wv, bq, bk, bv, wo, bo, g2, b2, w1, f1, w2, f2;
};

int round4(int v) { return (v + 3) & ~3; }

void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "solution: %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

// Scratch and weights of the current init(). Strides are rounded up to whole float4: Ds for x, Dh for the heads
// (each head padded to dp), Fs for the feed-forward width. Padding always holds zero.
TransformerSpec g_spec;
const float* g_x;
float* g_y;
int g_Ds, g_Dh, g_Fs, g_d, g_dp, g_tile_log, g_head_log;
bool g_direct;
float* g_arena = nullptr;
float* g_cur = nullptr;
float* g_u = nullptr;
float* g_q = nullptr;
float* g_k = nullptr;
float* g_v = nullptr;
float* g_o = nullptr;
float* g_t = nullptr;
float* g_m = nullptr;
float* g_a = nullptr;
std::vector<LayerWeights> g_layers;
size_t g_last_g, g_last_b;

// exp(v) for v <= 0 by 2^(v / ln 2): relative error below 3e-6, and zero below -87.
__device__ __forceinline__ float fast_exp(float v) {
    const float t = fmaxf(v, -87.0f) * 1.44269504089f;
    const float whole = floorf(t);
    const float f = t - whole;
    const float p = 1.0000025f + f * (0.69300663f + f * (0.2414275f + f * (0.05203743f + f * 0.013520603f)));
    const float r = p * __int_as_float(((int)whole + 127) << 23);
    return v < -87.0f ? 0.0f : r;
}

__device__ __forceinline__ float4 fast_exp4(const float4& a, const float4& top) {
    return make_float4(fast_exp(a.x - top.x), fast_exp(a.y - top.y), fast_exp(a.z - top.z), fast_exp(a.w - top.w));
}

__device__ __forceinline__ float gelu(float t) { return 0.5f * t * erfcf(-0.70710678118654752f * t); }

// o += a * v, element by element per row of the group.
__device__ __forceinline__ void add_scaled(float4& o, float a, const float4& v) {
    o.x += a * v.x;
    o.y += a * v.y;
    o.z += a * v.z;
    o.w += a * v.w;
}

__device__ __forceinline__ void add_dot(float& e, const float4& a, const float4& b) {
    e += a.x * b.x;
    e += a.y * b.y;
    e += a.z * b.z;
    e += a.w * b.w;
}

__global__ void copy_kernel(const float* src, int Ss, float* dst, int Ds, int D, int tokens) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;
    for (int k = 0; k < D; k++) dst[t * Ds + k] = src[t * Ss + k];
}

__global__ void layer_norm_kernel(const float* src, int Ss, float* dst, int Ds, int D, const float* gamma,
                                  const float* beta, int tokens) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tokens) return;
    const float* z = src + t * Ss;
    float* o = dst + t * Ds;
    float m = 0;
    for (int k = 0; k < D; k++) m += z[k];
    m /= D;
    float v = 0;
    for (int k = 0; k < D; k++) {
        const float c = z[k] - m;
        v += c * c;
    }
    const float scale = 1.0f / sqrtf(v / D + 1e-5f);
    for (int k = 0; k < D; k++) o[k] = gamma[k] * ((z[k] - m) * scale) + beta[k];
}

// out[t] = post(in[t] W + b), added to the old out when ADD is set. W has Is rows of Os columns.
// Tile x of the grid is the tile of tokens, y the tile of output columns.
template <bool GELU, bool ADD>
__global__ void matmul_kernel(const float* in, int Is, const float4* W, const float4* b, float4* out, int Os, int tokens) {
    extern __shared__ __align__(16) float smem[];
    const int nt = blockDim.x, tid = threadIdx.x;
    const int t0 = blockIdx.x * LIN_N, nT = min(LIN_N, tokens - t0);
    const int Os4 = Os / 4, k0 = blockIdx.y * COL_N, K4 = min(COL_N, Os4 - k0);
    float* Z = smem;                                  // [input column][token]
    float4* Wt = (float4*)(smem + LIN_N * LIN_N);     // [input column][output float4]
    float4* Acc = Wt + LIN_N * COL_N;                 // [token][output float4]
    for (int n = tid; n < LIN_N * COL_N; n += nt) {
        const int k = n & (COL_N - 1);
        if ((n >> COL_LOG) < nT && k < K4) Acc[n] = b[k0 + k];
    }
    for (int m0 = 0; m0 < Is; m0 += LIN_N) {
        const int M = min(LIN_N, Is - m0);
        for (int n = tid; n < LIN_N * LIN_N; n += nt) {
            const int m = n >> LIN_LOG, t = n & (LIN_N - 1);
            if (t < nT && m < M) Z[n] = in[(t0 + t) * Is + m0 + m];
        }
        for (int n = tid; n < LIN_N * COL_N; n += nt) {
            const int m = n >> COL_LOG, k = n & (COL_N - 1);
            if (m < M && k < K4) Wt[n] = W[(m0 + m) * Os4 + k0 + k];
        }
        __syncthreads();
        for (int n = tid; n < COL_N * COL_N; n += nt) {
            const int tg = n >> COL_LOG, k = n & (COL_N - 1);
            if (k >= K4 || (tg << 2) >= nT) continue;
            float4* acc = Acc + (tg << (COL_LOG + 2)) + k;
            float4 o0 = acc[0], o1 = acc[COL_N], o2 = acc[2 * COL_N], o3 = acc[3 * COL_N];
            const float4* z = (const float4*)Z + tg;
            const float4* w = Wt + k;
            for (int m = 0; m < M; m++, z += LIN_N / 4, w += COL_N) {
                const float4 zm = *z, wm = *w;
                add_scaled(o0, zm.x, wm);
                add_scaled(o1, zm.y, wm);
                add_scaled(o2, zm.z, wm);
                add_scaled(o3, zm.w, wm);
            }
            acc[0] = o0;
            acc[COL_N] = o1;
            acc[2 * COL_N] = o2;
            acc[3 * COL_N] = o3;
        }
        __syncthreads();
    }
    for (int n = tid; n < LIN_N * COL_N; n += nt) {
        const int t = n >> COL_LOG, k = n & (COL_N - 1);
        if (t >= nT || k >= K4) continue;
        float4 o = Acc[n];
        if (GELU) o = make_float4(gelu(o.x), gelu(o.y), gelu(o.z), gelu(o.w));
        float4* dst = out + (t0 + t) * Os4 + k0 + k;
        if (ADD) {
            const float4 p = *dst;
            o = make_float4(p.x + o.x, p.y + o.y, p.z + o.z, p.w + o.w);
        }
        *dst = o;
    }
}

// One launch of the attention: the earlier tokens j in [j0, j1) of every sequence and head, against every tile of
// output tokens they can reach. Tile x of the grid is the output tile, y is sequence * H + head. The tile is at least
// 4 tokens. Only the widest heads, which would not fit, read K and V straight from global memory (DIRECT).
template <bool DIRECT>
__global__ void attention_kernel(const float4* Q, const float4* K, const float4* V, float4* O, float* M_, float* A_,
                                 int S, int H, int Dh4, int d4, int tile_log, int head_log, int first_tile, int j0,
                                 int j1, float scale) {
    extern __shared__ __align__(16) float smem[];
    const int nt = blockDim.x, tid = threadIdx.x;
    const int T = 1 << tile_log, Hs = 1 << head_log, group_log = tile_log - 2, NG = 1 << group_log;
    const int b = blockIdx.y / H, h = blockIdx.y % H;
    const int i0 = (first_tile + blockIdx.x) << tile_log, nI = min(T, S - i0);
    const int base = b * S, head = h * d4, state = blockIdx.y * S + i0;
    const int cells = T << head_log, group_cells = NG << head_log, column_cells = NG << tile_log;
    float4* sQ = (float4*)smem;                       // [token][float4 of the head]
    float4* sO = sQ + cells;
    float4* sK = sO + cells;
    float4* sV = DIRECT ? sK : sK + cells;
    float4* E = DIRECT ? sK : sV + cells;             // [group of 4 tokens][earlier token]
    float4* W = E + column_cells;
    float4* sM = W + column_cells;                    // per group of 4 tokens
    float4* sA = sM + NG;
    float4* sF = sA + NG;

    for (int n = tid; n < cells; n += nt) {
        const int i = n >> head_log, c = n & (Hs - 1);
        if (i >= nI || c >= d4) continue;
        sQ[n] = Q[(base + i0 + i) * Dh4 + head + c];
        sO[n] = j0 == 0 ? make_float4(0, 0, 0, 0) : O[(base + i0 + i) * Dh4 + head + c];
    }
    if (tid < T) {
        float* m = (float*)sM;
        float* a = (float*)sA;
        const bool fresh = j0 == 0 || tid >= nI;
        m[tid] = fresh ? NEVER : M_[state + tid];
        a[tid] = fresh ? 0.0f : A_[state + tid];
    }
    for (int jt = j0; jt < j1 && jt < i0 + nI; jt += T) {
        const int nJ = min(T, j1 - jt);
        const float4* Kj = K + (base + jt) * Dh4 + head;
        const float4* Vj = V + (base + jt) * Dh4 + head;
        if (!DIRECT) {
            for (int n = tid; n < cells; n += nt) {
                const int j = n >> head_log, c = n & (Hs - 1);
                if (j >= nJ || c >= d4) continue;
                sK[n] = Kj[j * Dh4 + c];
                sV[n] = Vj[j * Dh4 + c];
            }
        }
        __syncthreads();
        // Scores: 4 output tokens against 4 earlier tokens at a time.
        for (int n = tid; n < NG * NG; n += nt) {
            const int ig = n >> group_log, jg = n & (NG - 1);
            const int ir = ig << 2, jr = jg << 2;
            if (ir >= nI || jr >= nJ) continue;
            const float4* q0 = sQ + (ir << head_log);
            const float4* q1 = q0 + Hs;
            const float4* q2 = q1 + Hs;
            const float4* q3 = q2 + Hs;
            const float4* k0 = DIRECT ? Kj + jr * Dh4 : sK + (jr << head_log);
            const int kstep = DIRECT ? Dh4 : Hs;
            const float4* k1 = k0 + kstep;
            const float4* k2 = k1 + kstep;
            const float4* k3 = k2 + kstep;
            float4 e0 = make_float4(0, 0, 0, 0), e1 = e0, e2 = e0, e3 = e0;
            for (int c = 0; c < d4; c++) {
                const float4 a0 = q0[c], a1 = q1[c], a2 = q2[c], a3 = q3[c];
                const float4 b0 = k0[c], b1 = k1[c], b2 = k2[c], b3 = k3[c];
                add_dot(e0.x, a0, b0); add_dot(e0.y, a1, b0); add_dot(e0.z, a2, b0); add_dot(e0.w, a3, b0);
                add_dot(e1.x, a0, b1); add_dot(e1.y, a1, b1); add_dot(e1.z, a2, b1); add_dot(e1.w, a3, b1);
                add_dot(e2.x, a0, b2); add_dot(e2.y, a1, b2); add_dot(e2.z, a2, b2); add_dot(e2.w, a3, b2);
                add_dot(e3.x, a0, b3); add_dot(e3.y, a1, b3); add_dot(e3.z, a2, b3); add_dot(e3.w, a3, b3);
            }
            float4 es[4] = {e0, e1, e2, e3};
            const bool clean = jt + jr + 3 <= i0 + ir && jr + 3 < nJ && ir + 3 < nI;
            for (int s = 0; s < 4; s++) {
                float4 e = make_float4(es[s].x * scale, es[s].y * scale, es[s].z * scale, es[s].w * scale);
                if (!clean) {
                    // Later tokens and positions beyond a short tile take no part.
                    const int j = jr + s, at = jt + j, base_i = i0 + ir;
                    const bool out = j >= nJ;
                    if (out || at > base_i || ir >= nI) e.x = NEVER;
                    if (out || at > base_i + 1 || ir + 1 >= nI) e.y = NEVER;
                    if (out || at > base_i + 2 || ir + 2 >= nI) e.z = NEVER;
                    if (out || at > base_i + 3 || ir + 3 >= nI) e.w = NEVER;
                }
                E[(ig << tile_log) + jr + s] = e;
            }
        }
        __syncthreads();
        // Online softmax: a new maximum rescales what has been summed so far.
        if (tid < NG) {
            const float4* row = E + (tid << tile_log);
            float4 top = make_float4(NEVER, NEVER, NEVER, NEVER);
            for (int j = 0; j < nJ; j++) {
                const float4 e = row[j];
                top = make_float4(fmaxf(top.x, e.x), fmaxf(top.y, e.y), fmaxf(top.z, e.z), fmaxf(top.w, e.w));
            }
            const float4 old = sM[tid], acc = sA[tid];
            const float4 best = make_float4(fmaxf(old.x, top.x), fmaxf(old.y, top.y), fmaxf(old.z, top.z), fmaxf(old.w, top.w));
            const float4 f = fast_exp4(old, best);
            sF[tid] = f;
            sM[tid] = best;
            sA[tid] = make_float4(acc.x * f.x, acc.y * f.y, acc.z * f.z, acc.w * f.w);
        }
        __syncthreads();
        for (int n = tid; n < column_cells; n += nt)
            if ((n & (T - 1)) < nJ) W[n] = fast_exp4(E[n], sM[n >> tile_log]);
        __syncthreads();
        if (tid < NG) {
            const float4* row = W + (tid << tile_log);
            float4 sum = sA[tid];
            for (int j = 0; j < nJ; j++) {
                const float4 w = row[j];
                sum = make_float4(sum.x + w.x, sum.y + w.y, sum.z + w.z, sum.w + w.w);
            }
            sA[tid] = sum;
        }
        // Output: 4 output tokens, one float4 of the head, over all the earlier tokens of the tile.
        for (int n = tid; n < group_cells; n += nt) {
            const int ig = n >> head_log, c = n & (Hs - 1);
            if (c >= d4 || (ig << 2) >= nI) continue;
            const float4 f = sF[ig];
            float4* acc = sO + (ig << (head_log + 2)) + c;
            float4 o0 = acc[0], o1 = acc[Hs], o2 = acc[2 * Hs], o3 = acc[3 * Hs];
            o0 = make_float4(o0.x * f.x, o0.y * f.x, o0.z * f.x, o0.w * f.x);
            o1 = make_float4(o1.x * f.y, o1.y * f.y, o1.z * f.y, o1.w * f.y);
            o2 = make_float4(o2.x * f.z, o2.y * f.z, o2.z * f.z, o2.w * f.z);
            o3 = make_float4(o3.x * f.w, o3.y * f.w, o3.z * f.w, o3.w * f.w);
            const float4* row = W + (ig << tile_log);
            const float4* v = DIRECT ? Vj + c : sV + c;
            const int vstep = DIRECT ? Dh4 : Hs;
            for (int j = 0; j < nJ; j++, v += vstep) {
                const float4 w = row[j], vj = *v;
                add_scaled(o0, w.x, vj);
                add_scaled(o1, w.y, vj);
                add_scaled(o2, w.z, vj);
                add_scaled(o3, w.w, vj);
            }
            acc[0] = o0;
            acc[Hs] = o1;
            acc[2 * Hs] = o2;
            acc[3 * Hs] = o3;
        }
        __syncthreads();
    }
    // A tile whose last token has seen all its earlier tokens is finished: divide by the running sum.
    const bool done = i0 + nI <= j1;
    const float* a = (const float*)sA;
    for (int n = tid; n < cells; n += nt) {
        const int i = n >> head_log, c = n & (Hs - 1);
        if (i >= nI || c >= d4) continue;
        float4 o = sO[n];
        if (done) {
            const float f = 1.0f / a[i];
            o = make_float4(o.x * f, o.y * f, o.z * f, o.w * f);
        }
        O[(base + i0 + i) * Dh4 + head + c] = o;
    }
    if (!done && tid < nI) {
        M_[state + tid] = ((const float*)sM)[tid];
        A_[state + tid] = a[tid];
    }
}

float* alloc_floats(size_t n) {
    float* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(float)), "cannot allocate scratch memory");
    check(cudaMemset(p, 0, n * sizeof(float)), "cannot clear scratch memory");
    return p;
}

void release() {
    for (float** p : {&g_arena, &g_cur, &g_u, &g_q, &g_k, &g_v, &g_o, &g_t, &g_m, &g_a}) {
        if (*p) check(cudaFree(*p), "cannot free scratch memory");
        *p = nullptr;
    }
    g_layers.clear();
}

template <bool GELU, bool ADD>
void matmul(const float* in, int Is, size_t W, size_t b, float* out, int Os) {
    const int tokens = g_spec.B * g_spec.S;
    const dim3 grid((tokens + LIN_N - 1) / LIN_N, (Os / 4 + COL_N - 1) / COL_N);
    matmul_kernel<GELU, ADD><<<grid, THREADS, SHARED_BYTES>>>(in, Is, (const float4*)(g_arena + W),
                                                               (const float4*)(g_arena + b), (float4*)out, Os, tokens);
}

void layer_norm(const float* src, int Ss, float* dst, int Ds, size_t gamma, size_t beta) {
    const int tokens = g_spec.B * g_spec.S;
    layer_norm_kernel<<<(tokens + THREADS - 1) / THREADS, THREADS>>>(src, Ss, dst, Ds, g_spec.D, g_arena + gamma,
                                                                      g_arena + beta, tokens);
}

void attention() {
    const int S = g_spec.S, B = g_spec.B, H = g_spec.H, T = 1 << g_tile_log;
    const float scale = 1.0f / sqrtf((float)g_d);
    const int last_tile = (S - 1) >> g_tile_log;
    for (int j0 = 0; j0 < S;) {
        const double per_j = (double)(S - j0) * B * H * 2 * g_d;
        const int span = std::max(T, (int)std::min(LAUNCH_FMA / per_j / T, (double)S) * T);
        const int j1 = (int)std::min((long long)S, (long long)j0 + span);
        const int first_tile = j0 >> g_tile_log;
        const dim3 grid(last_tile - first_tile + 1, B * H);
        if (g_direct) {
            attention_kernel<true><<<grid, THREADS, SHARED_BYTES>>>(
                (const float4*)g_q, (const float4*)g_k, (const float4*)g_v, (float4*)g_o, g_m, g_a, S, H, g_Dh / 4,
                g_dp / 4, g_tile_log, g_head_log, first_tile, j0, j1, scale);
        } else {
            attention_kernel<false><<<grid, THREADS, SHARED_BYTES>>>(
                (const float4*)g_q, (const float4*)g_k, (const float4*)g_v, (float4*)g_o, g_m, g_a, S, H, g_Dh / 4,
                g_dp / 4, g_tile_log, g_head_log, first_tile, j0, j1, scale);
        }
        j0 = j1;
    }
}

}  // namespace

void init(const TransformerSpec& spec, const float* x, float* y) {
    release();
    g_spec = spec;
    g_x = x;
    g_y = y;
    const int D = spec.D, F = spec.F, H = spec.H;
    g_d = D / H;
    g_dp = round4(g_d);
    g_Ds = round4(D);
    g_Dh = H * g_dp;
    g_Fs = round4(F);
    g_head_log = 0;
    while ((1 << g_head_log) < g_dp / 4) g_head_log++;
    // Q, O, K and V of a tile, the scores and weights (4 bytes per pair each), and 3 values per token.
    auto tile_bytes = [&](int log, int arrays) {
        return (arrays * 16 << log << g_head_log) + (8 << (2 * log)) + (12 << log);
    };
    g_tile_log = 6;
    while (g_tile_log > 2 && tile_bytes(g_tile_log, 4) > SHARED_BYTES) g_tile_log--;
    g_direct = tile_bytes(g_tile_log, 4) > SHARED_BYTES;
    check(cudaFuncSetAttribute(matmul_kernel<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES), "shared memory");
    check(cudaFuncSetAttribute(matmul_kernel<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES), "shared memory");
    check(cudaFuncSetAttribute(matmul_kernel<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES), "shared memory");
    check(cudaFuncSetAttribute(attention_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES), "shared memory");
    check(cudaFuncSetAttribute(attention_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_BYTES), "shared memory");

    // Weights with their rows and columns moved to the padded positions.
    std::vector<float> arena;
    const float* w = spec.weights;
    auto head_pos = [&](int k) { return (k / g_d) * g_dp + k % g_d; };
    auto same_pos = [](int k) { return k; };
    auto vec = [&](int n, int stride, auto pos) {
        const size_t at = arena.size();
        arena.resize(at + stride, 0.0f);
        for (int k = 0; k < n; k++) arena[at + pos(k)] = *w++;
        return at;
    };
    auto mat = [&](int rows, int cols, int Is, int Os, auto row_pos, auto col_pos) {
        const size_t at = arena.size();
        arena.resize(at + (size_t)Is * Os, 0.0f);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) arena[at + (size_t)row_pos(r) * Os + col_pos(c)] = *w++;
        return at;
    };
    for (int l = 0; l < spec.L; l++) {
        LayerWeights lw;
        lw.g1 = vec(D, g_Ds, same_pos);
        lw.b1 = vec(D, g_Ds, same_pos);
        lw.wq = mat(D, D, g_Ds, g_Dh, same_pos, head_pos);
        lw.wk = mat(D, D, g_Ds, g_Dh, same_pos, head_pos);
        lw.wv = mat(D, D, g_Ds, g_Dh, same_pos, head_pos);
        lw.bq = vec(D, g_Dh, head_pos);
        lw.bk = vec(D, g_Dh, head_pos);
        lw.bv = vec(D, g_Dh, head_pos);
        lw.wo = mat(D, D, g_Dh, g_Ds, head_pos, same_pos);
        lw.bo = vec(D, g_Ds, same_pos);
        lw.g2 = vec(D, g_Ds, same_pos);
        lw.b2 = vec(D, g_Ds, same_pos);
        lw.w1 = mat(D, F, g_Ds, g_Fs, same_pos, same_pos);
        lw.f1 = vec(F, g_Fs, same_pos);
        lw.w2 = mat(F, D, g_Fs, g_Ds, same_pos, same_pos);
        lw.f2 = vec(D, g_Ds, same_pos);
        g_layers.push_back(lw);
    }
    g_last_g = vec(D, g_Ds, same_pos);
    g_last_b = vec(D, g_Ds, same_pos);
    check(cudaMalloc(&g_arena, arena.size() * sizeof(float)), "cannot allocate weights");
    check(cudaMemcpy(g_arena, arena.data(), arena.size() * sizeof(float), cudaMemcpyHostToDevice), "cannot upload weights");

    const size_t tokens = (size_t)spec.B * spec.S;
    g_cur = alloc_floats(tokens * g_Ds);
    g_u = alloc_floats(tokens * g_Ds);
    g_q = alloc_floats(tokens * g_Dh);
    g_k = alloc_floats(tokens * g_Dh);
    g_v = alloc_floats(tokens * g_Dh);
    g_o = alloc_floats(tokens * g_Dh);
    g_t = alloc_floats(tokens * g_Fs);
    g_m = alloc_floats(tokens * H);
    g_a = alloc_floats(tokens * H);
}

void forward() {
    const int tokens = g_spec.B * g_spec.S, D = g_spec.D;
    copy_kernel<<<(tokens + THREADS - 1) / THREADS, THREADS>>>(g_x, D, g_cur, g_Ds, D, tokens);
    for (const LayerWeights& lw : g_layers) {
        layer_norm(g_cur, g_Ds, g_u, g_Ds, lw.g1, lw.b1);
        matmul<false, false>(g_u, g_Ds, lw.wq, lw.bq, g_q, g_Dh);
        matmul<false, false>(g_u, g_Ds, lw.wk, lw.bk, g_k, g_Dh);
        matmul<false, false>(g_u, g_Ds, lw.wv, lw.bv, g_v, g_Dh);
        attention();
        matmul<false, true>(g_o, g_Dh, lw.wo, lw.bo, g_cur, g_Ds);
        layer_norm(g_cur, g_Ds, g_u, g_Ds, lw.g2, lw.b2);
        matmul<true, false>(g_u, g_Ds, lw.w1, lw.f1, g_t, g_Fs);
        matmul<false, true>(g_t, g_Fs, lw.w2, lw.f2, g_cur, g_Ds);
    }
    layer_norm(g_cur, g_Ds, g_y, D, g_last_g, g_last_b);
}
