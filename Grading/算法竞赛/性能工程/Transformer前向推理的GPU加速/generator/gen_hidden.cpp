// gen_hidden.cpp: hidden inputs of the Transformer task and their double-precision answers.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o gen_hidden.exe gen_hidden.cpp
// Usage, from the task folder: gen_hidden.exe [case ...]   (cases like p01 or c07; no argument: all 31)
// Writes performance/ and correctness/, each with inputs/input_NN.bin and expected/expected_NN.bin.
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct Spec {
    const char* name;
    int B, S, D, H, F, L;
    uint64_t seed;
    bool mixed_x;  // x is not standard normal: constant, huge and tiny tokens
};

// 8 families, 3 instances each, numbered by family. Every instance outside the very long family does at least
// 800 GFLOP by L(2BS^2D + 8BSD^2 + 4BSDF), so that the timed run stays well above the noise of short measurements.
static const Spec PERFORMANCE[] = {
    {"p01", 3900, 64, 256, 8, 256, 4, 0x51c7a30e91d4a601ULL, false},   // large batch, short sequences
    {"p02", 7000, 128, 128, 4, 128, 4, 0x2f6b81d4c07e3902ULL, false},
    {"p03", 10000, 128, 128, 4, 128, 4, 0x9ae03d57b16c8403ULL, false},
    {"p04", 128, 128, 1024, 4, 1024, 4, 0x74d29c60ea3f1b04ULL, false},  // high dimension
    {"p05", 48, 256, 1024, 8, 2048, 4, 0xc85f13b7420d9e05ULL, false},
    {"p06", 32, 128, 2048, 16, 2048, 4, 0x06a7e49d35b1c206ULL, false},
    {"p07", 144, 2048, 128, 4, 128, 4, 0xe13c58a26f07d907ULL, false},   // long sequences
    {"p08", 40, 4096, 128, 8, 128, 4, 0x3bd0f672c9a54e08ULL, false},
    {"p09", 4, 16384, 128, 4, 128, 4, 0x8f24c1e07d36ba09ULL, false},
    {"p10", 1, 100000, 1024, 16, 1024, 2, 0x5ae97b032cd8f110ULL, false},  // very long sequences
    {"p11", 2, 50000, 512, 8, 512, 2, 0xd2016f8ea47c3b11ULL, false},
    {"p12", 1, 200000, 256, 8, 256, 1, 0x47b8e3d15f90a612ULL, false},
    {"p13", 1200, 512, 128, 16, 128, 4, 0xa96c02f7d3e18b13ULL, false},  // narrow heads
    {"p14", 430, 1024, 128, 32, 128, 4, 0x1e75d84ab60c9f14ULL, false},
    {"p15", 1100, 1024, 64, 8, 64, 4, 0xf30a6c927be1d415ULL, false},
    {"p16", 380, 512, 256, 1, 256, 4, 0x68c1b5f04ed72a16ULL, false},    // few wide heads
    {"p17", 430, 1024, 128, 1, 128, 4, 0xb74e29d3a1068c17ULL, false},
    {"p18", 110, 512, 512, 2, 512, 4, 0x2c9f70e6d85b3a18ULL, false},
    {"p19", 200, 512, 128, 4, 128, 24, 0x95d3a8176ec04f19ULL, false},   // deep
    {"p20", 38, 1024, 256, 8, 256, 16, 0x7e20c4b9f1a65d20ULL, false},
    {"p21", 1200, 256, 64, 4, 64, 32, 0xc06b1fe2583d9a21ULL, false},
    {"p22", 144, 512, 256, 8, 1024, 6, 0x3a8df6107cb24e22ULL, false},   // common shape, F = 4D
    {"p23", 13, 2048, 384, 6, 1536, 6, 0xe5714c0a9d3fb623ULL, false},
    {"p24", 18, 1024, 512, 8, 2048, 6, 0x10fb97d2c6e8a524ULL, false},
};

static const Spec CORRECTNESS[] = {
    {"c01", 3, 1, 64, 4, 128, 2, 0x8d1e60b7f2a43c01ULL, false},    // S = 1
    {"c02", 1, 2, 30, 3, 17, 3, 0x46f2ca09e7d15b02ULL, false},     // B = 1, S = 2, D and F not multiples of 4
    {"c03", 4, 32, 16, 16, 32, 2, 0xb3097de1a65c8f03ULL, false},   // H = D
    {"c04", 2, 64, 2048, 1, 64, 2, 0x29e8c5f4037db104ULL, false},  // H = 1 with the largest D, F < D
    {"c05", 2, 128, 90, 6, 25, 1, 0xf6a13b7e0c82d905ULL, false},   // L = 1, F != D
    {"c06", 2, 5000, 32, 4, 32, 2, 0x71cd04e9b8f36a06ULL, false},  // medium S, crosses any tile boundary
    {"c07", 2, 64, 64, 4, 64, 2, 0xda583f2c6b019e07ULL, true},     // constant, huge and tiny tokens
};

// Weights as in the statement's sample, checked before anything is written.
static const float SAMPLE_WEIGHTS[53] = {
    1.0f, 0.875f, 0.125f, 0.0f, -0.75f, 1.0f, -0.875f, -0.25f, -0.125f, 0.875f,
    -0.125f, -0.375f, -0.625f, -0.875f, 0.0f, -0.125f, -0.25f, -0.125f, 0.25f, 0.125f,
    -0.125f, 0.125f, 0.375f, -0.125f, -0.625f, -0.875f, 0.125f, -0.125f, 1.125f, 1.0f,
    0.0f, 0.125f, 0.25f, 0.75f, 0.375f, 0.5f, 0.375f, 0.5f, 0.0f, -0.25f,
    0.25f, 0.0f, -0.375f, -0.875f, 0.625f, -0.75f, 0.125f, 0.125f, -0.25f, 1.125f,
    0.875f, 0.125f, -0.125f,
};
static const float SAMPLE_X[6] = {-1.0f, -1.125f, -0.5f, -1.375f, -0.75f, 2.0f};
static const float SAMPLE_Y[6] = {1.249621f, -0.999705f, 1.249914f, -0.999933f, -0.999995f, 0.749996f};

static void die(const std::string& msg) {
    fprintf(stderr, "gen_hidden: %s\n", msg.c_str());
    exit(1);
}

// Random values: 64-bit Mersenne Twister with hand-written uniform and normal so the files do not depend on the library.
struct Random {
    std::mt19937_64 engine;
    explicit Random(uint64_t seed) : engine(seed) {}
    double unit() { return (double)(engine() >> 11) * (1.0 / 9007199254740992.0); }
    double uniform(double r) { return (2.0 * unit() - 1.0) * r; }
    double normal() {
        double u = unit();
        while (u <= 0.0) u = unit();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * unit());
    }
};

static size_t weight_count(const Spec& s) {
    const size_t D = s.D, F = s.F;
    return (size_t)s.L * (9 * D + 4 * D * D + 2 * D * F + F) + 2 * D;
}

static void generate(const Spec& s, std::vector<float>& w, std::vector<float>& x) {
    Random rnd(s.seed);
    w.clear();
    w.reserve(weight_count(s));
    const double rD = 1.0 / std::sqrt((double)s.D), rF = 1.0 / std::sqrt((double)s.F);
    auto add = [&](size_t n, double r) { for (size_t i = 0; i < n; i++) w.push_back((float)rnd.uniform(r)); };
    auto gamma = [&](size_t n) { for (size_t i = 0; i < n; i++) w.push_back((float)(1.0 + rnd.uniform(0.1))); };
    const size_t D = s.D, F = s.F;
    for (int l = 0; l < s.L; l++) {
        gamma(D); add(D, 0.1);                               // gamma1, beta1
        add(D * D, 3 * rD); add(D * D, 3 * rD); add(D * D, rD);  // W_Q, W_K, W_V
        add(3 * D, 0.1);                                     // b_Q, b_K, b_V
        add(D * D, rD); add(D, 0.1);                         // W_O, b_O
        gamma(D); add(D, 0.1);                               // gamma2, beta2
        add(D * F, rD); add(F, 0.1);                         // W_1, b_1
        add(F * D, rF); add(D, 0.1);                         // W_2, b_2
    }
    gamma(D); add(D, 0.1);                                   // gamma, beta
    if (w.size() != weight_count(s)) die("weight count mismatch");
    x.resize((size_t)s.B * s.S * s.D);
    size_t pos = 0;
    for (int b = 0; b < s.B; b++)
        for (int i = 0; i < s.S; i++) {
            const int kind = s.mixed_x ? (i + b) % 4 : -1;
            for (int k = 0; k < s.D; k++, pos++) {
                if (kind == 0) x[pos] = 1.5f;
                else if (kind == 1) x[pos] = -0.25f;
                else if (kind == 2) x[pos] = (float)(1000.0 * rnd.normal());
                else if (kind == 3) x[pos] = (float)(0.001 * rnd.normal());
                else x[pos] = (float)rnd.normal();
            }
        }
}

// exp(x) for x <= 0, x below -700 treated as -700; relative error about 1e-15.
static inline __m256d exp256(__m256d x) {
    x = _mm256_max_pd(x, _mm256_set1_pd(-700.0));
    const __m256d n = _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(1.4426950408889634)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(n, _mm256_set1_pd(0.693147180369123816490), x);
    r = _mm256_fnmadd_pd(n, _mm256_set1_pd(1.90821492927058770002e-10), r);
    __m256d p = _mm256_set1_pd(1.0 / 39916800.0);
    static const double C[] = {1.0 / 3628800, 1.0 / 362880, 1.0 / 40320, 1.0 / 5040, 1.0 / 720, 1.0 / 120, 1.0 / 24,
                               1.0 / 6, 0.5, 1.0, 1.0};
    for (double c : C) p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(c));
    const __m256i e = _mm256_slli_epi64(
        _mm256_add_epi64(_mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(n)), _mm256_set1_epi64x(1023)), 52);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(e));
}

static void layer_norm(const Spec& s, const double* src, double* dst, const float* gamma, const float* beta) {
    const long long tokens = (long long)s.B * s.S;
    const int D = s.D;
#pragma omp parallel for schedule(static)
    for (long long t = 0; t < tokens; t++) {
        const double* z = src + t * D;
        double* o = dst + t * D;
        double m = 0, v = 0;
        for (int k = 0; k < D; k++) m += z[k];
        m /= D;
        for (int k = 0; k < D; k++) v += (z[k] - m) * (z[k] - m);
        const double scale = 1.0 / std::sqrt(v / D + 1e-5);
        for (int k = 0; k < D; k++) o[k] = (double)gamma[k] * ((z[k] - m) * scale) + (double)beta[k];
    }
}

// dst[t][k] = sum_m src[t][m] W[m][k] + b[k], added to the old dst when add is set; W is In x Out, row by row.
static void linear(long long tokens, const double* src, int In, const float* W, const float* b, double* dst, int Out,
                   bool add) {
#pragma omp parallel
    {
        std::vector<double> row(Out);
#pragma omp for schedule(static)
        for (long long t = 0; t < tokens; t++) {
            const double* z = src + t * In;
            for (int k = 0; k < Out; k++) row[k] = (double)b[k];
            for (int m = 0; m < In; m++) {
                const double v = z[m];
                const float* w = W + (size_t)m * Out;
#pragma omp simd
                for (int k = 0; k < Out; k++) row[k] += v * (double)w[k];
            }
            double* o = dst + t * Out;
            if (add) for (int k = 0; k < Out; k++) o[k] += row[k];
            else for (int k = 0; k < Out; k++) o[k] = row[k];
        }
    }
}

// Causal multi-head attention, one tile of tokens against one tile of earlier tokens at a time, with an online
// softmax per output token: running maximum m, running sum a of exp(e - m), running vector v of exp(e - m) V.
static void attention(const Spec& s, const double* Q, const double* K, const double* V, double* O) {
    const int S = s.S, D = s.D, H = s.H, d = D / H;
    const int IB = 64, JB = std::max(16, std::min(1024, (1 << 20) / (16 * d)));
    const double scale = 1.0 / std::sqrt((double)d);
    const long long tiles = (long long)s.B * H * ((S + IB - 1) / IB);
#pragma omp parallel
    {
        std::vector<double> kt((size_t)JB * d), vt((size_t)JB * d), e(JB + 4), w(JB + 4), a(IB), m(IB), acc((size_t)IB * d);
#pragma omp for schedule(dynamic, 8)
        for (long long tile = 0; tile < tiles; tile++) {
            const int I = (int)(tile % ((S + IB - 1) / IB));
            const long long bh = tile / ((S + IB - 1) / IB);
            const int b = (int)(bh / H), h = (int)(bh % H);
            const size_t seq = (size_t)b * S * D;
            const int i0 = I * IB, i1 = std::min(S, i0 + IB);
            std::fill(acc.begin(), acc.end(), 0.0);
            std::fill(a.begin(), a.end(), 0.0);
            std::fill(m.begin(), m.end(), -std::numeric_limits<double>::infinity());
            for (int j0 = 0; j0 < i1; j0 += JB) {
                const int j1 = std::min(i1, j0 + JB);
                for (int j = j0; j < j1; j++) {
                    const size_t at = seq + (size_t)j * D + (size_t)h * d;
                    std::copy(K + at, K + at + d, kt.begin() + (size_t)(j - j0) * d);
                    std::copy(V + at, V + at + d, vt.begin() + (size_t)(j - j0) * d);
                }
                for (int i = i0; i < i1; i++) {
                    const int n = std::min(i + 1, j1) - j0;
                    if (n <= 0) continue;
                    const double* q = Q + seq + (size_t)i * D + (size_t)h * d;
                    double top = -std::numeric_limits<double>::infinity();
                    for (int j = 0; j < n; j++) {
                        const double* kj = kt.data() + (size_t)j * d;
                        double t = 0;
#pragma omp simd reduction(+ : t)
                        for (int c = 0; c < d; c++) t += q[c] * kj[c];
                        e[j] = t * scale;
                        top = std::max(top, e[j]);
                    }
                    double* v = acc.data() + (size_t)(i - i0) * d;
                    double& ai = a[i - i0];
                    double& mi = m[i - i0];
                    if (top > mi) {
                        const double f = std::exp(mi - top);
                        ai *= f;
                        for (int c = 0; c < d; c++) v[c] *= f;
                        mi = top;
                    }
                    const __m256d base = _mm256_set1_pd(mi);
                    for (int j = 0; j < n; j += 4)
                        _mm256_storeu_pd(w.data() + j, exp256(_mm256_sub_pd(_mm256_loadu_pd(e.data() + j), base)));
                    for (int j = 0; j < n; j++) {
                        const double wj = w[j];
                        const double* vj = vt.data() + (size_t)j * d;
                        ai += wj;
#pragma omp simd
                        for (int c = 0; c < d; c++) v[c] += wj * vj[c];
                    }
                }
            }
            for (int i = i0; i < i1; i++) {
                const double f = 1.0 / a[i - i0];
                const double* v = acc.data() + (size_t)(i - i0) * d;
                double* o = O + seq + (size_t)i * D + (size_t)h * d;
                for (int c = 0; c < d; c++) o[c] = v[c] * f;
            }
        }
    }
}

static std::vector<double> solve(const Spec& s, const std::vector<float>& weights, const std::vector<float>& input) {
    const size_t n = input.size();
    const long long tokens = (long long)s.B * s.S;
    const int D = s.D, F = s.F;
    const size_t D2 = (size_t)D * D, DF = (size_t)D * F;
    std::vector<double> x(input.begin(), input.end()), u(n), Q(n), K(n), V(n), O(n), T((size_t)tokens * F);
    const float* w = weights.data();
    auto take = [&](size_t count) { const float* p = w; w += count; return p; };
    for (int l = 0; l < s.L; l++) {
        const float* g1 = take(D); const float* b1 = take(D);
        const float* wq = take(D2); const float* wk = take(D2); const float* wv = take(D2);
        const float* bq = take(D); const float* bk = take(D); const float* bv = take(D);
        const float* wo = take(D2); const float* bo = take(D);
        const float* g2 = take(D); const float* b2 = take(D);
        const float* w1 = take(DF); const float* f1 = take(F);
        const float* w2 = take(DF); const float* f2 = take(D);
        layer_norm(s, x.data(), u.data(), g1, b1);
        linear(tokens, u.data(), D, wq, bq, Q.data(), D, false);
        linear(tokens, u.data(), D, wk, bk, K.data(), D, false);
        linear(tokens, u.data(), D, wv, bv, V.data(), D, false);
        attention(s, Q.data(), K.data(), V.data(), O.data());
        linear(tokens, O.data(), D, wo, bo, x.data(), D, true);
        layer_norm(s, x.data(), u.data(), g2, b2);
        linear(tokens, u.data(), D, w1, f1, T.data(), F, false);
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)T.size(); i++) T[i] *= 0.5 * std::erfc(-T[i] * 0.70710678118654752440);
        linear(tokens, T.data(), F, w2, f2, x.data(), D, true);
    }
    const float* g = take(D); const float* b = take(D);
    layer_norm(s, x.data(), u.data(), g, b);
    return u;
}

static void write_case(const Spec& s, const std::string& folder, const std::string& number) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> weights, input;
    generate(s, weights, input);
    std::filesystem::create_directories(folder + "/inputs");
    std::filesystem::create_directories(folder + "/expected");
    const std::string in_path = folder + "/inputs/input_" + number + ".bin";
    const std::string out_path = folder + "/expected/expected_" + number + ".bin";
    FILE* f = fopen(in_path.c_str(), "wb");
    if (!f) die("cannot write " + in_path);
    const int32_t head[6] = {s.B, s.S, s.D, s.H, s.F, s.L};
    fwrite(head, sizeof(int32_t), 6, f);
    fwrite(weights.data(), sizeof(float), weights.size(), f);
    if (fwrite(input.data(), sizeof(float), input.size(), f) != input.size()) die("short write " + in_path);
    fclose(f);
    const std::vector<double> y = solve(s, weights, input);
    std::vector<float> out(y.begin(), y.end());
    f = fopen(out_path.c_str(), "wb");
    if (!f) die("cannot write " + out_path);
    if (fwrite(out.data(), sizeof(float), out.size(), f) != out.size()) die("short write " + out_path);
    fclose(f);
    double top = 0;
    for (float v : out) top = std::max(top, (double)std::fabs(v));
    const double D = s.D, F = s.F, BS = (double)s.B * s.S;
    const double gflop = s.L * (2 * BS * s.S * D + 8 * BS * D * D + 4 * BS * D * F) / 1e9;
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    printf("%s %s %d/%d/%d/%d/%d/%d gflop=%.0f weights=%zu x=%zu max_abs_y=%.4f seconds=%.1f\n", s.name, number.c_str(),
           s.B, s.S, s.D, s.H, s.F, s.L, gflop, weights.size(), input.size(), top, sec);
    fflush(stdout);
}

static void self_check() {
    double xs[4] = {-1e-9, -1.0, -37.5, -699.0};
    __m256d r = exp256(_mm256_loadu_pd(xs));
    double out[4];
    _mm256_storeu_pd(out, r);
    for (int i = 0; i < 4; i++)
        if (std::fabs(out[i] / std::exp(xs[i]) - 1.0) > 1e-13) die("exp256 is inaccurate");
    const Spec sample{"sample", 1, 3, 2, 2, 3, 1, 0, false};
    const std::vector<double> y = solve(sample, std::vector<float>(SAMPLE_WEIGHTS, SAMPLE_WEIGHTS + 53),
                                        std::vector<float>(SAMPLE_X, SAMPLE_X + 6));
    for (int i = 0; i < 6; i++)
        if (std::fabs(y[i] - SAMPLE_Y[i]) > 1e-5) die("the solver does not reproduce the statement's sample");
}

int main(int argc, char** argv) {
    self_check();
    std::vector<std::string> wanted(argv + 1, argv + argc);
    auto selected = [&](const char* name) {
        return wanted.empty() || std::find(wanted.begin(), wanted.end(), name) != wanted.end();
    };
    int i = 0;
    for (const Spec& s : CORRECTNESS) {
        i++;
        if (selected(s.name)) write_case(s, "correctness", (i < 10 ? "0" : "") + std::to_string(i));
    }
    i = 0;
    for (const Spec& s : PERFORMANCE) {
        i++;
        if (selected(s.name)) write_case(s, "performance", (i < 10 ? "0" : "") + std::to_string(i));
    }
    return 0;
}
