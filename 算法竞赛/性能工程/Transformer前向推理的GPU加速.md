# 题目

本题是一道性能优化工程题：在结果正确的前提下，让你的程序尽可能快。采用什么优化策略、怎样实现，由你自己决定。

越接近性能极限，同样的提速在计分中的分量越大。请尽可能做出更好的结果。作答过程中，如果某个方向的方案难以继续推进，可以考虑换一个思路。

## 一、问题

用 CUDA 在 GPU 上实现一个 Transformer 的前向计算。输入 $x$ 是 $B$ 条序列，每条 $S$ 个 token，每个 token $D$ 维，形状 $[B,S,D]$，32 位浮点数。模型有 $L$ 层，每层一个 $H$ 头因果自注意力和一个隐藏宽度为 $F$ 的前馈网络，$H$ 整除 $D$，记 $d=D/H$。输出 $y$ 形状同 $x$。

下面的 token 都是行向量，每条序列单独计算，不同序列共用同一组权重。

**线性层**：$\mathrm{Lin}(z;W,b)=zW+b$。

**层归一化**：对一个 token $z$，

$$
\mathrm{LN}(z;\gamma,\beta)=\gamma\circ\frac{z-m}{\sqrt{v+10^{-5}}}+\beta,\qquad m=\frac1D\sum_j z_j,\quad v=\frac1D\sum_j(z_j-m)^2.
$$

**每一层**依次做下面两步，$x$ 更新后进入下一步：

1. 因果自注意力。令 $u=\mathrm{LN}(x;\gamma_1,\beta_1)$，$Q=\mathrm{Lin}(u;W_Q,b_Q)$，$K=\mathrm{Lin}(u;W_K,b_K)$，$V=\mathrm{Lin}(u;W_V,b_V)$。把每个 token 的 $D$ 维依次切成 $H$ 段，每段 $d$ 维，对第 $h$ 段（记 $Q_i^h$ 为位置 $i$ 的这一段，$K$、$V$ 同理）和每个位置 $i$：

$$
e_{ij}^h=\frac{Q_i^h\cdot K_j^h}{\sqrt d},\qquad \alpha_{ij}^h=\frac{\exp(e_{ij}^h)}{\sum_{j'\le i}\exp(e_{ij'}^h)}\ (j\le i),\qquad O_i^h=\sum_{j\le i}\alpha_{ij}^hV_j^h.
$$

   把各段 $O_i^h$ 按 $h$ 的顺序拼回 $D$ 维得到 $O_i$，然后 $x\leftarrow x+\mathrm{Lin}(O;W_O,b_O)$。

2. 前馈网络。令 $u=\mathrm{LN}(x;\gamma_2,\beta_2)$，然后

$$
x\leftarrow x+\mathrm{Lin}\big(g(\mathrm{Lin}(u;W_1,b_1));W_2,b_2\big),\qquad g(t)=t\,\Phi(t),
$$

   其中 $\Phi$ 是标准正态分布函数，$g$ 逐元素作用。

$L$ 层之后，$y=\mathrm{LN}(x;\gamma,\beta)$。

## 二、权重与测试文件格式

**权重顺序**。权重是一个连续的 `float` 数组。逐层依次存放下表各项，$L$ 层之后再存末尾的 $\gamma,\beta$。矩阵按行存放，$W_1$ 是 $D\times F$，$W_2$ 是 $F\times D$，其余矩阵是 $D\times D$。$b_1$ 长度为 $F$，其余向量长度为 $D$。

| 顺序 | 项 |
|---|---|
| 1 | $\gamma_1,\beta_1$ |
| 2 | $W_Q,W_K,W_V$ |
| 3 | $b_Q,b_K,b_V$ |
| 4 | $W_O,b_O$ |
| 5 | $\gamma_2,\beta_2$ |
| 6 | $W_1,b_1$ |
| 7 | $W_2,b_2$ |

共 $L(9D+4D^2+2DF+F)+2D$ 个数。

**测试文件**是二进制文件，由评测程序读取，解法不需要自己读文件。内容依次为：6 个 `int32` 的 $B,S,D,H,F,L$；上述权重；$x$，共 $BSD$ 个 `float`，按 $B$、$S$、$D$ 从外到内的顺序存放。数值都是小端序。

## 三、函数接口

提交的 `solution.cu` 中实现下面两个函数：

```cpp
struct TransformerSpec { int B, S, D, H, F, L; const float* weights; };

void init(const TransformerSpec& spec, const float* x, float* y);
void forward();
```

- `x`、`y` 指向评测程序用 `cudaMalloc` 分配的显存，各 $BSD$ 个 `float`，存放顺序同测试文件。`init()` 被调用时 `x` 还没有内容。`spec.weights` 在主机内存中，只保证在 `init()` 调用期间有效。
- `forward()` 读取 `x` 当前的内容，算出结果写入 `y`，不能改动 `x`。`forward()` 返回时 GPU 上的工作可以尚未完成，评测程序随后调用 `cudaDeviceSynchronize()` 等待并检查 CUDA 错误。
- `init()` 不计时，可能被调用多次，每次的规模和权重可以不同。

## 四、评测程序

评测程序 `grader.cu` 如下。它先在样例上调用一次 `init()` 和 `forward()` 并核对结果；再读取测试文件，分配显存，调用 `init()`，用一份同形状、内容不同的数据运行一次 `forward()`（不计时）；然后上传真实的 `x`，清空 `y`，计时调用 `forward()` 并等待 GPU 完成；最后把 `y` 写到标准输出，把计时写到标准错误。

```cpp
// grader.cu
// Usage: grader.exe <input file>
//   stdout: y as raw little-endian float32 values, B*S*D of them
//   stderr: "forward_ms=<milliseconds>", the wall time of the timed forward() call
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

struct TransformerSpec { int B, S, D, H, F, L; const float* weights; };

void init(const TransformerSpec& spec, const float* x, float* y);  // untimed; x has no content yet
void forward();                                                    // timed; reads x, writes y

static void fail(const char* msg) {
    fprintf(stderr, "grader: %s\n", msg);
    exit(1);
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "grader: %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

static size_t weight_count(const TransformerSpec& s) {
    const size_t D = s.D, F = s.F;
    return (size_t)s.L * (9 * D + 4 * D * D + 2 * D * F + F) + 2 * D;
}

// Calls init(), optionally runs one untimed forward() on a different input of the same shape, then uploads the
// real input, times forward() plus the synchronization after it, and downloads y.
static std::vector<float> run(const TransformerSpec& spec, const std::vector<float>& input, bool warm, double* ms) {
    const size_t n = input.size();
    float* x = nullptr;
    float* y = nullptr;
    check(cudaMalloc(&x, n * sizeof(float)), "cannot allocate x");
    check(cudaMalloc(&y, n * sizeof(float)), "cannot allocate y");
    init(spec, x, y);
    check(cudaDeviceSynchronize(), "CUDA error in init()");
    if (warm) {
        std::vector<float> other(n);
        uint32_t state = 12345;
        for (size_t i = 0; i < n; i++) {
            state = state * 1664525u + 1013904223u;
            other[i] = (float)(state >> 8) / 16777216.0f * 4.0f - 2.0f;
        }
        check(cudaMemcpy(x, other.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cannot upload x");
        forward();
        check(cudaDeviceSynchronize(), "CUDA error in forward()");
    }
    check(cudaMemcpy(x, input.data(), n * sizeof(float), cudaMemcpyHostToDevice), "cannot upload x");
    check(cudaMemset(y, 0, n * sizeof(float)), "cannot clear y");
    check(cudaDeviceSynchronize(), "CUDA error before forward()");
    auto t0 = std::chrono::steady_clock::now();
    forward();
    check(cudaDeviceSynchronize(), "CUDA error in forward()");
    auto t1 = std::chrono::steady_clock::now();
    *ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::vector<float> out(n);
    check(cudaMemcpy(out.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost), "cannot download y");
    check(cudaFree(x), "cannot free x");
    check(cudaFree(y), "cannot free y");
    return out;
}

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

int main(int argc, char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _putenv_s("CUDA_MODULE_LOADING", "EAGER");  // request eager module loading
#endif
    if (argc != 2) fail("usage: grader.exe <input file>");
    check(cudaFree(0), "cannot create CUDA context");  // untimed
    // Untimed check on the sample of the problem statement, through the same init() + forward() path.
    double ms = 0;
    TransformerSpec sample{1, 3, 2, 2, 3, 1, SAMPLE_WEIGHTS};
    std::vector<float> sr = run(sample, std::vector<float>(SAMPLE_X, SAMPLE_X + 6), false, &ms);
    for (int i = 0; i < 6; i++) {
        const double d = std::fabs((double)sr[i] - SAMPLE_Y[i]);
        if (!(d <= 0.005 || d <= 0.02 * std::fabs(SAMPLE_Y[i]))) fail("wrong result on the sample input");
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) fail("cannot open input file");
    int32_t head[6];
    if (fread(head, sizeof(int32_t), 6, f) != 6) fail("cannot read the header");
    TransformerSpec spec{head[0], head[1], head[2], head[3], head[4], head[5], nullptr};
    std::vector<float> weights(weight_count(spec));
    std::vector<float> input((size_t)spec.B * spec.S * spec.D);
    if (fread(weights.data(), sizeof(float), weights.size(), f) != weights.size()) fail("cannot read the weights");
    if (fread(input.data(), sizeof(float), input.size(), f) != input.size()) fail("cannot read x");
    fclose(f);
    spec.weights = weights.data();
    std::vector<float> y = run(spec, input, true, &ms);
    if (fwrite(y.data(), sizeof(float), y.size(), stdout) != y.size()) fail("cannot write y");
    fflush(stdout);
    fprintf(stderr, "forward_ms=%.3f\n", ms);
    return 0;
}
```

## 五、样例

$B=1,S=3,D=2,H=2,F=3,L=1$。权重（按第二节的顺序）：

| 项 | 数值 |
|---|---|
| $\gamma_1,\beta_1$ | 1 0.875 0.125 0 |
| $W_Q,W_K,W_V$ | -0.75 1 -0.875 -0.25 -0.125 0.875 -0.125 -0.375 -0.625 -0.875 0 -0.125 |
| $b_Q,b_K,b_V$ | -0.25 -0.125 0.25 0.125 -0.125 0.125 |
| $W_O,b_O$ | 0.375 -0.125 -0.625 -0.875 0.125 -0.125 |
| $\gamma_2,\beta_2$ | 1.125 1 0 0.125 |
| $W_1,b_1$ | 0.25 0.75 0.375 0.5 0.375 0.5 0 -0.25 0.25 |
| $W_2,b_2$ | 0 -0.375 -0.875 0.625 -0.75 0.125 0.125 -0.25 |
| $\gamma,\beta$（末尾） | 1.125 0.875 0.125 -0.125 |

$x$：`-1 -1.125 -0.5 -1.375 -0.75 2`

$y$（保留 6 位小数）：`1.249621 -0.999705 1.249914 -0.999933 -0.999995 0.749996`

这就是评测程序开始时核对的样例。

## 六、数据规模

**24 个隐藏性能输入**：8 类参数配比，每类 3 个实例。

- $1\le B\le10^4$，$64\le S\le2\times10^5$，$64\le D\le2048$，$1\le H\le32$，$64\le F\le2048$，$1\le L\le32$，$BSD\le1.7\times10^8$。
- 权重和输入按下面的分布生成，相互独立：
  - $W_Q,W_K$ 的元素取 $[-3/\sqrt D,\,3/\sqrt D]$ 上的均匀分布。
  - $W_V,W_O,W_1$ 的元素取 $[-1/\sqrt D,\,1/\sqrt D]$ 上的均匀分布，$W_2$ 的元素取 $[-1/\sqrt F,\,1/\sqrt F]$ 上的均匀分布。
  - 所有 $b$ 和 $\beta$ 的元素取 $[-0.1,\,0.1]$ 上的均匀分布，所有 $\gamma$ 的元素取 $[0.9,\,1.1]$ 上的均匀分布。
  - $x$ 的元素取标准正态分布。

**7 个小规模隐藏正确性输入**：$BSD\le10^6$，$D$ 和 $F$ 可以小于上述下限，$H$ 可以等于 $D$，$S$ 可以为 1，$x$ 的数值分布也可能不同，但都是有限值。

所有隐藏输入事先固定，对所有提交相同，不公开。

## 七、评测环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 专业版（内部版本 26200） |
| CPU | Intel Core i7-14700KF：8 个性能核 + 12 个能效核，共 28 个逻辑处理器，L3 缓存 33 MB；支持 AVX2，不支持 AVX-512 |
| 内存 | 64 GB |
| GPU | NVIDIA GeForce RTX 4080：16 GB 显存，计算能力 8.9（sm_89），驱动 610.88，WDDM 模式（同时负责显示输出） |
| CUDA | CUDA Toolkit 13.2（nvcc V13.2.78） |
| 主机编译器 | MSVC 14.44（Visual Studio 2022 Build Tools） |

在 Visual Studio 的 x64 开发环境中（先执行 `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`），使用以下命令编译：

```
nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/O2 /utf-8" -o grader.exe grader.cu solution.cu
```

评测时进程被限制在 8 个性能核（逻辑处理器 0–15，共 16 个）上，并以高优先级运行。

## 八、计时与评价

- 只计评测程序对隐藏输入的那一次 `forward()` 调用的墙钟时间：从调用开始，到其后的 `cudaDeviceSynchronize()` 返回为止。评测程序中的其余步骤都不计时。开始时的样例同样必须算对，且不能出现 CUDA 错误。
- **正确性**：标准答案按第一节的公式用双精度算出。每个元素都必须是有限值，并且与标准答案之差的绝对值不超过 0.005，或者不超过标准答案绝对值的 2%。正确性输入和性能输入的每次运行都要满足。
- 正确性输入在独立进程中各运行 1 次，不计时。
- 同一次评测中，基线（出题人写的通用优化解法，代码不公开）和提交在每个性能输入上各运行 3 次，每次都是独立进程，交错进行，各取中位数。两者的运行条件完全相同：同一组处理器、同样的优先级和环境变量。

## 九、规则

1. 提交单个源文件 `solution.cu`。其中实现 `init()` 和 `forward()`，不包含 `main()`。
2. 只能使用 C++ 标准库、CUDA Runtime API、CUDA C++ 语言本身的设备端功能（含内建函数和内联 PTX），以及 CUDA Toolkit 自带、无需额外链接的头文件库；需要额外链接的库（如 cuBLAS）和其他第三方库都不能用，也不能通过任何方式调用 CUDA Driver API。
3. 由 $x$ 算出 $y$ 的全部计算必须在 `forward()` 调用中由 GPU 完成；只要结果在容差内，计算方式和内部精度不限。
   - 主机端可以读回由 GPU 算出的少量控制信息（总量与 $BSD$ 相比可以忽略），并据此决定是否继续提交工作；不能把张量内容复制回主机，也不能在 CPU 上对张量数值做任何计算。
   - `init()` 可以对权重做任何处理，也可以分配资源、预先建立运行时对象。`init()` 返回时，解法创建的主机线程不能再做任何工作。
4. 解法不能读写任何文件，不能使用或改动标准输入、标准输出和标准错误，不能联网，不能启动其他进程。
5. 同一个进程中 `init()` 和 `forward()` 会被调用不止一次，规模和内容可以不同。`init()` 必须能在已经初始化过的情况下重新初始化。不能释放 `x`、`y` 所指的显存；所有 GPU 工作都必须在 `forward()` 返回前提交，返回后不能再有主机线程继续提交工作。
6. 不能修改评测程序，不能以任何方式干扰任何一次运行的计时（包括基线的运行）。不能修改进程或线程的处理器亲和性和优先级，使用的逻辑处理器不能超出评测时给定的范围；线程数可以自行决定。不能修改系统设置或任何能延续到本进程结束之后的状态，不能访问或影响本进程以外的任何进程。
7. 每次运行的整个进程不超过 120 秒。内存和显存请合理使用（显卡同时负责显示输出），不要因占用过多导致内存、显存紧张或运行异常。
8. 允许使用多线程。

## 十、规定

1. 全程不能联网：不能使用网页搜索、网页抓取等联网工具，不能下载或安装软件包，也不能调用其他 AI 模型或服务。
2. 工作目录是本题面所在的文件夹，即 `H:\AI_OneShot_Model\<厂商>\<型号>\<题名>\`，开始作答时你就在这个目录中。所有文件只能在这个文件夹内读取、创建和修改；绝对禁止访问其他任何文件路径，包括同一型号下其他题目的文件夹；只有运行编译器、解释器、库、浏览器等必需工具时必须访问的安装路径和临时目录除外。
3. 开发期间可以在本机上编译、运行和测试，也可以编写任何辅助程序；只有最终提交的 `solution.cu` 需要遵守第九节的规则。
4. 测试输入需要自己构造。
5. 工作目录是一个 git 仓库。评分时先看工作目录：工作目录完整可用时，以那时工作目录中的 `solution.cu` 作为最终提交；停在半途（例如改到一半、文件不完整、无法编译或运行）的，以最近一次 git 提交中的 `solution.cu` 作为最终提交。
6. 每完成一个完整可用的阶段，就用 git 提交（`git add -A` 后 `git commit`）。有已提交的版本兜底，可以大胆推进，每有可用改进就提交。
7. 全程独立作答，不得开启子代理（subagent），也不得使用任何多代理或并行协作功能。
8. 严禁使用以下命令：`runas`，以及任何以其他账户身份登录、需要输入账户密码的命令；按进程名批量结束进程的命令（`taskkill /IM`、`pkill`、`killall`、`Stop-Process -Name` 等）。需要结束进程时，只按 PID 结束你自己启动的进程。这些命令会锁定评测账户或误杀评测系统的进程，导致你的会话被中止。

## 十一、注意事项

1. 隐藏性能输入覆盖面很广：24 个输入分属 8 类参数配比，在批量、序列长度、维度、头数和层数上差别很大。只在一两种配比上调优，换一类可能明显变慢甚至出错。建议优化时自己构造多种配比的测试用例，分别测速，并与一个简单可靠的实现核对结果。
2. Windows WDDM 启用了 GPU 超时检测与恢复（TDR），默认抢占等待超时约为 2 秒；长时间无法完成或无法被抢占的 GPU 任务可能触发驱动重置。部分输入的计算量很大，单次 GPU 提交不要超过这一时限。
3. RTX 4080 只有 16 GB 显存，同时负责显示输出。部分输入的序列很长，$S\times S$ 规模的中间结果放不进显存。评测时会关闭其他占用 GPU 的程序，但运行时间仍可能有少量波动。
4. 本机系统代码页为 936（简体中文）。编译命令中的 `/utf-8` 用于避免 CUDA 头文件触发大量 C4819 警告；源文件请保存为 UTF-8 编码。
5. 源文件所在路径含有中文时，nvcc 生成的中间文件会让 MSVC 报出大量 C4828、C4129 警告。这些警告不影响编译结果，在纯英文路径下编译可以避免。
6. 部分命令行环境设置了环境变量 `NoDefaultCurrentDirectoryInExePath`，此时 `cmd` 不会在当前目录中查找可执行文件。运行当前目录下的程序时，请写成 `.\grader.exe`。

## 十二、评分标准

评分对提交格式很宽容，格式上的小出入不用担心；但评测必需的内容仍按题面：`solution.cu` 能用第七节的命令与评测程序一起编译，`init()`、`forward()` 的接口与第三节一致，并遵守第九节的规则。

满分 100 分。24 个性能输入各自单独计分，本题得分为 24 个输入得分的平均值。

设性能输入 $i$（$i=1,\dots,24$）上基线与提交的计时分别为 $R_i$、$C_i$，加速比 $s_i=R_i/C_i$。输入 $i$ 的得分按 $s_i$ 在对数尺度上分段插值，锚点如下：

| $s_i$ | 0.2 | 0.6 | 1.2 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| 得分 | 0 | 5 | 15 | 25 | 45 | 70 | 86 | 100 |

$s_i\le0.2$ 时该输入得 0 分，$s_i\ge16$ 时得 100 分。$s_i$ 位于相邻锚点 $a_k\le s_i\le a_{k+1}$ 之间、两锚点的得分为 $p_k$、$p_{k+1}$ 时：

$$
\text{得分}_i=p_k+(p_{k+1}-p_k)\,\frac{\ln(s_i/a_k)}{\ln(a_{k+1}/a_k)}
$$

$$
\text{本题得分}=\frac{k}{7}\cdot\frac{1}{24}\sum_{i=1}^{24}\text{得分}_i
$$

其中 $k$ 为通过的正确性输入个数（0–7）。

各输入得分和本题得分都保留一位小数。越接近高加速比，同样的提速得分越多。
