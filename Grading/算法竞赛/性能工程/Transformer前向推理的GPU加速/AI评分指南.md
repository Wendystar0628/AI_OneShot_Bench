# Transformer前向推理的GPU加速 · AI 评分指南

- 材料：题面 `E:\Project\AI_OneShot_Bench\算法竞赛\性能工程\Transformer前向推理的GPU加速.md`（第八、九、十二节）；本目录的 `grader/grader.cu`、基线 `baselines/reference_solution.cu`、`performance/`（24 个性能输入）、`correctness/`（7 个正确性输入）；上一级目录的 `evaluate.py` 及其说明 `README.md`。
- 满分 100 分。本题整题在评分最后阶段统一串行测量。

## 测量与计分

- 把 `grader.cu`、提交的 `solution.cu` 和基线复制到纯英文路径（如 `C:\eval\transformer\`，中文路径会触发大量无害警告）。先执行 `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat` 进入 x64 开发环境，再用题面第七节的命令分别编译：`nvcc -O3 -std=c++17 -arch=sm_89 -Xcompiler "/O2 /utf-8" -o submission.exe grader.cu solution.cu`；基线把 `solution.cu` 换成 `reference_solution.cu`、输出 `reference.exe`。然后运行一次：

  ```
  python E:\Project\AI_OneShot_Bench\Grading\算法竞赛\性能工程\evaluate.py --task transformer --exe <提交.exe> --reference <基线.exe> --report <报告.json>
  ```

- `evaluate.py` 在每个性能输入上让基线和提交各运行 3 次，交错进行，各取中位数。
- 得分以 `evaluate.py` 的 `score` 为准：每个性能输入按 $s_i=R_i/C_i$ 在题面第十二节的锚点表上插值（$s_i$ 为 0.2、0.6、1.2、2、4、8、12、16 时得 0、5、15、25、45、70、86、100 分），答案超出容差或运行失败的输入记 0 分；24 个输入得分的平均 × 通过的正确性输入个数 ÷ 7。$G$ 只作参考。
- 容差：每个元素都是有限值，且与标准答案之差的绝对值不超过 0.005，或者不超过标准答案绝对值的 2%。
- 基线本身运行失败时本次评测作废，来信说明。
- 源文件名或位置与题面略有出入的从宽；函数接口、编译和第九节规则照常要求。

## 记 0 分的情形

- 编译失败。
- 违反题面第九节第 1–6 条任一条（以题面原文为准）：
  1. 单个 `solution.cu`，实现 `init()` 和 `forward()`，不含 `main()`。
  2. 只用 C++ 标准库、CUDA Runtime API、设备端功能和 CUDA Toolkit 自带、无需额外链接的头文件库；不用 cuBLAS 等需要链接的库，不调用 Driver API，不用其他第三方库。
  3. 由 $x$ 算 $y$ 的全部计算在 `forward()` 中由 GPU 完成：主机只读回少量控制信息，不把张量复制回主机、不在 CPU 上算张量；`init()` 返回后解法的主机线程不再工作。
  4. 不读写文件，不用或改动标准输入、输出和错误，不联网，不启动其他进程。
  5. 多次调用时 `init()` 能重新初始化，不释放 `x`、`y`，返回前提交全部 GPU 工作。
  6. 不改评测程序，不干扰任何一次运行的计时；不改处理器亲和性和优先级，不超出给定的逻辑处理器；不改系统设置或留下进程结束后的状态，不影响其他进程。

## 输出

得分；正确性输入通过个数（k/7）；24 个输入得分的平均和 $G$；evaluate.py 报告的路径。
