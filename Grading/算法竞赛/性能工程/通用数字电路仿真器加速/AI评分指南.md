# 通用数字电路仿真器加速 · AI 评分指南

- 材料：题面 `E:\Project\AI_OneShot_Bench\算法竞赛\性能工程\通用数字电路仿真器加速.md`（第九、十、十三节）；本目录的 `grader/grader.cpp`、基线 `baselines/reference_solution.cpp`、`performance/`（24 个性能输入）、`correctness/`（7 个正确性输入）；上一级目录的 `evaluate.py` 及其说明 `README.md`。
- 满分 100 分。本题整题在评分最后阶段统一串行测量。

## 测量与计分

- 用题面第八节的命令分别编译（g++ 在 `C:\msys64\ucrt64\bin`，运行时也要在 PATH 中）：`g++ -O2 -std=c++20 -march=native -fopenmp -o submission.exe grader.cpp solution.cpp`；基线把 `solution.cpp` 换成 `reference_solution.cpp`、输出 `reference.exe`。然后运行一次：

  ```
  python E:\Project\AI_OneShot_Bench\Grading\算法竞赛\性能工程\evaluate.py --task circuit --exe <提交.exe> --reference <基线.exe> --report <报告.json>
  ```

- 得分以 `evaluate.py` 的 `score` 为准：每个性能输入按 $s_i=R_i/C_i$ 在题面第十三节的锚点表上插值（$s_i$ 为 0.1、0.5、1、1.75、2.75、4 时得 0、10、30、52、75、100 分），运行失败的输入记 0 分；24 个输入得分的平均 × 通过的正确性输入个数 ÷ 7。$G$ 只作参考。
- 基线本身运行失败时本次评测作废，来信说明。
- 源文件名或位置与题面略有出入的从宽；函数接口、`Circuit` 的定义、编译和第十节规则照常要求。

## 记 0 分的情形

- 编译失败。
- 违反题面第十节第 1–6 条任一条（以题面原文为准）：
  1. 单个 `solution.cpp`，实现 `simulate()`，不含 `main()`。
  2. 只用 C++ 标准库、OpenMP、Windows API、编译器内建函数和 CPU 指令内在函数，不用其他第三方库。
  3. 只用 CPU 计算，不用 GPU。
  4. 不读写文件，不用或改动标准输入、输出和错误，不联网，不启动其他进程。
  5. 多次调用 `simulate()` 时，每次都在返回前算完。
  6. 不改评测程序，不干扰任何一次运行的计时；不改处理器亲和性和优先级，不超出给定的逻辑处理器；不改系统设置或留下进程结束后的状态，不影响其他进程。

## 输出

得分；正确性输入通过个数（k/7）；24 个输入得分的平均和 $G$；evaluate.py 报告的路径。
