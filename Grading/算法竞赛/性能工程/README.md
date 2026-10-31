# 性能工程评测数据与工具

## 数据

- 各题都有 24 个性能输入和 7 个正确性输入。性能输入分 8 类，每类 3 个参数不同的实例，按类连续编号（`input_01`–`input_03` 为第 1 类，依此类推）；个数以 `performance/inputs` 中的实际文件为准。
- 电路正确性输入是 6 个小规模用例和 1 个完整规模的同步语义用例；Transformer 的正确性输入是小规模用例。
- `baselines/reference_solution.*` 是基线。生成参数见各题目录的 `README.md`。
- Transformer 的输入和答案是二进制文件（`.bin`，共约 9 GiB），不进 git，由题目目录的 `generator/gen_hidden.cpp` 重建；答案按容差比较，电路题逐字节比较。

## 使用方法

按各题题面的编译命令，把 `grader/grader.cpp` 或 `grader/grader.cu` 分别与提交、基线编译成两个可执行文件（编译失败直接记 0 分，脚本不负责编译），再在本目录用 Python 3.12 运行：

```powershell
python evaluate.py --task circuit --exe C:\eval\circuit\grader.exe --reference C:\eval\circuit\reference.exe --report C:\eval\circuit-result.json
python evaluate.py --task transformer --exe C:\eval\transformer\grader.exe --reference C:\eval\transformer\reference.exe --report C:\eval\transformer-result.json
```

## 一次评测会话

- 所有进程顺序运行，不同会话不能同时进行；每个进程以高优先级、亲和掩码 `0xffff` 和 `OMP_NUM_THREADS=16` 启动，超过 60 秒（Transformer 为 120 秒）判为失败。
- 先跑 7 个正确性输入，记下通过个数 $k$，失败不影响后续步骤。
- 再逐个跑性能输入：基线和提交各运行 3 次，交错进行（奇数编号先跑基线，偶数编号先跑提交，每跑完一对交换先后），各取中位数记为 $R_i$、$C_i$。

## 计分

- 每个性能输入按 $s_i=R_i/C_i$ 在题面锚点表上对 $\ln s_i$ 分段线性插值，得 0–100 分，保留一位小数；不高于最低锚点得 0 分，达到最高锚点得 100 分。
- 提交在某个输入上答案错误、崩溃、超时或计时无效，该输入记 0 分；3 次中任一次失败即算失败。
- 本题得分 `score` = 24 个输入得分的平均 × $k/7$，保留一位小数；`performance_score` 是未乘系数的平均；$G=\exp(\mathrm{mean}\ \ln s_i)$ 只作参考。
- 提交计时低于 0.001 ms 时按 0.001 ms 计；基线本身运行失败时会话作废，`score` 为 `null`，排查环境后重测。

## 报告

- `summary` 每个性能输入一行：`case`、`family`（取自 `validation.json`）、`R_i_ms`、`C_i_ms`、`s_i`、`points_i`、`status`（`PASS` 为通过）；`performance` 记每次运行的原始结果和顺序；`score`、`G` 等在顶层。

## 退出码

- 0 得分大于 0；1 记 0 分；2 会话作废。

## 重新生成正确性输入

`python gen_correctness.py` 重建电路题的正确性输入及答案，答案由脚本内独立的串行求解器算出。Transformer 的正确性输入由它自己的 `generator/gen_hidden.cpp` 生成。
