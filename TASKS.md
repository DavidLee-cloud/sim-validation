# 任务单（本机会话写入，云端会话执行）

**云端会话的固定流程**：用户说“继续”时——

1. `git fetch origin && git checkout <你的工作分支> && git merge origin/main`（取最新代码与任务）；
2. 如环境未装好，运行 `bash setup_cloud.sh`；一律用 `.venv/bin/python`；
3. 按顺序执行下表中状态为“待办”的任务，最多 3 个进程并行；不要运行表中没有的网格；
4. 每完成一个任务：把状态改为“完成”，在“云端回报”一栏写用时、异常、汇总命令的关键输出（几行即可）；
5. 连同 results/ 提交并推送到你的工作分支。遇到代码问题：先在回报里写清原因与修法，小修（不改变任何实验含义）可直接改并单独提交，涉及实验含义的不要改。

| 编号 | 状态 | 任务 | 云端回报 |
|---|---|---|---|
| T1 | 完成 | 测试＋calib 网格（48 单元） | 已完成，结果在 claude/hello-pmznrh；并发缓存 bug 已修 |
| T2 | 完成 | 运行测试；`calib2` 网格（3 进程：`--n-chunks 3 --chunk 0/1/2`）；再 `calib_e2e` 网格（同样 3 进程）；然后 `python summarize.py calib2 --by ic,protocol` 与 `python summarize.py calib_e2e --by ic,model,protocol`；另外回报 `.venv/bin/python -c "import sys,torch,numpy,pandas;print(sys.version,torch.__version__,numpy.__version__,pandas.__version__)"` 的输出 | 完成（2026-09-30）。测试 5/5 通过（7.6s）；calib2 27/27 单元 111s，calib_e2e 24/24 单元 124s（各 3 进程，无报错、无缓存冲突）。环境：`3.10.18 (main, Sep  2 2025, 14:19:37) [Clang 20.1.4 ] 2.10.0+cpu 2.2.6 2.3.3`。<br>`summarize.py calib2 --by ic,protocol`（real_ic / true_ic / ann_net / excess_ir，n=3）：ic0.1: p1 -0.017/0.207/-0.112/-0.25, p1x 0.015/0.145/0.008/0.40, p3 0.015/0.246/-0.004/0.34；ic0.15: p1 0.008/0.279/-0.048/0.10, p1x 0.027/0.211/0.026/0.50, p3 0.040/0.328/0.055/0.65；ic0.2: p1 0.027/0.326/0.035/0.53, p1x 0.046/0.254/0.077/0.77, p3 0.068/0.384/0.147/1.08。<br>`summarize.py calib_e2e --by ic,model,protocol`（同上四列）：ic0.15: e2e-p1 0.002/0.101/-0.041/0.04, e2e-p3 0.020/0.112/0.035/0.76, reg-p1 -0.001/0.246/-0.110/-0.23, reg-p3 0.031/0.282/0.007/0.42；ic0.2: e2e-p1 0.006/0.118/-0.038/0.09, e2e-p3 0.028/0.133/0.095/1.30, reg-p1 0.023/0.310/-0.032/0.18, reg-p3 0.059/0.351/0.107/1.02。汇总另存 results/summary_calib2_all.json、results/summary_calib_e2e_all.json。 |
| T3 | 完成 | 运行 `calib_top` 网格（30 单元，3 进程）；然后 `python summarize.py calib_top --by lowvol_share,model,risk,temp` | 完成（2026-09-30）。测试 5/5 通过；calib_top 30/30 单元，3 进程 274s，无报错、无缓存冲突。<br>`summarize.py calib_top --by lowvol_share,model,risk,temp`（real_ic / true_ic / k_over_c / ann_net / excess_ir，n=3）：lv0.2: e2e_fid r20 t0.3 0.025/0.134/4.20/0.103/1.01, r20 t1 0.021/0.134/4.25/0.098/0.92, r5 t0.3 0.023/0.117/3.60/0.099/0.92, r5 t1 0.017/0.114/3.25/0.098/0.93, reg 0.085/0.327/1.72/0.243/1.32；lv0.4: e2e_fid r20 t0.3 0.026/0.133/3.93/0.113/1.07, r20 t1 0.023/0.134/3.52/0.105/0.94, r5 t0.3 0.025/0.118/3.31/0.090/0.89, r5 t1 0.018/0.113/2.83/0.091/0.86, reg 0.094/0.369/1.61/0.282/1.43。所有组 chosen_epoch=30。汇总另存 results/summary_calib_top_all.json。 |
| T4 | 完成 | 运行 `calib_top2` 网格（12 单元，其中 reg 的 risk 两档结果相同属正常，3 进程）；然后 `python summarize.py calib_top2 --by model,risk` | 完成（2026-09-30）。测试 5/5 通过；calib_top2 12/12 单元，3 进程 104s，无报错。<br>`summarize.py calib_top2 --by model,risk`（real_ic / true_ic / oracle_capture / k_over_c / ann_net / excess_ir / bucket0_r_ann / bucket10_r_ann，n=3）：e2e_fid r20 0.020/0.100/0.128/6.79/0.055/0.66/0.132/0.050；e2e_fid r5 0.015/0.078/0.126/4.90/0.036/0.50/0.116/0.054；reg r5 与 r20 完全相同（符合预期）0.065/0.244/0.238/1.99/0.155/0.96/0.229/0.158。所有组 chosen_epoch=30。汇总另存 results/summary_calib_top2_all.json。 |
| T5 | 完成 | 运行 `B_score` 网格（分数层蒙特卡洛，无训练；3 进程）；回报每个单元的 b0/b5/b10/b20/b40/b80 与各 rule_ 值（写入 results/summary_B_score.json 即可，回报里列要点） | 完成（2026-09-30）。测试 5/5 通过；B_score 17/17 单元，3 进程 17s，无报错。每单元 b0/b5/b10/b20/b40/b80 与 rule_* 已写入 results/summary_B_score.json。要点：b1（b=0）各档随名次单调下降，b0 在 noise 1/3/10 时为 1.57/0.59/0.18，rule_topk 1.32/0.55/0.17，rule_oracle 1.93；b2（noise 3）b=0 时 b0≈0.78 单调下降，b≥1 且 df=3 时出现顶端倒挂（b=1: b0 0.74<b5 0.82；b=2: b0 0.44<b5 0.83），df=6/正态下 b=2 时 b0 与 b5 接近（0.93/0.92、1.13/0.95）；b1/b2 所有单元 rule_lcb=0.000（b3 中非零：0.384/0.232），请确认是否符合设计；b3 noise 1/3：b0 0.40/0.29，rule_topk 0.39/0.25，rule_lcb 0.38/0.23，rule_oracle 0.40。 |
| T6 | 待办 | 正式网格（登记 PREREG_A_zh.md 已冻结，**不得修改 run_grid.py、sim/、analyze_A.py**；发现缺陷先回报、不要自行改）：`A_val`（120 单元）然后 `A_e2e`（160 单元），各 3 进程 | |
| T7 | 待办 | `A_hl`（480 单元，3 进程） | |
| T8 | 待办 | `A_main` 前半：3 进程分别跑 `--n-chunks 6 --chunk 0`、`1`、`2` | |
| T9 | 待办 | `A_main` 后半：`--n-chunks 6 --chunk 3`、`4`、`5`；全部完成后运行 `python analyze_A.py`，把输出要点写进回报 | |
