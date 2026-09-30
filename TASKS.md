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
