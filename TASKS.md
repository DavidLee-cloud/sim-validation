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
| T2 | 待办 | 运行测试；`calib2` 网格（3 进程：`--n-chunks 3 --chunk 0/1/2`）；再 `calib_e2e` 网格（同样 3 进程）；然后 `python summarize.py calib2 --by ic,protocol` 与 `python summarize.py calib_e2e --by ic,model,protocol`；另外回报 `.venv/bin/python -c "import sys,torch,numpy,pandas;print(sys.version,torch.__version__,numpy.__version__,pandas.__version__)"` 的输出 | |
