# P7 任务单（本机会话写入，P7 专用云端会话执行）

本任务单只给 P7 的云端会话用，与 `TASKS.md`（论文六的云端会话）互不相干：不要执行 `TASKS.md` 里的任务，也不要改它。

**P7 云端会话的固定流程**：用户说“继续”时——

1. 第一次：从 `origin/main` 新建自己的工作分支（建议名 `claude/p7-screen`），以后每次 `git fetch origin && git merge origin/main` 取最新代码与任务；只在自己的工作分支上提交，不要推送到 `main`，也不要碰 `claude/hello-pmznrh`。
2. 环境：首次运行 `bash setup_p7_ext.sh`（Qlib、数据与独立虚拟环境都在仓库外的 `~/ext/` 与 `~/.qlib/`，不要提交数据或运行目录）。Qlib 相关一律用 `~/ext/.venv-qlib/bin/python`，并设 `export MLFLOW_ALLOW_FILE_STORE=true`。
3. 单条命令最长 1 小时（环境变量 `BASH_MAX_TIMEOUT_MS=3600000`），而一轮实验可能更久：训练一律用 `nohup … > ~/ext/logs/<名称>.log 2>&1 &` 放到后台，定期查看日志与 `~/ext/runs/p7s_*/` 下已完成的 json 数，全部完成后再做下一步；会话中断后重新执行同一命令即可续跑（已完成的种子会跳过）。
4. 按顺序执行下表中状态为“待办”的任务，最多 3 个进程并行；不要运行表中没有的实验；不要改动候选、采纳规则与种子集合（已事先登记于 `external/PREREG_P7_screen_zh.md`）。
5. 每完成一个任务：把状态改为“完成”，在“云端回报”一栏写用时、异常与汇总输出的关键几行；连同 `results/external/` 下的新文件提交并推送到工作分支。遇到代码问题：先在回报里写清原因与修法，小修（不改变实验含义）可直接改并单独提交，涉及实验含义的不要改。

| 编号 | 状态 | 任务 | 云端回报 |
|---|---|---|---|
| P1 | 完成 | **环境重建与冒烟**。`bash setup_p7_ext.sh`；然后 `~/ext/.venv-qlib/bin/python external/p7_screen.py --smoke`（基础配置、种子 0）。回报：各包版本、数据最后一个交易日、数据集构建用时、单次训练用时、该次的 IC 与含成本年化超额（与 T14 的 LightGBM 冒烟 IC 0.0468、年化超额 0.081 对照，线程数不同可有小差异）。 || 2026-10-12 完成。setup 约 5 分钟（无报错）；版本 Python 3.10.20、pyqlib 0.9.7、lightgbm 4.7.0、mlflow 3.16.1、numpy 2.2.6、pandas 2.3.3、torch 2.6.0+cpu；数据最后一个交易日 2020-09-25。冒烟（base，种子 0，num_threads=4）：数据集构建 141 秒，单次训练＋记录 128.5 秒；IC 0.0463、ICIR 0.380、Rank IC 0.0491，年化超额含成本 0.1308（不含成本 0.1769），信息比 1.556，最大回撤 −0.0705。与 `results/external/seeds_lgb_mlp.csv` 的 LightGBM 种子 0 逐位一致（IC 0.046346、含成本 0.130786）。与 T14 冒烟（IC 0.0468、0.081）相比 IC 接近而年化超额高约 0.05：T14 用的是官方配置不设种子，此处设 seed=0；LightGBM 的 20 种子含成本年化超额本就为 0.107±0.016，组合层对种子敏感，非环境问题。 |
| P2 | 待办（P1 完成后） | **第 0、1 轮**。`--round 0`；`--round 1`（6 个候选，可用 `--only` 分给 3 个进程，例如 `--only r1_lr010,r1_lr030`、`--only r1_leaves064,r1_depth6`、`--only r1_col060,r1_l2low`）；全部完成后 `--adopt 1`。回报：采纳了哪个配置、7 个配置在种子 0—2 上的均值（`--adopt` 的输出）、用时。提交 `results/external/p7_screen_adopted.json`。 | |
| P3 | 待办（P2 完成后） | **第 2 轮**。`--round 2`（同样分 3 个进程）；完成后 `--adopt 2`。回报同上。若脚本提示某候选与已采纳配置相同而跳过，照实回报。 | |
| P4 | 待办（P3 完成后） | **第 3 轮与分析**。`--round 3`；完成后 `--adopt 3`（只为记录）；再运行 `~/ext/.venv-qlib/bin/python external/p7_screen_analyze.py`，提交 `results/external/p7_screen.md`、`p7_screen_seeds.csv`、`p7_screen_screens.csv`、`p7_screen_adopted.json`。回报：分析输出的 P1—P5 几段原文。 | |
