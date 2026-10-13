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
| P2 | 完成 | **第 0、1 轮**。`--round 0`；`--round 1`（6 个候选，可用 `--only` 分给 3 个进程，例如 `--only r1_lr010,r1_lr030`、`--only r1_leaves064,r1_depth6`、`--only r1_col060,r1_l2low`）；全部完成后 `--adopt 1`。回报：采纳了哪个配置、7 个配置在种子 0—2 上的均值（`--adopt` 的输出）、用时。提交 `results/external/p7_screen_adopted.json`。 || 2026-10-12 完成。采纳：**base**（第 1 轮无候选胜过基础配置）。`--adopt 1` 输出：`round 1: adopted base (mean over seeds [0, 1, 2] = 0.1174); all: base 0.1174, r1_depth6 0.1172, r1_lr010 0.1149, r1_lr030 0.1114, r1_col060 0.1066, r1_l2low 0.1017, r1_leaves064 0.0971`。参考（非登记量，仅供核对）：15 个种子的含成本年化超额均值 base 0.1100、depth6 0.1134、lr010 0.1131、col060 0.1100、lr030 0.1031、l2low 0.1021、leaves064 0.1014；base 各种子与 `seeds_lgb_mlp.csv` 的 LightGBM 逐位一致。用时：UTC 16:37—21:10，约 4.5 小时（含下述中断）；单次训练＋记录中位数约 105 秒，每次进程启动另需约 135 秒构建数据集。异常：① 按建议 3 进程并行时内存不足（每进程峰值约 7 GB，机器 15 GB），两进程被 OOM 杀死；② 改单进程后发现同一进程内每次运行内存增长约 0.8 GB，约 12 次后到 14 GB 再被 OOM 杀死。修法（小修，不改实验含义，单独提交 e4cd393）：`p7_screen.py` 加 `--max-runs N`，跑满 N 次即退出，外层循环以 `--max-runs 6` 重复调用直到 `all runs done`；已完成的种子照旧跳过。此后全部单进程运行，无 OOM。③ UTC 19:25 左右容器中断一次（进程丢失，`~/ext` 与仓库保留），20:02 续跑，约损失 35 分钟。所有 json 为原子写入，无半成品。 |
| P3 | 完成 | **第 2 轮**。`--round 2`（同样分 3 个进程）；完成后 `--adopt 2`。回报同上。若脚本提示某候选与已采纳配置相同而跳过，照实回报。 || 2026-10-13 完成。脚本未提示跳过（6 个候选的 overrides 都与 base 不同）。`--adopt 2` 输出：`round 2: adopted r2_sub060 (mean over seeds [0, 1, 2] = 0.1174); all: r2_sub060 0.1174, r2_leaves400 0.1174, base 0.1174, r2_lr005 0.1099, r2_col040 0.1043, r2_minleaf500 0.1017, r2_l1low 0.0877`。**须注意：采纳是一次三方平局，由浮点舍入决定。** r2_sub060、r2_leaves400 与 base 在全部 15 个种子上的 IC、含成本与不含成本年化超额逐位相同（差 <1e-12）；三者的种子 0—2 均值仅在第 16 位不同（0.11738949484591982 / …1970 / …1969），`max` 取了累加舍入后最大的 r2_sub060。原因（推断）：官方配置未设 `bagging_freq`，LightGBM 此时忽略 `subsample`（bagging_fraction），故 subsample=0.6 不起作用；num_leaves 400 与 210 在 max_depth=8 及强正则下长出的树相同。影响：被采纳的配置在功能上等同于 base，故第 3 轮候选等价于“base＋一项改动”，结果不受平局如何打破的影响；但论文里“第 2 轮采纳了 sub060”应表述为“第 2 轮无有效改进，两项候选为无效改动”。按规则未改脚本或采纳规则，是否需要在分析中把这两项候选视为与对照相同的筛查（d≡0）请本机判断。参考（15 种子含成本年化超额均值）：lr005 0.1113、base/sub060/leaves400 0.1100、col040 0.1067、minleaf500 0.0988、l1low 0.0923。用时：UTC 21:10—01:56，约 4.8 小时，其中实际计算约 2.6 小时（单次中位数约 98 秒）；其余是容器被回收后的等待：23:50 前后、00:05 后、01:11 前后各中断一次，均在下次检查时续跑，无数据损失，无 OOM。 |
| P4 | 完成 | **第 3 轮与分析**。`--round 3`；完成后 `--adopt 3`（只为记录）；再运行 `~/ext/.venv-qlib/bin/python external/p7_screen_analyze.py`，提交 `results/external/p7_screen.md`、`p7_screen_seeds.csv`、`p7_screen_screens.csv`、`p7_screen_adopted.json`。回报：分析输出的 P1—P5 几段原文。 || 2026-10-13 完成。`--adopt 3` 输出：`round 3: adopted r2_sub060 (mean over seeds [0, 1, 2] = 0.1174); all: r2_sub060 0.1174, r3_leaves128 0.1174, r3_lr015 0.1158, r3_depth10 0.1109, r3_l2high 0.1049, r3_l1high 0.1038, r3_col080 0.1033`（r2_sub060 与 r3_leaves128 再次逐位相同、并列；按 `max` 取先出现的对照）。**与 base 逐位相同的第 3 轮候选：r3_leaves128**（15 个种子的 IC、含/不含成本年化超额差均 <1e-12）；其余 5 个均不同。推断原因：num_leaves 取 128、210、400 结果全同，说明在 max_depth=8 与强 L1/L2 正则（λ1≈206、λ2≈581）下每棵树实际叶子数从未超过 128，num_leaves 上限不起作用；subsample 因未设 `bagging_freq` 被忽略（同 P3）。因此全部 18 次筛查中有 3 次是对照与候选完全相同的“零筛查”（第 2 轮 r2_sub060、r2_leaves400，第 3 轮 r3_leaves128，d_A＝d_B＝e＝0），分析脚本按原样把它们计入 P2—P4，未作改动，如何处理由本机决定。`p7_screen_analyze.py` 未改，正常退出；提交了 `p7_screen.md`、`p7_screen_seeds.csv`、`p7_screen_screens.csv`、`p7_screen_adopted.json`。分析输出 P1—P5 原文见下方“P4 回报附录”。要点：P1 采纳配置差距 +0.96pp，未采纳 17 个均值 +0.10（s.d. 0.88），命题 1 参照 1.22；P2 12 次筛查 d_A−d_B 均值 −0.53（t＝−3.15）；P3 年化超额斜率 0.18（−0.38 至 0.61）、事前 κ₃＝0.00（τ²＝0），IC 斜率 0.55（0.24 至 0.72）、κ₃＝0.60；P4 e 对 d_A 斜率高于对 d_B（年化 0.36 vs 0.13，IC 0.70 vs 0.49），方向与预期相反；P5 IC 的斜率与 κ₃ 均高于年化超额。用时：第 3 轮 UTC 01:56—05:05，约 3.1 小时（单次中位数约 101 秒），全程在会话回合内前台等待，无中断、无 OOM。另：前几次中断的原因已查明为会话空闲时容器被回收（后台进程不算活动），非程序问题。 |
| P5 | 待办 | **导出逐年指标**（不跑模型，几分钟）。先 `git merge origin/main` 取新脚本。运行 `~/ext/.venv-qlib/bin/python external/p7_screen_years.py`，提交 `results/external/p7_screen_years.csv`。回报：行数（应为 285 次运行 × 4 年 ＝ 1140 行）、有无缺失的运行。若 `~/ext/runs/p7s_*` 已不在（容器被重置），不要重跑，照实回报。补充登记见 `external/PREREG_P7_screen_addendum_zh.md`（A 部分）。 | |
| P6 | 待办（P5 后） | **第二个模型：MLP 上的同一筛查实验**（登记见 `external/PREREG_P7_screen_addendum_zh.md` B 部分；脚本 `external/p7_screen_mlp.py`、`external/p7_screen_mlp_analyze.py`，用法与 LightGBM 版相同，只换脚本名）。① `--smoke`：回报用时、IC 与含成本年化超额，并与 `results/external/seeds_lgb_mlp.csv` 中 MLP 种子 0（IC 0.0353 左右、年化超额 0.0766）对照，线程数不同可有小差异；若脚本因参数名不被接受而退出，照实回报、不要改候选。② `--round 0`、`--round 1`、`--adopt 1`、`--round 2`、`--adopt 2`、`--round 3`、`--adopt 3`，最后 `external/p7_screen_mlp_analyze.py`；内存允许时最多 2 个进程（用 `--only` 分配），沿用 `--max-runs 6` 外层循环续跑；MLP 单次约 1—10 分钟。③ 全部完成后再运行一次 `external/p7_screen_years.py`（这次会同时写出 `p7_screen_mlp_years.csv`）。④ 提交 `results/external/` 下的 `p7_screen_mlp.md`、`p7_screen_mlp_seeds.csv`、`p7_screen_mlp_screens.csv`、`p7_screen_mlp_adopted.json`、`p7_screen_mlp_years.csv`。回报：各轮采纳结果与 `--adopt` 原文、与对照逐位相同的候选（若有）、分析输出 P1—P5 原文、用时与异常。会话空闲时容器会被回收，请在回合内前台等待或定期查看。 | |

## P4 回报附录：`p7_screen_analyze.py` 输出 P1—P5 原文（云端 2026-10-13）

## P1 Overstatement on seeds 0-2 (annualized excess after costs, pp)

| configuration | adopted after round | mean seeds 0-2 | mean seeds 3-11 | gap |
|---|---|---|---|---|
| base | 1 | 11.74 | 10.78 | +0.96 |
| r1_col060 | — | 10.66 | 11.09 | -0.43 |
| r1_depth6 | — | 11.72 | 11.21 | +0.51 |
| r1_l2low | — | 10.17 | 10.48 | -0.31 |
| r1_leaves064 | — | 9.71 | 10.13 | -0.41 |
| r1_lr010 | — | 11.49 | 10.71 | +0.78 |
| r1_lr030 | — | 11.14 | 10.35 | +0.78 |
| r2_col040 | — | 10.43 | 10.66 | -0.24 |
| r2_l1low | — | 8.77 | 9.49 | -0.71 |
| r2_leaves400 | — | 11.74 | 10.78 | +0.96 |
| r2_lr005 | — | 10.99 | 11.15 | -0.17 |
| r2_minleaf500 | — | 10.17 | 9.59 | +0.59 |
| r2_sub060 | 2, 3 | 11.74 | 10.78 | +0.96 |
| r3_col080 | — | 10.33 | 11.91 | -1.58 |
| r3_depth10 | — | 11.09 | 9.41 | +1.68 |
| r3_l1high | — | 10.38 | 11.78 | -1.40 |
| r3_l2high | — | 10.49 | 9.73 | +0.76 |
| r3_leaves128 | — | 11.74 | 10.78 | +0.96 |
| r3_lr015 | — | 11.58 | 11.61 | -0.04 |

Never adopted (17 configurations): mean gap +0.10, s.d. 0.88. Adopted: base +0.96, r2_sub060 +0.96. Proposition 1 benchmark (independent noise, m = 7): sigma_cfg / sqrt(3) * mu_7 = 1.56 / 1.732 * 1.3522 = 1.22.

## P2 Rounds 2-3: development estimate on the adoption seeds minus on fresh seeds (pp)

12 screens: mean -0.53, s.d. 0.58, t = -3.15. By round: round 2: -0.74 (baseline gap +0.96); round 3: -0.32 (baseline gap +0.96)

## P3, P5 Retention on clean development seeds, against the ex-ante prediction

| metric | screens | observed slope (95% bootstrap) | ex-ante kappa_3 | sigma^2 (dev seeds) | tau^2 (dev seeds) |
|---|---|---|---|---|---|
| annualized excess (pp) | 18 | 0.18 (-0.38 to 0.61) | 0.00 | 5.971 | 0 |
| IC | 18 | 0.55 (0.24 to 0.72) | 0.60 | 3.115e-06 | 1.554e-06 |

## P4 Rounds 2-3: retention slope on the adoption seeds against fresh seeds

| metric | screens | slope of e on d_A | slope of e on d_B |
|---|---|---|---|
| ann | 12 | 0.36 | 0.13 |
| ic | 12 | 0.70 | 0.49 |
