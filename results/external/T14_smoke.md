# T14：P7 外部复现的可行性冒烟（Qlib，沪深300，Alpha158）

日期：2026-10-02；云端会话（4 核 CPU，15 GB 内存，无 GPU）。只做冒烟，未跑正式网格；官方配置未改。

## 结论

可行。安装和数据下载都很顺利；LightGBM、MLP 用官方配置各跑 1 个种子，主要指标与 Qlib 官方 README 同一量级（LightGBM 的 IC 略高，MLP 的 IC 与信息比略低）。
途中遇到三个版本兼容问题，都在独立环境里绕过，没有改 Qlib 源码，也没有改配置：MLflow 拒绝文件后端、torch 版本判断缺陷、`qlib.run.get_data` 已移除（详见第一节）。
CPU 上 GRU、LSTM、ALSTM 每个 epoch 约 80 秒，单种子约 0.5—1.5 小时（视早停轮数），20 种子×3 模型约 30—90 CPU 小时。

## 一、安装与下载

| 项目 | 情况 |
|---|---|
| 位置 | 全部在仓库外：源码 `~/ext/qlib`（`--depth 1`，提交 be72549），独立环境 `~/ext/.venv-qlib`，数据 `~/.qlib/qlib_data/cn_data` |
| 环境 | Python 3.10.18；pyqlib 0.9.7（`pip install pyqlib` 直接成功，约 10 秒）；lightgbm 4.7.0；mlflow 3.16.1；numpy 2.2.6；pandas 2.3.3；torch 最终用 2.6.0+cpu（理由见问题 3） |
| 数据 | 官方源可用，未用社区版。下载 196 MB，用时约 3 秒，解压后 521 MB |

遇到的问题与绕法：

1. **`python -m qlib.run.get_data` 不存在**：pyqlib 0.9.7 已不含 `qlib.run` 模块（报 ModuleNotFoundError）。改用源码里的 `python ~/ext/qlib/scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn`，参数不变。
2. **MLflow 3.x 拒绝文件后端**：`qrun` 启动即报 `MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode`。按报错提示设环境变量 `MLFLOW_ALLOW_FILE_STORE=true` 后正常，配置不改。
3. **torch 版本判断缺陷（Qlib 自身）**：`qlib/contrib/model/pytorch_nn.py` 第 151 行用字符串比较 `str(torch.__version__) <= "2.6.0"` 判断是否传 `verbose`。torch 2.14.1 时 `"2.14.1" <= "2.6.0"` 按字典序为真，于是向已删除该参数的 `ReduceLROnPlateau` 传 `verbose=True`，MLP 报 `TypeError`。把独立环境的 torch 降到 2.6.0+cpu 后正常。正式网格建议固定 torch 2.6.0，或向 Qlib 报告此缺陷。
4. 其他无害提示：CatBoost、XGBoost 未装，相关模型被跳过（本任务不用）；在非 git 目录运行时 Qlib 记录代码版本失败，日志里出现 `fatal: not a git repository` 和 `unknown option 'cached'`，不影响结果。

## 二、数据区间与沪深300 成分

- 日历：1999-11-10 至 2020-09-25，共 4943 个交易日。官方配置用 2008-01-01 至 2020-08-01，完全覆盖。
- 股票：`features/` 下 3875 只。`instruments/csi300.txt` 有 820 条成分区间，最后一条到 2020-09-25。
- 成分是否齐全：抽查 2008-01-02、2014-12-31、2017-01-03、2020-08-03 四天，成分股数分别为 302、300、300、300。
- 缺失：测试期（2017-01 至 2020-08）沪深300 成分股的收盘价共 261,300 行，缺失 7,627 行（2.9%），应为停牌。

## 三、官方配置单种子结果（沪深300，Alpha158）

配置（两模型相同）：训练 2008—2014，验证 2015—2016，测试 2017-01-01 至 2020-08-01；TopkDropout（topk 50，n_drop 5）；基准 SH000300；成本：开仓 0.0005，平仓 0.0015，最低 5 元。

| 指标 | LightGBM 本次 | LightGBM 官方 | MLP 本次 | MLP 官方（20 种子均值±标准差） |
|---|---|---|---|---|
| IC | 0.0468 | 0.0448 | 0.0353 | 0.0376±0.00 |
| ICIR | 0.3816 | 0.3660 | 0.2545 | 0.2846±0.02 |
| Rank IC | 0.0490 | 0.0469 | 0.0423 | 0.0429±0.00 |
| Rank ICIR | 0.4067 | 0.3877 | 0.2971 | 0.3220±0.01 |
| 年化超额（含成本） | 0.0807 | 0.0901 | 0.0643 | 0.0895±0.02 |
| 信息比（含成本） | 0.9145 | 1.0164 | 0.7730 | 1.1408±0.23 |
| 最大回撤（含成本） | −0.0861 | −0.1038 | −0.1292 | −0.1103±0.02 |
| 年化超额（不含成本） | 0.1260 | — | 0.1105 | — |
| 信息比（不含成本） | 1.4286 | — | 1.3265 | — |

官方数值取自 `~/ext/qlib/examples/benchmarks/README.md` 的 Alpha158 表，表中“Annualized Return / Information Ratio / Max Drawdown”按含成本超额对照。同期基准（沪深300）年化 0.1136。

对照：
- **LightGBM：** 本身是确定性模型，官方标准差为 0。本次 IC 类指标比官方高约 0.002；含成本年化超额 0.081，官方 0.090。差异可能来自数据版本（本次为最新官方数据包）或库版本。
- **MLP：** IC 比官方均值低 0.002（官方 IC 的标准差四舍五入后为 0.00，无法换算成标准差倍数）。含成本年化超额 0.064、信息比 0.77，分别比官方均值低约 1.3、1.6 个标准差，在单种子波动范围内。这正是 P7 关心的种子离散。

用时（4 核 CPU）：

| 阶段 | LightGBM | MLP |
|---|---|---|
| 读数据（Alpha158 特征生成） | 66 秒 | 66 秒 |
| 预处理 | 7 秒 | 141 秒（主要是 CSZFillna 131 秒） |
| 训练 | 约 50 秒 | 约 70 秒（第 3200 步早停） |
| 回测与分析 | 约 15 秒 | 约 30 秒 |
| **合计** | **144 秒** | **319 秒** |

## 四、GRU、LSTM、ALSTM 单种子 CPU 用时估计

方法：复制官方 Alpha158 配置（20 个特征，TSDatasetH，step_len 20，batch 800），只把 `n_epochs: 200` 改为 `n_epochs: 1`，各跑一次。副本只放在会话临时目录，没有提交。

| 模型 | 数据准备 | 每 epoch 训练 | 每 epoch 验证评估 | 每 epoch 合计 | 1-epoch 整体流程 |
|---|---|---|---|---|---|
| GRU | 70 秒 | 46 秒 | 32 秒 | 约 78 秒 | 217 秒 |
| LSTM | 71 秒 | 48 秒 | 36 秒 | 约 84 秒 | 221 秒 |
| ALSTM | 71 秒 | 49 秒 | 34 秒 | 约 83 秒 | 225 秒 |

官方配置为 `n_epochs: 200`、`early_stop: 10`，单种子用时约 = 2.5 分钟 + 实际轮数 × 约 80 秒：

| 早停轮数 | 单种子用时 |
|---|---|
| 20 轮 | 约 0.5 小时 |
| 50 轮 | 约 1.2 小时 |
| 200 轮（上限） | 约 4.5 小时 |

按 20—60 轮估计，3 个模型 × 20 种子约 30—90 CPU 小时；云端 3 进程并行约 10—30 小时。
单 epoch 的 IC 只有 0.02—0.03，不代表收敛后的水平，不作对照。

## 五、复现命令

```
mkdir -p ~/ext && git clone --depth 1 https://github.com/microsoft/qlib ~/ext/qlib
uv venv -p 3.10 ~/ext/.venv-qlib
uv pip install -p ~/ext/.venv-qlib pyqlib lightgbm
uv pip install -p ~/ext/.venv-qlib torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
cd ~/ext/qlib && ~/ext/.venv-qlib/bin/python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
export MLFLOW_ALLOW_FILE_STORE=true
cd ~/ext/qlib/examples/benchmarks/LightGBM && ~/ext/.venv-qlib/bin/qrun workflow_config_lightgbm_Alpha158.yaml
cd ~/ext/qlib/examples/benchmarks/MLP && ~/ext/.venv-qlib/bin/qrun workflow_config_mlp_Alpha158.yaml
```
