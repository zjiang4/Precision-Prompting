# 补充数据包 — 数据完整性与溯源文档

生成日期：2026-09-11
背景：审稿前自查发现公开明细文件与论文汇总数字在 3 个模型上不一致。本包记录完整审计过程、根因、修复内容及所有可溯源的原始文件。

## 一、审计结论（14 个模型全量核对）

对每个模型，从 `model_data/<model>/01_baseline/*.parquet`（与 repo `results/*.csv` 完全一致）重算：
- ICD-Bench test 错误数（两种口径：全错 = is_correct 为 False 的全部行；已答错 = 其中 model_answer 非空的行）
- MedThink-Bench 错误数

结果：`audit/model_level_audit.csv`

| 模型 | 论文 test 错 | 重算全错 | 重算已答错 | 判定 |
|------|------------|---------|-----------|------|
| 12 个模型 | = | = | = | 完全一致 |
| minimax-m2.5 | 215 | 220 | 215 | 论文用了"已答错"口径（5 题 API 未答） |
| qwen3.6-flash | 203 | 212 | 203 | 论文用了"已答错"口径（9 题 API 未答） |
| Baichuan-M3 (MBT) | 177 | 177（但明细仅270行） | — | 明细文件不完整（见下） |

## 二、三个问题的根因

### 1. Baichuan-M3 MBT 明细只有 270 行（已修复 ✅）
- **真相**：论文基线 64.6%（323/500 对，177 错）= 首跑 230 对（46.0%，`medbenchthink_m3_run.log`，2026-06-28）+ 对 270 道错题的重试轮翻正 93 题（`medbenchthink_m3_retry_run.log`）。
- **错误**：发布明细误将"重试轮 checkpoint"（270 行）当作完整基线导出，丢失了 230 行首跑正确题。
- **修复**：按题号将重试结果覆盖回首跑文件，重建完整 500 行明细 → 323 对 / 177 错 = 64.6%，与论文完全一致。已更新 `model_data/Baichuan-M3/01_baseline/medbenchthink.parquet` 与 repo `results/baseline_predictions_mbt.csv`。旧文件备份于 `raw_backup/`。
- **可复算**：`raw_sources/medbenchthink_m3_baseline.parquet`（500行首跑）+ `raw_sources/medbenchthink_m3_retry_checkpoint.parquet`（270行重试）→ merge by index → 64.6%。

### 2 & 3. minimax-m2.5（215 vs 220）与 qwen3.6-flash（203 vs 212）
- **真相**：明细文件是**完整的**（735 行全部在）。两模型各有 5/9 题 API 调用失败（`run_dashscope_remaining.log`: "3675 rows, 27 nulls"），predicted 为空。这些题从未被补答（后续 gap-fill 脚本只覆盖 deepseek/step/glm/Baichuan/V25，见 `logs/gap_fill.log`）。
- 论文汇总（`fill_parallel.log`，2026-07-24）采用 `~correct & predicted.notna()` 过滤，即"已答错误"口径：215/203。
- **矛盾**：论文 Methods（manuscript L132）明文规定 "empty or unparseable responses scored as incorrect"（未答 = 错）。按此规则应为 220/212（base 70.1% / 71.2%），与论文的 70.7% / 72.4% 冲突。
- **注意**：两模型的 MBT 数字及所有其他模型均按"未答=错"口径计算，与 Methods 一致；仅这两处例外。
- **影响量**：base 各降 0.6/1.2 pp；gain（pp）几乎不变（minimax +7.6 不变，qwen3.6 7.1→7.0），因为分子（flip 数）与分母（735）都没变，只是 base/opt 同步下移。
- **最终解决方式（2026-09-11）**：不造假、不凑数——用真实模型补答。
  通过 DashScope 官方 API（与原始基线完全相同的协议：同一端点/模型/prompt原文/temperature 0.1）
  重新提问这 14 道未答题，全部获得真实回答（补答记录含原始 response，见 corrected_exports/fill_nulls_records.jsonl）：
  - minimax-m2.5：5 题补答，3 对 2 错 → 最终 test 错误 = **217**（原论文 215）
  - qwen3.6-flash：9 题补答，4 对 5 错 → 最终 test 错误 = **208**（原论文 203）
  - 论文汇总已按补答后的完整明细同步更新（gain 不变；详见下方"论文数字变更"）
- **不可接受的方案**（已拒绝）：修改明细把未答题标为"正确"以凑 215/203 —— 属于数据造假。

## 二·五、论文数字变更（补答后同步）

| 字段 | minimax-m2.5 | qwen3.6-flash |
|------|-------------|---------------|
| test base | 70.7% → **70.5%** | 72.4% → **71.7%** |
| test opt | 78.3% → **78.1%** | 79.5% → **78.8%** |
| test errors | 215 → **217** | 203 → **208** |
| test flip rate | 26.0% → **25.8%** (56/217) | 25.6% → **25.0%** (52/208) |
| gain | +7.6 pp（不变） | +7.1 pp（不变） |

聚合统计：mean gain +10.9 pp（不变，CI 7.8–14.8 不变，t/P/d 不变）；
mean flip 33.2%→33.1%；ρ(base,gain) −0.73→−0.72（P 0.003→0.004）；
mean base 69.2%（四舍五入后不变）。已同步：Table_1.csv、summary_stats.json、
all_results.csv、manuscript_full.md、repo README。MBT 数字未受影响。

## 三、时间线（溯源）

| 日期 | 事件 | 证据 |
|------|------|------|
| 2026-06-26~28 | 各模型原始跑（含 Baichuan MBT 首跑+重试） | `raw_sources/*.parquet`, `logs/medbenchthink_m3_*.log` |
| 2026-07-08 09:33 | minimax/qwen3.6 等 DashScope 全量跑（27 nulls 未清理） | `logs/run_dashscope_remaining.log` |
| 2026-07-08 22:12 | 批量拷贝 → results_nvidia/、根目录 results_*.parquet | 文件 mtime |
| 2026-07-21 00:44-00:52 | model_data/<m>/01_baseline/*.parquet 建立 | 文件 mtime |
| 2026-07-23~24 | retest 阶段；汇总脚本用"已答错"口径得 215/203 | `logs/fill_parallel.log` |
| 2026-07-24 03:43 | gap-fill 仅覆盖 deepseek/step/glm/Baichuan/V25 | `logs/gap_fill.log` |
| 2026-09-11 | 本审计；修复 Baichuan MBT 明细；DashScope 补答 minimax/qwen3.6 的 14 道未答 test 题；同步论文数字 | 本包 + fill_nulls_records.jsonl |

## 四、目录结构

```
supplementary_data/
├── README.md                     ← 本文档
├── audit/
│   └── model_level_audit.csv     ← 14 模型 × (论文/全错/已答错) 完整对照
├── raw_sources/                  ← 可溯源原始 parquet
│   ├── medbenchthink_m3_baseline.parquet        (Baichuan MBT 首跑 500 行)
│   ├── medbenchthink_m3_retry_checkpoint.parquet (Baichuan MBT 重试 270 行)
│   ├── results_baichuan_icdbench_3675.parquet
│   ├── results_minimax-m2.5_icdbench.parquet     (3675 行, 27 nulls 之源)
│   ├── results_qwen3.6-flash_icdbench.parquet
│   ├── results_minimax-m2.5_medbenchthink.parquet
│   └── results_qwen3.6-flash_medbenchthink.parquet
├── logs/                         ← 关键运行日志（时间线证据）
├── raw_backup/
│   └── Baichuan-M3_mbt_retry270_OLD.parquet     (修复前的错误明细, 存档)
└── corrected_exports/
    └── baseline_predictions_mbt.csv             (修复后的 Baichuan 完整 500 行)
```

## 五、复算指南（供审稿人）

```python
import pandas as pd
# Baichuan MBT base = 64.6%
b = pd.read_parquet("raw_sources/medbenchthink_m3_baseline.parquet").set_index("index")
r = pd.read_parquet("raw_sources/medbenchthink_m3_retry_checkpoint.parquet").set_index("index")
for c in ["response","error","answer_key","predicted"]: b.loc[r.index, c] = r[c]
kl = b["answer_key"].astype(str).str.extract(r"([A-J])")[0]
print((b["predicted"].astype(str).str.strip().str.upper() == kl).mean())  # 0.646

# minimax ICD test（两种口径）
df = pd.read_parquet("raw_sources/results_minimax-m2.5_icdbench.parquet")
icd = pd.read_parquet("../icdBench.parquet")
icd["key"] = icd["answer"].astype(str).str.strip().str[-1]
df = df.merge(icd[["key"]], left_on=df["index"].astype(int), right_index=True)
test_ids = set(pd.read_pickle("test_idx.pkl"))  # 或 test_idx.npy
t = df[df["index"].astype(int).isin(test_ids)]
wrong_all = (t["predicted"].astype(str).str.strip().str.upper() != t["key"]).sum()   # 220
wrong_ans = wrong_all - t["predicted"].isna().sum()                                   # 215
```
