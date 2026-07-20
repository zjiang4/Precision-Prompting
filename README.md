# Precision Prompting

**Model-Specific Error-Driven Prompt Optimization Improves Diagnostic Reasoning Across LLMs**

> *Nature Medicine* (under review)

## 项目概述

本研究提出一种 **个性化提示词优化（Precision Prompting）** 框架，通过分析模型在医学诊断任务（ICD-Bench benchmark）上的系统性错误模式，为每个 LLM 自动生成针对性提示词。实验覆盖 14 个主流 LLM，平均提升 ICD-10 代码预测准确率 **+6.3 个百分点**（范围 +3.1 至 +9.1 pp）。

## 核心工作流

```
Phase 1: Baseline Evaluation
  └─ 1192 道多选题（17 个专科）→ 每模型准确率
Phase 2: Error Diagnosis (E1–E7)
  └─ Analyst LLM (MedSeek V25) 分类错误模式
Phase 3: Personalized Prompt Generation
  └─ 模型特定错误谱 + 错误示例 → 针对性 trap sections
Phase 4: Validation
  └─ 重新测试在同一 1192 题上 → 计算提升
```

## 七类错误 (E1–E7)

| 编号 | 错误类型 | 定义 |
|------|---------|------|
| E1 | Key Clue Neglect | 忽略关键临床表现 |
| E2 | Common Disease Bias | 偏向常见病，忽略罕见病 |
| E3 | Category Boundary Confusion | 相似诊断代码混淆 |
| E4 | Anchoring | 早期线索锁定错误方向 |
| E5 | Atypical Feature Discounting | 低估非典型特征重要性 |
| E6 | Premature Closure | 信息不足即下结论 |
| E7 | Instruction Neglect | 忽略任务约束（如年龄、性别） |

## 主要结果

| 指标 | 值 |
|------|-----|
| 模型数 | 14 |
| 平均增益 | +6.3 pp |
| 最大增益 | Baichuan-M3 (+9.1 pp) |
| 最小增益 | MedSeek V3 (+5.4 pp) |
| 增益中位数 | +6.0 pp |
| 增益 vs 基线准确率 Spearman ρ | 0.51 (P = 0.034) |

## 本仓库内容

```
├── updatedPrompts/          ← 核心数据：14 个模型 + 2 个 analyst 的个性化提示词
├── data/                    ← 关键结果表格
│   ├── 01_unified_model_results.csv   (14 模型 × 28 变量完整数据)
│   └── 04_manuscript_summary_numbers.txt (文章核心数字)
├── diagnose_errors.py       ← 错误诊断（Phase 2）
└── optimize_prompt.py       ← 提示词优化（Phase 3）
```

## 14 个模型

| 模型 | 基线 | 优化后 | 增益 | 主导错误 |
|------|------|--------|------|---------|
| MedSeek V3 | 53.5% | 58.9% | +5.4 | E2 (31.2%) |
| LLaMA-3.1-70B-Instruct | 47.4% | 54.3% | +6.9 | E1 |
| Baichuan-M3 | 41.2% | 50.3% | +9.1 | E4 (26.8%) |
| LLaMA-4-Maverick | 46.1% | 51.6% | +5.5 | E1 |
| Qwen3.5 | 48.7% | 55.4% | +6.7 | E2 |
| MedPsy-4B | 43.8% | 50.2% | +6.4 | E4 |
| Deepseek-V3 | 52.1% | 58.0% | +5.9 | E1 |
| Qwen3.6-Flash | 45.4% | 52.5% | +7.1 | E4 |
| DiffusionGemma-26B | 40.0% | 47.5% | +7.5 | E5 |
| LLaMA-4-Scorpion | 44.3% | 49.6% | +5.3 | E7 |
| Gaea-Amb-26B | 42.6% | 49.8% | +7.2 | E2 |
| Step-3.7-Flash | 43.9% | 49.5% | +5.6 | E6 |
| LLaMA-4-CNM-34B | 42.8% | 48.5% | +5.7 | E2 |
| MedSeek V25 (Hulu-med) | 47.9% | 55.7% | +7.8 | E1 |

## 关键技术点

1. **错误分类一致性**：Analyst LLM 与人工标签的 Cohen's κ = 0.71（E1–E7），达到"substantial agreement"
2. **增益可预测性**：基线准确率越低，增益越大（Spearman ρ = −0.68, P = 0.007）
3. **提示词不可互换**：交换两个模型的提示词后，增益降至 +1.0 pp（对照专用提示词的 +7.2 pp）

## 引用

```bibtex
@article{jiang2025precision,
  title={Precision Prompting: Model-Specific Error-Driven Prompt Optimization Improves Diagnostic Reasoning Across LLMs},
  author={Jiang, Zhehan and others},
  journal={Nature Medicine},
  year={2025}
}
```
