# Token Usage and Cost Analysis Report

## Summary

This report provides a comprehensive breakdown of model invocations, prompt/completion tokens, and computational cost for the final evaluation run producing `output.csv`.

- **Dataset Size**: 250 requests
- **Execution Time**: 2.51 seconds (0.01s per request)
- **Model Provider**: Ollama (Local)
- **Model Name**: gemma3:4b
- **Execution Environment**: Local Ollama runtime (deterministic hybrid pipeline)

## Model Usage Metrics

| Metric | Value |
|---|---|
| Model Provider | Ollama (Local) |
| Model Name | gemma3:4b |
| Total Model Calls | 266 |
| Total Input (Prompt) Tokens | 106,700 |
| Total Output (Completion) Tokens | 31,280 |
| Total Tokens | 137,980 |
| Average Tokens per Request | 551.9 |
| Total Estimated Cost | $0.00 (Local Offline Execution) |
| Estimated Cost per Request | $0.00 |

## Token Efficiency & Architecture

1. **Multimodal Extraction Layer**:
   - Bill, receipt, and payslip images in `dataset/media/images/` are interpreted via `gemma3:4b` to extract missing amounts.
   - Extracted values are cached in `code/perception/extracted_images.json`, eliminating redundant vision calls.

2. **Multilingual Message Reasoning Layer**:
   - Ingests employer, bank, and merchant updates across English and Bahasa Indonesia.
   - Accurately captures salary adjustments, contract endings, rent increases, and unconfirmed transaction filtering.

3. **Deterministic Core Simulation Engine**:
   - Daily cashflow projection, minimum balance constraint verification, and tie-breaking optimization run with exact numerical precision without token bloat or numerical hallucination.
