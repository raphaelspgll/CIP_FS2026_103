---
documentclass: article
fontsize: 11pt
geometry: margin=1in
linestretch: 1
---

# Appendix: Disclosure of AI and GenAI Tool Usage

In accordance with the HSLU guidelines on the use of AI and GenAI tools in academic assessments
(October 2025), all AI tool usage is documented below. The table is organized by chapter of the
final report. All AI-generated content was reviewed, verified, and adapted by the authors. The
authors take full responsibility for the correctness and quality of all submitted content.

Note on language: English is not the first language of any team member. DeepL Write was used
in all chapters to improve academic phrasing and fluency. All content, arguments, interpretations,
and conclusions were formulated by the authors. DeepL Write was used only to improve sentence
structure and word choice.

---

## Project Setup (prior to writing)

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Environment and project initialisation | Assisted with setting up the virtual environment, Git repository, and folder structure. | Entire project | Claude Code (claude.ai/code) |

---

## Abstract

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Language improvement | DeepL Write was used to improve academic fluency. All content was written by the authors. | Abstract | DeepL Write (deepl.com/write) |

---

## Chapter 1 – Introduction

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Language improvement | DeepL Write was used to improve academic fluency. All content and research questions were formulated by the authors. | Chapter 1 | DeepL Write (deepl.com/write) |

---

## Chapter 2 – Methods

### 2.1 Data Source and Collection

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Code development | Assisted with writing `src/coinmarket_scrape.py`, a Playwright-based scraper for CoinMarketCap OHLCV data. The authors reviewed and verified all generated code. | `src/coinmarket_scrape.py`, Section 2.1 | Claude Code (claude.ai/code), ChatGPT (chatgpt.com) |
| Language improvement | DeepL Write was used to improve academic fluency. All content was written by the authors. | Section 2.1 | DeepL Write (deepl.com/write) |

### 2.2 Preprocessing and Feature Engineering

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Code development | Assisted with writing `src/features.py`, which computes log-transformed features and defines the stationary predictor set. The authors reviewed and tested all generated code. | `src/features.py`, Section 2.2 | Claude Code (claude.ai/code), ChatGPT (chatgpt.com) |
| Statistical bug fix — stationarity | Helped identify that non-stationary level features produce spurious correlations with the target. The authors validated the finding and defined the final feature set. | `src/features.py`, `src/coinmarket_eda.py`, Section 2.2 | Claude Code (claude.ai/code) |
| Data leakage fix | Helped identify a same-day data leakage issue that caused 100% Decision Tree accuracy. The authors verified and implemented the fix in `src/models.py`. | `src/models.py`, Section 2.2 | Claude Code (claude.ai/code) |
| Language improvement | DeepL Write was used to improve academic fluency. All content was written by the authors. | Section 2.2 | DeepL Write (deepl.com/write) |

### 2.3 Analysis Methods

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Code development — EDA | Assisted with writing `src/coinmarket_eda.py`, covering data quality checks, lag-1 correlation, ACF analysis, cross-asset correlation, and rolling volatility. The authors defined all analysis goals and verified all outputs. | `src/coinmarket_eda.py`, Section 2.3 | Claude Code (claude.ai/code), ChatGPT (chatgpt.com) |
| Code development — ML pipeline | Assisted with writing `src/models.py`, implementing a chronological 70/15/15 split, model evaluation, and results visualisation. The authors defined the evaluation strategy and verified all outputs. | `src/models.py`, Section 2.3 | Claude Code (claude.ai/code), ChatGPT (chatgpt.com) |
| Language improvement | DeepL Write was used to improve academic fluency. All content was written by the authors. | Section 2.3 | DeepL Write (deepl.com/write) |

---

## Chapter 3 – Discussion

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Report writing support | Assisted with structuring the discussion sections. All numerical results, interpretations, and conclusions were written and verified by the authors. | Chapter 3 | Claude Code (claude.ai/code) |
| Language improvement | DeepL Write was used to improve academic fluency. All content was written by the authors. | Chapter 3 | DeepL Write (deepl.com/write) |

---

## Chapter 4 – Conclusion

| Purpose | Description | Affected Parts | Tool |
|---|---|---|---|
| Language improvement | DeepL Write was used to improve academic fluency. All content, limitations, and outlook points were written by the authors. | Chapter 4 | DeepL Write (deepl.com/write) |

---

*Tool references:*
- *Claude Code: CLI tool by Anthropic, accessed locally via the terminal. Model: claude-sonnet-4-6. URL: https://claude.ai/code*
- *ChatGPT: Generative AI tool by OpenAI. URL: https://chatgpt.com*
- *DeepL Write: AI-assisted writing and language tool by DeepL. URL: https://deepl.com/write*
