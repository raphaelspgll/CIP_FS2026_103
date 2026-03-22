# Management Summary
## Cryptocurrency Price Prediction: CIP FS2026 Group 103

**Authors:** Krishnathasan Tharrmeehan, Lemma Emanuel, Spagolla Raphaël
**Date:** 2026-03-22

---

### Recommendation

We recommend training two machine learning models per cryptocurrency: **Logistic Regression** and a **Decision Tree**. Both are directly justified by the data. More complex models such as XGBoost are not supported by the evidence and carry a high risk of producing inflated, unreliable results.

Realistic prediction accuracy lies in the range of **52 to 58%**, slightly above the 50% baseline of a random guess. This is the expected outcome for an efficient financial market and should not be interpreted as a failure of the approach.

---

### What Was Done

Historical daily price data for three cryptocurrencies (Bitcoin, XRP, and Internet Computer) was collected from CoinMarketCap, covering 1,198 trading days from December 2022 to March 2026. Ten market indicators were derived from the raw data, including daily returns, intraday price movements, short-term volatility, and trading volume changes. An Exploratory Data Analysis (EDA) was conducted to identify which indicators carry predictive information and to determine which models are appropriate.

---

### Key Findings

**Cryptocurrency returns are close to random.** Statistical tests showed that for all three coins, past daily returns carry almost no information about the next day's direction. This is consistent with efficient market theory and sets a hard ceiling on prediction accuracy.

**Two indicators show a weak but statistically significant signal.** For Bitcoin and XRP, a strong positive day slightly predicts a negative next day (mean reversion). The correlation is small (r = 0.07 to 0.08) but statistically confirmed (p < 0.05). No comparable signal was found for ICP.

**Several indicators must be excluded.** Price level, moving averages, and market capitalisation are nearly identical to each other (pairwise correlation above 0.91) and trend over time in ways that produce misleading results. They were excluded from the predictive models.

---

### Why These Models

**Logistic Regression** is the natural choice when the underlying signal is weak and linear. Its output is interpretable: each coefficient directly shows how much a given indicator influences the prediction. This answers the first research question directly.

**Decision Tree** captures simple threshold effects that Logistic Regression cannot, without requiring the large datasets that more complex models need. It is fully transparent and can be visualised as a flowchart.

**XGBoost and Random Forest** are powerful tools designed for datasets with rich, complex patterns. The EDA found no such patterns here. Applied to near-random data, these models learn noise rather than signal, which produces results that look good on training data but fail in practice. XGBoost also requires tuning many parameters, which is unreliable with only 1,190 observations.

---

### Next Step

Implement and evaluate Logistic Regression and Decision Tree models for each of the three cryptocurrencies (six models total), using time-series cross-validation to ensure results are not inflated by data leakage.
