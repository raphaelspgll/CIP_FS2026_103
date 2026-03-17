---
documentclass: article
fontsize: 11pt
geometry: margin=1in
linestretch: 1
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \renewcommand{\headrulewidth}{0pt}
  - \renewcommand{\arraystretch}{0.9}
---

# Feasibility Study – CIP Data Science Project

## Cryptocurrency Price Prediction Using Market Indicators  

\noindent\rule{\textwidth}{0.5pt}

\begin{tabular}{@{}ll@{}}
Course & Data Collection Integration and Preprocessing \\
Semester & FS 2026 \\
Team Members & Krishnathasan Tharrmeehan, Lemma Emanuel, Spagolla Raphaël \\
Group Number & 103 \\
Date & 12/03/2026 \\
\end{tabular}

\noindent\rule{\textwidth}{0.5pt}

## 1. Introduction / Project Scope

The cryptocurrency market is highly volatile and influenced by indicators such as trading volume, market capitalization, and historical price trends. Understanding the relationship between these indicators and price movements can provide insights into market behavior.

The goal of this project is to investigate whether cryptocurrency prices can be predicted using publicly available market indicators. Historical cryptocurrency data will be collected and analyzed to identify relationships between indicators and price changes.

Python will be used for data acquisition, preprocessing, analysis, and visualization, with the aim of evaluating whether simple models can provide useful short-term price predictions.

## 2. Research Questions

The project aims to answer several research questions related to the relationship between market indicators and cryptocurrency price movements.

1. **Which market indicators have the strongest relationship with cryptocurrency price changes?**

2. **Can historical market indicators be used to predict short-term cryptocurrency price movements?**

3. **How accurately can machine learning models predict the next-day price movement of major cryptocurrencies such as Bitcoin or Ethereum?**

An additional exploratory question is whether unusual changes in trading volume can signal upcoming price movements.

## 3. Data Sources

The dataset will be obtained from publicly available cryptocurrency market platforms such as CoinMarketCap, CoinGecko, and Binance. These platforms provide historical market data for a wide range of cryptocurrencies.

The collected dataset will include variables such as date, cryptocurrency name, price, market capitalization, trading volume, and percentage price change. These indicators are commonly used to analyze market behavior and may help identify patterns related to price movements.

If some indicators are missing from one platform, additional sources will be used to complement the dataset.

## 4. Data Acquisition Method

Cryptocurrency market data will be collected using automated data acquisition methods in Python. Web scraping techniques will retrieve publicly available data from cryptocurrency websites.

The Python library **Selenium** will be used to interact with dynamic website elements such as tables or filters, while **BeautifulSoup** will parse the HTML structure and extract the relevant information. The collected data will be stored in structured formats such as CSV files or Pandas DataFrames for further analysis.

## 5. Data Processing / Cleaning

After collection, the data will be cleaned and prepared for analysis using the Python libraries **Pandas** and **NumPy**. Missing values will be identified and handled appropriately, and data types will be checked and converted where necessary. Outliers and inconsistent values will also be detected and treated.

Additional variables such as daily returns, moving averages, and volatility indicators will be calculated to enrich the dataset and improve the predictive analysis.

## 6. Planned Analysis

The analysis will begin with exploratory data analysis to examine trends, distributions, and relationships between variables. Correlation analysis will be used to identify which indicators are most strongly related to cryptocurrency price changes.

Machine learning models such as linear regression and decision tree models will then be applied to evaluate their ability to predict short-term price movements. The results will be visualized using Python libraries such as **Matplotlib** and **Seaborn**.

## 7. Risks and Mitigation

One potential risk is that the structure of cryptocurrency websites may change, which could break the web scraping scripts and prevent data collection. To mitigate this risk, alternative data sources such as other cryptocurrency market websites or available APIs will be used if necessary.

Another possible issue is missing or incomplete data. Some indicators may not be available for all cryptocurrencies or time periods. This risk will be addressed by combining data from multiple sources and carefully handling missing values during the data cleaning process.

A further challenge could be limited predictive power of the selected indicators. Cryptocurrency markets are influenced by many external factors that may not be captured in the dataset. To address this limitation, multiple indicators and models will be tested to evaluate whether meaningful predictive relationships can still be identified.