# Cryptocurrency Price Prediction  
### CIP – Data Collection, Integration and Preprocessing (HSLU)

## Overview
This project is part of the course *Data Collection, Integration and Preprocessing (CIP)*.  
The goal is to analyze whether cryptocurrency prices can be predicted using market indicators.

The project follows a typical data science workflow: data acquisition, preprocessing, analysis, and visualization using Python.

## Objectives
- Identify indicators influencing cryptocurrency price changes
- Predict price movement direction (increase/decrease)
- Compare performance of simple machine learning models

## Research Questions
1. Which market indicators are most strongly related to price changes?
2. How accurately can indicators from day (t-1) predict price movement on day (t)?
3. Which model achieves the best prediction performance?

## Data Sources
Data is collected from publicly available platforms such as:
- CoinMarketCap
- CoinGecko
- Binance

The dataset includes:
- Date
- Cryptocurrency name
- Price
- Market capitalization
- Trading volume
- Percentage price change


## Data Acquisition
- Web scraping using Selenium and BeautifulSoup
- Data stored as CSV files / Pandas DataFrames


## Data Processing
- Handling missing values
- Data type correction
- Outlier detection and treatment
- Feature engineering (e.g. returns, moving averages, volatility)


## Analysis
- Exploratory Data Analysis (EDA)
- Correlation analysis
- Machine learning models:
  - Linear Regression
  - Decision Trees


## Visualization
Data is visualized using:
- Matplotlib
- Seaborn

Focus on clear, interpretable plots (labels, scaling, readability).


## Project Structure
```

CIP_FS2026_103/
│── data/ # raw and processed datasets
│── docs/ # reports, feasibility study, documentation
│── notebooks/ # exploratory analysis (EDA)
│── scraper/ # web scraping scripts (Selenium, BeautifulSoup)
│── scripts/ # data processing and utility scripts
│── src/ # core logic (models, pipelines)
│── tests/ # testing code
│── .venv/ # virtual environment (not tracked)

```


## Risks and Limitations
- Website structure changes (scraping issues)
- Missing or incomplete data
- Limited predictive power due to external market factors

Mitigation:
- Use multiple data sources / APIs
- Robust data cleaning
- Test multiple models and features

## Technologies
- Python
- Pandas, NumPy
- Selenium, BeautifulSoup
- Matplotlib, Seaborn
- Scikit-learn

## Authors
Krishnathasan Tharrmeehan  
Lemma Emanuel  
Spagolla Raphaël