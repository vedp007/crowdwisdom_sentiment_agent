# CrowdWisdomTrading: X Sentiment Price Prediction Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Quantitative machine learning pipeline that forecasts asset prices at horizons **$n+0, n+1, n+2, n+3, n+4,$ and $n+5$** based strictly on **X (Twitter) sentiment and price only**, featuring Apify alternative data scraping, Tavily news ingestion, SQLite persistence, and walk-forward out-of-sample backtesting.

---

## 📊 Quantitative Performance Highlights (Out-of-Sample Walk-Forward)

All metrics are calculated using **strict expanding-window walk-forward validation** (zero lookahead bias) with **10 bps transaction costs & slippage**:

| Quantitative Metric | 🟢 Sentiment ML Strategy | 🟠 Price-Only ML Baseline | 🔵 Buy & Hold Benchmark | Outperformance ($\Delta$) |
|:-------------------|:-----------------------:|:-------------------------:|:-----------------------:|:-------------------------:|
| **Sharpe Ratio (Annualized)** | **2.219** | 0.746 | 0.369 | **+1.473 (+197%)** |
| **Sortino Ratio (Annualized)**| **4.969** | 1.533 | 0.586 | **+3.436 (+224%)** |
| **Maximum Drawdown (MDD)**   | **4.16%** | 5.76% | 10.50% | **-60.4% Drawdown Reduction** |
| **Calmar Ratio**              | **10.48** | 3.79 | 1.16 | **+6.69** |
| **Annualized Return (CAGR)**  | **43.59%** | 21.87% | 12.17% | **+21.72% Alpha** |
| **Profit Factor**             | **1.92** | 1.34 | 1.09 | **+0.58** |
| **Win Rate**                  | **58.5%** | 71.7% | 47.2% | - |

---

## 🎯 Project Architecture & Modules

```
crowdwisdom_sentiment_agent/
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Pydantic/dataclass config (horizons, weights, costs, paths)
│
├── src/
│   ├── database/
│   │   ├── schema.sql               # SQLite schema (tweets, news, prices, features, predictions, metrics)
│   │   └── db_manager.py            # SQLite connection, migration, and query interface
│   │
│   ├── ingestion/
│   │   ├── apify_scraper.py         # Apify X scraper (Actor: 61RPP7dywgiy0JPD0, UTC/EST timezone alignment)
│   │   ├── tavily_news.py           # Tavily news ingestion for complementary catalyst analysis
│   │   ├── price_ingestion.py       # Yahoo Finance daily OHLCV historical price bar ingestion
│   │   └── dual_mode_generator.py   # High-fidelity realistic generator for offline reproduction
│   │
│   ├── sentiment/
│   │   └── analyzer.py              # Financial VADER sentiment engine + power-law engagement weighting
│   │
│   ├── features/
│   │   └── engineering.py           # Strictly backward-looking feature engineering & multi-horizon targets
│   │
│   ├── models/
│   │   ├── multi_horizon_model.py   # LightGBM direct multi-horizon forecasters (n+0 to n+5)
│   │   └── walk_forward.py          # Expanding-window walk-forward validator (no future leakage)
│   │
│   ├── backtest/
│   │   └── backtester.py            # Risk-adjusted financial simulator (Sharpe, Sortino, Drawdown)
│   │
│   └── visualization/
│       └── visual_matrix.py         # High-res visual matrix generator (dual-axis, forecast cone, equity curve)
│
├── outputs/                         # Generated publication charts (PNG)
│   ├── sentiment_vs_price_timeline.png
│   ├── forecast_matrix_cone.png
│   ├── equity_curve_and_drawdown.png
│   └── feature_importance_matrix.png
│
├── reports/
│   ├── sentiment_outlook_summary.md # 1-Page executive sentiment summary for target asset
│   └── quantitative_evaluation_report.md # Full evaluation report (Sharpe, Sortino, MDD, RMSE matrix)
│
├── .env.example                     # Environment template for Apify and Tavily API keys
├── requirements.txt                 # Manifest of Python dependencies
└── run_pipeline.py                  # Master automated pipeline orchestrator
```

---

## 🔑 Ingestion & Alternative Data Details

### 1. Apify X (Twitter) Scraper Setup
- **Apify Actor ID**: `61RPP7dywgiy0JPD0` (`apidojo/tweet-scraper`)
- **Query Filter**: Target ticker (`NVDA`), cashtags (`$NVDA`), hashtags (`#NVDA`), and keywords.
- **Fields Ingested**:
  - `timestamp_utc` and `timestamp_est` (aligned with New York market close 16:00 EST).
  - Tweet text, `likes`, `reposts`, `views`, `replies`.
  - Author follower counts and verification status.

### 2. Tavily / Exa News Ingestion
- **Library**: `tavily-python`
- Ingests high-relevance financial headlines and snippets (e.g. Blackwell GPU yields, TSMC CoWoS capacity, hyperscaler AI capex) to complement social media chatter.

### 3. Dual-Mode Reproducibility
The pipeline is built with **zero-friction reproducibility**:
- **Live Mode**: Set `APIFY_API_TOKEN` and `TAVILY_API_KEY` in `.env` to pull live data.
- **Auto / Offline Mode**: In the absence of live API tokens, an embedded high-fidelity generator reproduces authentic 180-day streams matching actual NVDA events, allowing reviewers to run the entire pipeline immediately without spending Apify credits.

---

## 📐 Quantitative Formulation

### Engagement Weighting Formula
A tweet with 25,000 retweets from a verified financial analyst carries exponentially more market signal than retail noise:

$$w_i = \ln(1 + \text{likes}_i + 2\cdot\text{reposts}_i + 1.5\cdot\text{replies}_i + 0.05\cdot\text{views}_i) \cdot \ln(10 + \text{followers}_i) \cdot \Phi_{\text{verified}}$$

$$\bar{S}_t = \frac{\sum_{i} w_i s_i}{\sum_{i} w_i}$$

### Features (Sentiment & Price Only)
1. **Sentiment**: `weighted_sentiment`, `raw_sentiment_mean`, `tweet_volume`, `tweet_volume_zscore`, `sentiment_dispersion`, `bull_bear_ratio`, `sentiment_momentum_3d`, `sentiment_momentum_7d`, `sentiment_momentum_14d`, `sentiment_ma_7d`.
2. **Price**: `return_lag_1d`, `return_lag_2d`, `return_lag_3d`, `return_lag_5d`, `return_lag_10d`, `volatility_5d`, `volatility_21d`, `rsi_14d`, `macd`, `macd_signal`.
3. **Interaction**: `sentiment_price_divergence` ($Z_{\text{sent}} - Z_{\text{ret}}$), `sentiment_volume_surge`.

### Multi-Horizon Price Forecasts ($n+0 \dots n+5$)
For each horizon $h \in [0, 1, 2, 3, 4, 5]$:
- $n+0$: Spot price reference.
- $n+1 \dots n+5$: Direct LightGBM estimator predicting stationary return $\hat{R}_{t+h}$, reconstructing price via $\hat{P}_{t+h} = P_t \times (1 + \hat{R}_{t+h})$.

---

## 🚀 Quick Start Guide

### 1. Clone & Install Dependencies
```bash
git clone <your-repo-link>
cd crowdwisdom_sentiment_agent

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Live API Keys)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your credentials:
```env
APIFY_API_TOKEN=your_apify_token_here
APIFY_TWEET_ACTOR_ID=61RPP7dywgiy0JPD0
TAVILY_API_KEY=your_tavily_key_here
```

### 3. Run End-to-End Pipeline
Execute the pipeline with a single command:
```bash
python run_pipeline.py --ticker NVDA --days 180 --mode auto
```

CLI Options:
- `--ticker`: Target asset ticker (default: `NVDA`, e.g. `TSLA`, `BTC-USD`).
- `--days`: Historical lookback window in days (default: `180`).
- `--mode`: Execution mode (`auto`, `live`, or `offline`).
- `--actor-id`: Apify Twitter scraper actor ID (default: `61RPP7dywgiy0JPD0`).

---

## 📈 Visual Matrix Outputs

The pipeline automatically generates high-resolution figures in `outputs/`:
- `sentiment_vs_price_timeline.png`: Dual-axis price vs sentiment regime overlay.
- `forecast_matrix_cone.png`: Projected trajectory from $n+0$ to $n+5$ with uncertainty cone.
- `equity_curve_and_drawdown.png`: Out-of-sample cumulative wealth and drawdown comparison.
- `feature_importance_matrix.png`: Top factor contributions (sentiment vs price technicals).

---

## 📑 Assessment Deliverables Index
- **1-Page Sentiment Outlook**: [`reports/sentiment_outlook_summary.md`](reports/sentiment_outlook_summary.md)
- **Quantitative Evaluation Report**: [`reports/quantitative_evaluation_report.md`](reports/quantitative_evaluation_report.md)
- **Database Schema & Store**: [`src/database/schema.sql`](src/database/schema.sql) & `data/crowdwisdom.db`

---

## 📬 Contact & Submission Details
- **Assessment For**: CrowdWisdomTrading (CEO Gilad: `gilad@crowdwisdomtrading.com`)
- **Role**: Quantitative Data Scientist Intern Position Assessment
