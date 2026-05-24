# 📈 Stock Signal Dashboard

A stock analysis dashboard that generates a daily morning brief using AI-powered news sentiment, options flow data, and interest rate signals to provide a single buy/sell conclusion for any stock on Yahoo Finance.

One day, while browsing through DataCamp (a learning platform recommended by UWaterloo finance professors), I came across an open-source project about stock trading simulation with Gymnasium which is doing. After finishing the project, I started wondering: since we are all using historical data, is there a way to connect to live market data and track our own stocks in real time?

I connected the dashboard to Yahoo Finance and researched several commonly used indicators such as RSI, MACD, and P/E ratio, etc. I also have friends who actively trade stocks, so I thought it would be interesting to build something that could help visualize portfolio creation and buy/sell signals more clearly.

To make the dashboard more interactive, I added AI chatbots including Gemini, Claude, and DeepSeek so users can ask questions about stocks or anything they interested in. After doing some of the technical analysis indicators calculations in Excel, I used Claude to help combine all the features together, and the final website was built with the assistance of [Claude](https://claude.ai) by Anthropic.

I also want to say thank you to Anthony Markham and Shreya Vashist for developing the original DataCamp project that inspired this dashboard:
https://app.datacamp.com/learn/projects/2468

Live app: https://lesley-stock-signals.streamlit.app/

---

## What it does

Every morning, you can open the dashboard, enter a ticker, and get a clear **BULLISH / BEARISH / NEUTRAL** conclusion based on 8 signals computed from live market data. You can also ask the AI chatbot questions about the stock in plain English.

---

## Features

- **Any stock** — type any Yahoo Finance ticker: AAPL, NVDA, MSFT, RY.TO, BTC-USD, SPY, and thousands more
- **8-signal scoring engine** — each signal votes +1 (bullish), −1 (bearish), or 0 (neutral). Score ≥ +3 = BULLISH, ≤ −3 = BEARISH
- **Live candlestick chart** — TradingView-style chart with MA50, MA200, volume bars, and MACD histogram for the last 180 days
- **Fundamental data** — P/E ratio, market cap, beta, dividend yield, 52-week range, next earnings date with countdown
- **AI Morning Brief** — click once to get a 3-sentence AI-generated summary of what to watch today
- **AI news sentiment** — VADER NLP scores headlines from Yahoo Finance and Reuters automatically on every page load. No API key needed for this.
- **AI chatbot** — ask anything about the stock. Supports Google Gemini (free), Claude (Anthropic), and DeepSeek
- **Portfolio tracker** — add multiple stocks with custom weights, see a combined portfolio signal score and diversification pie chart
- **3 languages** — English, Français, 简体中文 — switch from the sidebar
- **2 themes** — Dark and Pink, switch instantly
- **20% stop-loss rule** — the PPO trading agent (in the Jupyter notebook) stops trading if the balance drops 20% below starting capital

---

## The 8 signals explained

| Signal | Bullish condition | Bearish condition |
|---|---|---|
| RSI (14-day) | RSI < 30 (oversold) | RSI > 70 (overbought) |
| MACD | MACD line above signal line | MACD line below signal line |
| Interest rates | 10Y Treasury yield fell > 5bps | Yield rose > 5bps |
| News sentiment | VADER score > 0.2 | VADER score < −0.2 |
| Options flow | Put/call ratio < 0.7 | Put/call ratio > 1.0 |
| MA50 position | Price above 50-day MA | Price below 50-day MA |
| MA crossover | 50MA above 200MA (golden cross) | 50MA below 200MA (death cross) |
| Volume | Today's volume > 120% of 20-day average | Volume below average |

---

## The Reinforcement Learning trading bot

In addition to the dashboard, the Jupyter notebook (`Untitled.ipynb`) contains a PPO (Proximal Policy Optimization) reinforcement learning agent trained on 15 years of AAPL data:

- Downloads live AAPL data from yfinance on every run — no stale CSV
- Computes all 8 signals as features for the PPO agent
- Normalizes features using RobustScaler (handles outliers better than MinMax for financial data)
- Trains a neural network with architecture [256, 256, 128] for 100,000 timesteps
- Runs a trading loop buying/selling 10% of balance per trade
- Hard stop-loss: if portfolio value drops below $80,000, sells everything and stops
- Produces two charts: trade actions on the price chart, and balance over time

---

## How to run locally

```bash
# 1. Create environment
conda create -n trading python=3.11 -y
conda activate trading

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## API keys (all optional)

The dashboard works without any API keys. Keys only unlock the AI features:

| Feature | Key needed | Get it |
|---|---|---|
| AI Morning Brief | Gemini, Claude, or DeepSeek | See below |
| AI Chatbot | Gemini, Claude, or DeepSeek | See below |
| Better news scoring | Gemini, Claude, or DeepSeek | See below |
| More headlines | NewsAPI | newsapi.org (free tier) |

**Google Gemini (free)** — `aistudio.google.com` → Get API key → Create API key in new project

**Anthropic Claude** — `console.anthropic.com` → API Keys → Create Key ($5 free credit on signup)

**DeepSeek** — `platform.deepseek.com` → API Keys → top up ~$2 (lasts thousands of requests)

---

## Project structure

```
stock-dashboard/
├── app.py              ← Streamlit dashboard (this file)
├── requirements.txt    ← Python dependencies
├── Untitled.ipynb      ← Jupyter notebook with PPO trading bot
├── AAPL.csv            ← Original DataCamp dataset (used for initial exploration)
└── README.md           ← This file
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Data | yfinance, pandas_datareader, feedparser |
| Technical indicators | ta (RSI, MACD, ATR, Bollinger Bands, MA) |
| Sentiment analysis | VADER, Google Gemini, Claude, DeepSeek |
| Trading agent | stable-baselines3 (PPO), gymnasium, gym-anytrading |
| Web app | Streamlit |
| Charts | Plotly (candlestick, volume, MACD, pie) |
| Deployment | Streamlit Cloud |

---

## Limitations and honest disclaimers

- **Not financial advice.** This is an educational project.
- The PPO agent trained on AAPL over 15 years looks profitable, but AAPL was one of the best-performing stocks in history. A buy-and-hold strategy would have significantly outperformed the bot.
- Historical backtests do not guarantee future performance.
- The signal system uses fixed thresholds (RSI > 70 = overbought) which work statistically but are not guarantees.
- Options flow data (put/call ratio) is only available for today — historical backtest uses a neutral default.
- Sentiment scores are automated — the AI does not have access to the full article, only the headline.

---

## Acknowledgements

- Original DataCamp project: *Stock Trading Simulation with Gymnasium* — provided the foundation PPO trading bot and AAPL dataset
- Built entirely with [Claude](https://claude.ai) by Anthropic — signal engine, dashboard design, AI integration, portfolio tracker, multilingual support, and deployment

---

*⚠️ This project is for educational purposes only. Nothing here constitutes financial advice.*
