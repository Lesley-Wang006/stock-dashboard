📈 Stock Signal Dashboard
A multi-factor stock analysis dashboard that combines technical indicators, AI-powered news sentiment, options flow data, and interest rate signals into a single daily buy/sell conclusion for any stock on Yahoo Finance.

Note: This project was originally inspired by a DataCamp reinforcement learning project on stock trading simulation with Gymnasium. The dashboard, signal engine, AI integration, portfolio tracker, and web app were built entirely with the help of Claude by Anthropic.

Live app: [your Streamlit URL here]

What it does
Every morning you open the dashboard, enter a ticker, and get a clear BULLISH / BEARISH / NEUTRAL conclusion based on 8 signals computed from live market data. You can also ask the AI chatbot questions about the stock in plain English.

Features

Any stock — type any Yahoo Finance ticker: AAPL, NVDA, MSFT, RY.TO, BTC-USD, SPY, and thousands more
8-signal scoring engine — each signal votes +1 (bullish), −1 (bearish), or 0 (neutral). Score ≥ +3 = BULLISH, ≤ −3 = BEARISH
Live candlestick chart — TradingView-style chart with MA50, MA200, volume bars, and MACD histogram for the last 180 days
Fundamental data — P/E ratio, market cap, beta, dividend yield, 52-week range, next earnings date with countdown
AI Morning Brief — click once to get a 3-sentence AI-generated summary of what to watch today
AI news sentiment — VADER NLP scores headlines from Yahoo Finance and Reuters automatically on every page load. No API key needed for this.
AI chatbot — ask anything about the stock. Supports Google Gemini (free), Claude (Anthropic), and DeepSeek
Portfolio tracker — add multiple stocks with custom weights, see a combined portfolio signal score and diversification pie chart
3 languages — English, Français, 简体中文 — switch from the sidebar
2 themes — Dark and Pink, switch instantly
20% stop-loss rule — the PPO trading agent (in the Jupyter notebook) stops trading if the balance drops 20% below starting capital


The 8 signals explained
SignalBullish conditionBearish conditionRSI (14-day)RSI < 30 (oversold)RSI > 70 (overbought)MACDMACD line above signal lineMACD line below signal lineInterest rates10Y Treasury yield fell > 5bpsYield rose > 5bpsNews sentimentVADER score > 0.2VADER score < −0.2Options flowPut/call ratio < 0.7Put/call ratio > 1.0MA50 positionPrice above 50-day MAPrice below 50-day MAMA crossover50MA above 200MA (golden cross)50MA below 200MA (death cross)VolumeToday's volume > 120% of 20-day averageVolume below average

The Reinforcement Learning trading bot
In addition to the dashboard, the Jupyter notebook (Untitled.ipynb) contains a PPO (Proximal Policy Optimization) reinforcement learning agent trained on 15 years of AAPL data:

Downloads live AAPL data from yfinance on every run — no stale CSV
Computes all 8 signals as features for the PPO agent
Normalizes features using RobustScaler (handles outliers better than MinMax for financial data)
Trains a neural network with architecture [256, 256, 128] for 100,000 timesteps
Runs a trading loop buying/selling 10% of balance per trade
Hard stop-loss: if portfolio value drops below $80,000, sells everything and stops
Produces two charts: trade actions on the price chart, and balance over time


How to run locally
bash# 1. Create environment
conda create -n trading python=3.11 -y
conda activate trading

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
Opens at http://localhost:8501

API keys (all optional)
The dashboard works without any API keys. Keys only unlock the AI features:
FeatureKey neededGet itAI Morning BriefGemini, Claude, or DeepSeekSee belowAI ChatbotGemini, Claude, or DeepSeekSee belowBetter news scoringGemini, Claude, or DeepSeekSee belowMore headlinesNewsAPInewsapi.org (free tier)
Google Gemini (free) — aistudio.google.com → Get API key → Create API key in new project
Anthropic Claude — console.anthropic.com → API Keys → Create Key ($5 free credit on signup)
DeepSeek — platform.deepseek.com → API Keys → top up ~$2 (lasts thousands of requests)

Project structure
stock-dashboard/
├── app.py              ← Streamlit dashboard (this file)
├── requirements.txt    ← Python dependencies
├── Untitled.ipynb      ← Jupyter notebook with PPO trading bot
├── AAPL.csv            ← Original DataCamp dataset (used for initial exploration)
└── README.md           ← This file

Tech stack
LayerTechnologyDatayfinance, pandas_datareader, feedparserTechnical indicatorsta (RSI, MACD, ATR, Bollinger Bands, MA)Sentiment analysisVADER, Google Gemini, Claude, DeepSeekTrading agentstable-baselines3 (PPO), gymnasium, gym-anytradingWeb appStreamlitChartsPlotly (candlestick, volume, MACD, pie)DeploymentStreamlit Cloud

Limitations and honest disclaimers

Not financial advice. This is an educational project.
The PPO agent trained on AAPL over 15 years looks profitable, but AAPL was one of the best-performing stocks in history. A buy-and-hold strategy would have significantly outperformed the bot.
Historical backtests do not guarantee future performance.
The signal system uses fixed thresholds (RSI > 70 = overbought) which work statistically but are not guarantees.
Options flow data (put/call ratio) is only available for today — historical backtest uses a neutral default.
Sentiment scores are automated — the AI does not have access to the full article, only the headline.


Acknowledgements

Original DataCamp project: Stock Trading Simulation with Gymnasium — provided the foundation PPO trading bot and AAPL dataset
Built entirely with Claude by Anthropic — signal engine, dashboard design, AI integration, portfolio tracker, multilingual support, and deployment


⚠️ This project is for educational purposes only. Nothing here constitutes financial advice.
