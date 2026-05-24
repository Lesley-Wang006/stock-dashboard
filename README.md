# 📈 Stock Signal Dashboard

A multi-factor stock analysis dashboard that combines:

- Technical indicators
- AI-powered news sentiment
- Options flow data
- Interest rate signals

into a single daily **BUY / SELL / NEUTRAL** conclusion for any stock on Yahoo Finance.

---

## 🚀 Features

### 📊 Market Analysis
- RSI
- MACD
- MA50 / MA200
- Volume analysis
- Put/Call ratio
- Interest rate signals
- News sentiment scoring

### 🤖 AI Integration
- AI Morning Brief
- AI chatbot for stock questions
- Supports:
  - Google Gemini
  - Claude
  - DeepSeek

### 📈 Visualization
- Live candlestick charts
- MACD histogram
- Volume bars
- Portfolio diversification pie chart

### 🌍 Other Features
- Portfolio tracker
- 3 languages:
  - English
  - Français
  - 简体中文
- Dark / Pink themes
- PPO reinforcement learning trading bot

---

## 🧠 8-Signal Scoring Engine

Each signal gives:

- `+1` → Bullish
- `0` → Neutral
- `-1` → Bearish

### Final Score
- Score ≥ +3 → **BULLISH**
- Score ≤ −3 → **BEARISH**
- Otherwise → **NEUTRAL**

---

## 📌 Signals Used

| Signal | Bullish | Bearish |
|---|---|---|
| RSI | RSI < 30 | RSI > 70 |
| MACD | MACD above signal line | MACD below signal line |
| Interest Rates | 10Y yield falls | 10Y yield rises |
| News Sentiment | Positive VADER score | Negative VADER score |
| Options Flow | Put/Call < 0.7 | Put/Call > 1.0 |
| MA50 | Price above MA50 | Price below MA50 |
| MA Cross | Golden cross | Death cross |
| Volume | Volume above average | Volume below average |

---

## 🤖 Reinforcement Learning Trading Bot

The project also includes a PPO reinforcement learning trading agent trained on historical AAPL data.

### Features
- Live data using `yfinance`
- Signal-based features
- PPO model using Stable-Baselines3
- Stop-loss protection
- Portfolio balance tracking
- Trading visualization

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data | yfinance, pandas |
| AI | Gemini, Claude, DeepSeek |
| NLP | VADER |
| ML | Stable-Baselines3 PPO |
| Web App | Streamlit |
| Charts | Plotly |

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```text
stock-dashboard/
│
├── app.py
├── requirements.txt
├── PPO Training.ipynb
├── README.md
└── Excel basic calculations.xlsx
```

---

## ⚠️ Disclaimer

This project is for educational purposes only.

It is NOT financial advice.

Historical performance does not guarantee future results.

---

## 🙌 Acknowledgements

- DataCamp reinforcement learning project inspiration
- Claude by Anthropic for AI-assisted development
