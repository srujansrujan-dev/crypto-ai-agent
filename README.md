# 🤖 Crypto AI Agent

A cryptocurrency market analysis agent that continuously scans the wider
market, filters results down to CoinDCX-listed tradable symbols, prioritises
CoinDCX futures instruments, evaluates opportunities with Gemini AI, and
displays trading suggestions on a live web dashboard.

> ⚠️ **DISCLAIMER:** This tool provides analysis ONLY. It does NOT execute
> trades and does NOT connect to any exchange account. Use at your own risk.

---

## ✨ Features

| Feature | Detail |
|---|---|
| Market scanner | 500+ coins via CoinGecko, filtered to CoinDCX tradable universe |
| Indicators | RSI, MA20, MA50, Momentum, Volatility, Volume Ratio |
| Pump detector | Scoring system (0–100), flags ≥70 |
| Trend detector | Multi-cycle volume & momentum tracking |
| AI analysis | Gemini 1.5 Flash → BUY / HOLD / AVOID |
| Self-learning | Tracks WIN/LOSS, adjusts weights automatically |
| Futures layer | CoinDCX active futures instruments + conservative leverage hint |
| Dashboard | Live web UI at port 8080 |
| Backtester | Historical signal testing with win rate & drawdown |

---

## 🚀 Quick Start (Local)

### 1. Get your free Gemini API key
Go to → https://aistudio.google.com → **Get API Key** → Copy it

### 2. Clone / download this project
```bash
cd crypto-ai-agent
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
# Copy the example file
cp .env.example .env

# Open .env in any text editor and paste your key:
# GEMINI_API_KEY=AIza...your_key_here
```

### 5. Run the agent
```bash
python main.py
```

### 6. Open the dashboard
Open your browser → **http://localhost:8080**

That's it! The agent will scan every 5 minutes and show signals on the dashboard.

---

## 📊 Run the Backtester

```bash
# Test on Bitcoin + Ethereum for last 30 days
python backtester.py --coins bitcoin,ethereum --days 30

# Test more coins
python backtester.py --coins bitcoin,ethereum,solana,cardano --days 60
```

---

## ☁️ Deploy to Render (Free Cloud)

Render can host the dashboard, but free web services sleep after inactivity, so it is better for demos than for a strict 24/7 futures scanner.

### Steps:

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "initial commit"
   git push origin main
   ```

2. **Go to** https://render.com → Sign up free

3. Click **"New +"** → **"Web Service"**

4. Connect your GitHub repo

5. Render auto-detects `render.yaml` — just click **Deploy**

6. Go to **Environment** tab → Add:
   ```
   GEMINI_API_KEY = your_key_here
   ```

7. Your dashboard will be live at:
   ```
   https://crypto-ai-agent.onrender.com
   ```

> ⚠️ Free Render instances sleep after 15 min of inactivity.
> Use https://uptimerobot.com (free) to ping it every 10 min to keep it awake.

---

## ☁️ Deploy to Railway (Alternative)

1. Go to https://railway.app → Sign up
2. **New Project** → **Deploy from GitHub repo**
3. Add environment variable: `GEMINI_API_KEY=your_key`
4. Railway auto-detects Python and runs `main.py`
5. Get your public URL from the **Settings** tab

---

## 📁 Project Structure

```
crypto-ai-agent/
├── main.py            # Main agent loop
├── config.py          # All settings & env vars
├── scanner.py         # CoinGecko market data fetcher
├── indicators.py      # RSI, MA, momentum, volatility
├── pump_detector.py   # Scoring engine (0–100)
├── trend_detector.py  # Multi-cycle trend analysis
├── ai_engine.py       # Gemini AI evaluation
├── learning_engine.py # SQLite storage + self-learning weights
├── dashboard.py       # Flask web dashboard
├── alerts.py          # Console display
├── backtester.py      # Historical backtesting
├── data/              # SQLite DB + weights.json (auto-created)
├── logs/              # Log files (auto-created)
├── requirements.txt
├── render.yaml        # Render deployment config
├── .env.example       # Template for your secrets
└── README.md
```

---

## 🧠 How Self-Learning Works

Every scan cycle the agent:
1. Checks all **PENDING** signals — did price hit target (+15%) or stop-loss (-5%)?
2. Labels each signal **WIN**, **LOSS**, or **NEUTRAL**
3. If win rate > 60% → slightly increases scoring weights (more aggressive)
4. If win rate < 40% → slightly decreases weights (more conservative)
5. Saves updated weights to `data/weights.json`

The scoring weights start at:
- Volume spike: **40**
- Price change: **25**
- Momentum breakout: **20**
- Small cap: **15**

These adjust automatically over time based on real performance.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Free from aistudio.google.com |
| `COINGECKO_API_KEY` | ❌ No | Leave blank (free tier works) |
| `COINDCX_PREFERRED_MARGIN_ASSET` | ❌ No | Preferred CoinDCX futures margin asset, default `USDT` |
| `COINDCX_RECOMMENDED_LEVERAGE_CAP` | ❌ No | Hard cap for suggested leverage, default `3` |
| `PORT` | ❌ No | Set automatically by Render/Railway |

---

## ⚙️ Configuration

Edit `config.py` to change:
- `COINS_PER_CYCLE` — how many coins to scan (default: 500)
- `SCAN_INTERVAL_SECONDS` — time between cycles (default: 300 = 5 min)
- `OPPORTUNITY_SCORE_THRESHOLD` — minimum score to flag (default: 70)
- `DEFAULT_TARGET_PCT` — take-profit % (default: 15%)
- `DEFAULT_STOP_LOSS_PCT` — stop-loss % (default: 5%)
