"""
dashboard.py - Lightweight Flask web dashboard.

Serves a live-updating HTML page showing:
  - agent stats
  - latest published signals
  - persistent signal history with outcomes
  - futures context for latest signals
  - adaptive weights
"""

import logging
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template_string, request

from config import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    HISTORY_SIGNALS_LIMIT,
    LATEST_SIGNALS_LIMIT,
)
from learning_engine import (
    enrich_signal_simulation,
    get_recent_signals,
    get_simulation_stats,
    get_stats,
    get_storage_backend_name,
    load_weights,
)

logger = logging.getLogger(__name__)
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta http-equiv="refresh" content="30"/>
  <title>Crypto AI Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Segoe UI", sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      padding: 24px;
    }
    h1 { color: #58a6ff; font-size: 1.7rem; margin-bottom: 4px; }
    .subtitle { color: #8b949e; font-size: 0.9rem; margin-bottom: 24px; }
    .warning {
      background: #3d1f00;
      border: 1px solid #f0883e;
      color: #f0883e;
      padding: 10px 16px;
      border-radius: 8px;
      margin-bottom: 20px;
      font-size: 0.85rem;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .stat-card, .mini-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 16px;
    }
    .stat-card { text-align: center; }
    .stat-card .val { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .stat-card .lbl { font-size: 0.76rem; color: #8b949e; margin-top: 4px; }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .mini-card .k {
      color: #8b949e;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
    }
    .mini-card .v {
      color: #f0f6fc;
      font-size: 1.05rem;
      font-weight: 700;
      margin-top: 8px;
    }
    .win { color: #3fb950 !important; }
    .loss { color: #f85149 !important; }
    .section-title {
      color: #8b949e;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin: 24px 0 10px;
    }
    .tab-bar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .tab-btn {
      background: #161b22;
      border: 1px solid #30363d;
      color: #c9d1d9;
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
      font-size: 0.84rem;
    }
    .tab-btn.active {
      background: #1f6feb;
      border-color: #1f6feb;
      color: #ffffff;
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .table-wrap {
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid #30363d;
      background: #161b22;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #161b22;
      font-size: 0.82rem;
    }
    th {
      background: #21262d;
      color: #8b949e;
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid #30363d;
      white-space: nowrap;
    }
    td {
      padding: 9px 12px;
      border-bottom: 1px solid #21262d;
      white-space: nowrap;
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #1c2128; }
    .badge {
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .badge-buy { background: #0d4429; color: #3fb950; }
    .badge-short { background: #3d0b0b; color: #ff7b72; }
    .badge-avoid { background: #3d0b0b; color: #f85149; }
    .badge-hold { background: #2d2106; color: #d29922; }
    .badge-win { background: #0d4429; color: #3fb950; }
    .badge-loss { background: #3d0b0b; color: #f85149; }
    .badge-pending { background: #1c2631; color: #58a6ff; }
    .badge-neutral { background: #21262d; color: #8b949e; }
    .weights {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }
    .weight-pill {
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 20px;
      padding: 4px 14px;
      font-size: 0.78rem;
    }
    .weight-pill span { color: #58a6ff; font-weight: 700; }
    .hero-card {
      background: linear-gradient(135deg, #182235, #142033);
      border: 1px solid #2f4868;
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 24px;
    }
    .hero-title {
      color: #8b949e;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 10px;
      font-weight: 700;
    }
    .hero-coin {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .hero-coin strong { font-size: 1.2rem; color: #f0f6fc; }
    .hero-metrics {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 10px;
      color: #c9d1d9;
      font-size: 0.88rem;
    }
    .hero-reason { color: #8b949e; font-size: 0.84rem; line-height: 1.5; }
    .note {
      color: #8b949e;
      font-size: 0.82rem;
      line-height: 1.5;
      margin-bottom: 14px;
    }
    .refresh-note { color: #484f58; font-size: 0.75rem; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>Crypto AI Agent</h1>
  <p class="subtitle">Live market analysis · Auto-refreshes every 30 seconds</p>

  <div class="warning">
    <strong>Disclaimer:</strong> This tool provides analysis only and does not execute trades.
    All signals are suggestions. Crypto is high risk, so always do your own research.
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="val">{{ stats.total }}</div>
      <div class="lbl">Total Signals</div>
    </div>
    <div class="stat-card">
      <div class="val win">{{ stats.win_rate }}%</div>
      <div class="lbl">Win Rate</div>
    </div>
    <div class="stat-card">
      <div class="val win">{{ stats.wins }}</div>
      <div class="lbl">Wins</div>
    </div>
    <div class="stat-card">
      <div class="val loss">{{ stats.losses }}</div>
      <div class="lbl">Losses</div>
    </div>
    <div class="stat-card">
      <div class="val">{{ stats.pending }}</div>
      <div class="lbl">Pending</div>
    </div>
  </div>

  <div class="mini-grid">
    <div class="mini-card">
      <div class="k">Storage Backend</div>
      <div class="v">{{ storage_backend }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Closed Signals</div>
      <div class="v">{{ stats.closed }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Latest Loaded</div>
      <div class="v">{{ latest_signals|length }}</div>
    </div>
    <div class="mini-card">
      <div class="k">History Loaded</div>
      <div class="v">{{ history_signals|length }}</div>
    </div>
  </div>

  <div class="section-title">Paper Simulation</div>
  <div class="mini-grid">
    <div class="mini-card">
      <div class="k">Stake / Signal</div>
      <div class="v">{{ fmt_inr(simulation.stake_inr) }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Total Invested</div>
      <div class="v">{{ fmt_inr(simulation.total_invested_inr) }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Current Value</div>
      <div class="v">{{ fmt_inr(simulation.current_value_inr) }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Total P&amp;L</div>
      <div class="v {{ 'win' if simulation.total_pnl_inr >= 0 else 'loss' }}">{{ fmt_inr(simulation.total_pnl_inr, show_sign=True) }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Realized P&amp;L</div>
      <div class="v {{ 'win' if simulation.realized_pnl_inr >= 0 else 'loss' }}">{{ fmt_inr(simulation.realized_pnl_inr, show_sign=True) }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Unrealized P&amp;L</div>
      <div class="v {{ 'win' if simulation.unrealized_pnl_inr >= 0 else 'loss' }}">{{ fmt_inr(simulation.unrealized_pnl_inr, show_sign=True) }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Portfolio ROI</div>
      <div class="v {{ 'win' if simulation.roi_pct >= 0 else 'loss' }}">{{ fmt_pct(simulation.roi_pct, 2) }}</div>
    </div>
    <div class="mini-card">
      <div class="k">Open Paper Trades</div>
      <div class="v">{{ simulation.open_trades }}</div>
    </div>
  </div>

  <div class="tab-bar">
    <button class="tab-btn active" data-tab="latest">Latest</button>
    <button class="tab-btn" data-tab="simulation">Simulation</button>
    <button class="tab-btn" data-tab="history">History</button>
    <button class="tab-btn" data-tab="futures">Futures Layer</button>
    <button class="tab-btn" data-tab="weights">Weights</button>
  </div>

  <div id="tab-latest" class="tab-panel active">
    {% if best_signal %}
    <div class="hero-card">
      <div class="hero-title">Best Current Signal</div>
      <div class="hero-coin">
        <strong>{{ best_signal.symbol }}</strong>
        <span>{{ best_signal.coin }}</span>
        <span class="badge badge-{{ best_signal.ai_action | lower }}">{{ best_signal.ai_action }}</span>
        <span class="badge badge-{{ best_signal.outcome | lower }}">{{ best_signal.outcome }}</span>
      </div>
      <div class="hero-metrics">
        <span>Entry: {{ fmt_money(best_signal.entry_price) }}</span>
        <span>Target: {{ fmt_money(best_signal.target_price) }}</span>
        <span>Stop: {{ fmt_money(best_signal.stop_loss) }}</span>
        <span>Paper Value: {{ fmt_inr(best_signal.simulation_value_inr) }}</span>
        <span>P&amp;L: {{ fmt_inr(best_signal.simulation_pnl_inr, show_sign=True) }}</span>
        <span>Confidence: {{ best_signal.confidence | int }}%</span>
        <span>Final: {{ (best_signal.aggregate_score or best_signal.quality_score or best_signal.pump_score) | int }}/100</span>
        <span>Regime: {{ best_signal.market_regime or "UNKNOWN" }}</span>
        <span>Setup: {{ best_signal.futures_bias or "UNAVAILABLE" }}</span>
        <span>Suggested Lev: {{ best_signal.leverage_hint or "Unavailable" }}</span>
      </div>
      <p class="hero-reason">{{ best_signal.ai_reason }}</p>
    </div>
    {% endif %}

    <div class="section-title">Latest Signals</div>
    <p class="note">
      These are the highest-conviction signals being published now. Older signals do not disappear;
      they move into history below and remain visible with their outcome.
    </p>
    {% if latest_signals %}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Time (UTC)</th>
            <th>Coin</th>
            <th>Action</th>
            <th>Entry</th>
            <th>Target</th>
            <th>Stop</th>
            <th>Confidence</th>
            <th>Final</th>
            <th>Paper P&amp;L</th>
            <th>Paper Value</th>
            <th>Setup</th>
            <th>Suggested Lev</th>
            <th>Funding</th>
            <th>OI</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {% for s in latest_signals %}
          <tr>
            <td style="color:#8b949e">{{ fmt_timestamp(s.timestamp) }}</td>
            <td><strong>{{ s.symbol }}</strong><br><small style="color:#8b949e">{{ s.coin }}</small></td>
            <td><span class="badge badge-{{ s.ai_action | lower }}">{{ s.ai_action }}</span></td>
            <td>{{ fmt_money(s.entry_price) }}</td>
            <td>{{ fmt_money(s.target_price) }}</td>
            <td>{{ fmt_money(s.stop_loss) }}</td>
            <td>{{ s.confidence | int }}%</td>
            <td>{{ (s.aggregate_score or s.quality_score or s.pump_score) | int }}/100</td>
            <td class="{{ 'win' if s.simulation_pnl_inr >= 0 else 'loss' }}">{{ fmt_inr(s.simulation_pnl_inr, show_sign=True) }}</td>
            <td>{{ fmt_inr(s.simulation_value_inr) }}</td>
            <td>{{ s.futures_bias or "UNAVAILABLE" }}</td>
            <td>{{ s.leverage_hint or "Unavailable" }}</td>
            <td>{{ fmt_pct(s.funding_rate, 4) }}</td>
            <td>{{ fmt_compact(s.open_interest) }}</td>
            <td><span class="badge badge-{{ s.outcome | lower }}">{{ s.outcome }}</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <p class="note">No live signals yet. The agent is still scanning the market.</p>
    {% endif %}
  </div>

  <div id="tab-simulation" class="tab-panel">
    <div class="section-title">Paper Trade History</div>
    <p class="note">
      Every published signal is treated as a fixed {{ fmt_inr(simulation.stake_inr) }} paper trade.
      Closed trades use the recorded exit market price. Open trades are marked to the latest seen market price.
    </p>
    {% if history_signals %}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Time (UTC)</th>
            <th>Coin</th>
            <th>Action</th>
            <th>Outcome</th>
            <th>Entry</th>
            <th>Exit / Mark</th>
            <th>Return</th>
            <th>Stake</th>
            <th>P&amp;L</th>
            <th>Value</th>
            <th>Age</th>
          </tr>
        </thead>
        <tbody>
          {% for s in history_signals %}
          <tr>
            <td style="color:#8b949e">{{ fmt_timestamp(s.timestamp) }}</td>
            <td><strong>{{ s.symbol }}</strong><br><small style="color:#8b949e">{{ s.coin }}</small></td>
            <td><span class="badge badge-{{ s.ai_action | lower }}">{{ s.ai_action }}</span></td>
            <td><span class="badge badge-{{ s.outcome | lower }}">{{ s.outcome }}</span></td>
            <td>{{ fmt_money(s.entry_price) }}</td>
            <td>{{ fmt_money(s.simulation_exit_price) }}</td>
            <td class="{{ 'win' if s.simulation_return_pct >= 0 else 'loss' }}">{{ fmt_pct(s.simulation_return_pct, 2) }}</td>
            <td>{{ fmt_inr(s.simulation_stake_inr) }}</td>
            <td class="{{ 'win' if s.simulation_pnl_inr >= 0 else 'loss' }}">{{ fmt_inr(s.simulation_pnl_inr, show_sign=True) }}</td>
            <td>{{ fmt_inr(s.simulation_value_inr) }}</td>
            <td>{{ fmt_age(s.simulation_age_hours) }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <p class="note">No paper trades yet. Once the agent publishes signals, this tab will start compounding the simulated history.</p>
    {% endif %}
  </div>

  <div id="tab-history" class="tab-panel">
    <div class="section-title">Signal History</div>
    <p class="note">
      This history is the learning memory of the agent. Every saved signal stays here with its latest
      outcome, so you can review what happened even after it leaves the latest list.
    </p>
    {% if history_signals %}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Time (UTC)</th>
            <th>Coin</th>
            <th>Action</th>
            <th>Setup</th>
            <th>Suggested Lev</th>
            <th>Entry</th>
            <th>Target</th>
            <th>Stop</th>
            <th>Final</th>
            <th>Funding</th>
            <th>OI</th>
            <th>Basis</th>
            <th>Spread</th>
            <th>Outcome</th>
            <th>Outcome Px</th>
          </tr>
        </thead>
        <tbody>
          {% for s in history_signals %}
          <tr>
            <td style="color:#8b949e">{{ fmt_timestamp(s.timestamp) }}</td>
            <td><strong>{{ s.symbol }}</strong><br><small style="color:#8b949e">{{ s.coin }}</small></td>
            <td><span class="badge badge-{{ s.ai_action | lower }}">{{ s.ai_action }}</span></td>
            <td>{{ s.futures_bias or "UNAVAILABLE" }}</td>
            <td>{{ s.leverage_hint or "Unavailable" }}</td>
            <td>{{ fmt_money(s.entry_price) }}</td>
            <td>{{ fmt_money(s.target_price) }}</td>
            <td>{{ fmt_money(s.stop_loss) }}</td>
            <td>{{ (s.aggregate_score or s.quality_score or s.pump_score) | int }}/100</td>
            <td>{{ fmt_pct(s.funding_rate, 4) }}</td>
            <td>{{ fmt_compact(s.open_interest) }}</td>
            <td>{{ fmt_pct(s.basis, 2) }}</td>
            <td>{{ fmt_pct(s.spread, 2) }}</td>
            <td><span class="badge badge-{{ s.outcome | lower }}">{{ s.outcome }}</span></td>
            <td>{{ fmt_money(s.outcome_price) if s.outcome_price else "-" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <p class="note">History is empty right now. Once signals are saved, they will remain visible here.</p>
    {% endif %}
  </div>

  <div id="tab-futures" class="tab-panel">
    <div class="section-title">Futures Layer</div>
    <p class="note">
      This section shows the derivatives confirmation being stored with current signals:
      `LONG` means futures support a long setup, `SHORT` means futures support a short setup,
      `WAIT` means stand aside for now, and `UNAVAILABLE` means live CoinDCX futures
      confirmation is missing.
    </p>
    {% if latest_signals %}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Coin</th>
            <th>Setup</th>
            <th>Leverage</th>
            <th>Futures Score</th>
            <th>Funding</th>
            <th>Open Interest</th>
            <th>Basis</th>
            <th>Spread</th>
            <th>Futures Vol 24h</th>
            <th>Exchange</th>
          </tr>
        </thead>
        <tbody>
          {% for s in latest_signals %}
          <tr>
            <td><strong>{{ s.symbol }}</strong><br><small style="color:#8b949e">{{ s.coin }}</small></td>
            <td>{{ s.futures_bias or "UNAVAILABLE" }}</td>
            <td>{{ s.leverage_hint or "Unavailable" }}</td>
            <td>{{ (s.futures_score or 0) | int }}/100</td>
            <td>{{ fmt_pct(s.funding_rate, 4) }}</td>
            <td>{{ fmt_compact(s.open_interest) }}</td>
            <td>{{ fmt_pct(s.basis, 2) }}</td>
            <td>{{ fmt_pct(s.spread, 2) }}</td>
            <td>{{ fmt_compact(s.futures_volume_24h) }}</td>
            <td>{{ s.futures_exchange or "-" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <p class="note">No futures-confirmed signals have been published yet.</p>
    {% endif %}
  </div>

  <div id="tab-weights" class="tab-panel">
    <div class="section-title">Adaptive Scoring Weights</div>
    <div class="weights">
      {% for k, v in weights.items() %}
      <div class="weight-pill">{{ k.replace('_', ' ').title() }}: <span>{{ v }}</span></div>
      {% endfor %}
    </div>
  </div>

  <p class="refresh-note">Last render: {{ now }} UTC · Page auto-refreshes every 30s</p>
  <script>
    const activateTab = (tabName) => {
      document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tabName);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tab-${tabName}`);
      });
      window.location.hash = tabName;
    };

    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => activateTab(btn.dataset.tab));
    });

    const initialTab = window.location.hash ? window.location.hash.slice(1) : "latest";
    activateTab(initialTab);
  </script>
</body>
</html>
"""


def _format_money(value):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if abs(value) >= 1000:
        return f"${value:,.2f}"
    if abs(value) >= 1:
        return f"${value:,.4f}"
    return f"${value:,.6f}"


def _format_inr(value, show_sign=False):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    prefix = "+" if show_sign and value > 0 else ""
    return f"{prefix}Rs {value:,.2f}"


def _format_compact(value):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def _format_pct(value, digits=2):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{value:.{digits}f}%"


def _format_timestamp(value):
    if not value:
        return "-"
    return str(value)[:19]


def _format_age(hours):
    try:
        hours = float(hours)
    except (TypeError, ValueError):
        return "-"
    total_minutes = max(0, int(round(hours * 60)))
    days, rem_minutes = divmod(total_minutes, 1440)
    rem_hours, minutes = divmod(rem_minutes, 60)
    if days > 0:
        return f"{days}d {rem_hours}h"
    if rem_hours > 0:
        return f"{rem_hours}h {minutes}m"
    return f"{minutes}m"


def _normalise_signal(signal: dict) -> dict:
    merged = dict(signal)
    bias = str(merged.get("futures_bias") or "").upper()
    if bias == "NO-DATA":
        merged["futures_bias"] = "UNAVAILABLE"
    elif bias == "NO-TRADE":
        merged["futures_bias"] = "WAIT"
    else:
        merged.setdefault("futures_bias", "UNAVAILABLE")

    leverage_hint = str(merged.get("leverage_hint") or "").strip()
    if merged.get("futures_bias") == "UNAVAILABLE" and (not leverage_hint or leverage_hint == "1x"):
        merged["leverage_hint"] = "Unavailable"
    elif merged.get("futures_bias") == "WAIT" and (not leverage_hint or leverage_hint == "1x"):
        merged["leverage_hint"] = "Wait"
    elif not leverage_hint:
        merged["leverage_hint"] = "Unavailable"
    else:
        merged.setdefault("leverage_hint", leverage_hint)

    merged.setdefault("futures_exchange", "")
    merged.setdefault("futures_symbol", "")
    merged.setdefault("funding_rate", 0.0)
    merged.setdefault("open_interest", 0.0)
    merged.setdefault("basis", 0.0)
    merged.setdefault("spread", 0.0)
    merged.setdefault("futures_volume_24h", 0.0)
    merged.setdefault("futures_score", 0.0)
    merged.setdefault("quality_score", merged.get("pump_score", 0.0))
    merged.setdefault("aggregate_score", merged.get("quality_score", merged.get("pump_score", 0.0)))
    merged.setdefault("market_regime", "UNKNOWN")
    return merged


def _is_live_trade_signal(signal: dict) -> bool:
    action = str(signal.get("ai_action") or "").upper()
    bias = str(signal.get("futures_bias") or "").upper()
    return (action == "BUY" and bias == "LONG") or (action == "SHORT" and bias == "SHORT")


@app.route("/")
def index():
    recent_pool = [
        enrich_signal_simulation(_normalise_signal(row))
        for row in get_recent_signals(max(LATEST_SIGNALS_LIMIT * 5, 40))
    ]
    latest_signals = [row for row in recent_pool if _is_live_trade_signal(row)][:LATEST_SIGNALS_LIMIT]
    history_signals = [
        enrich_signal_simulation(_normalise_signal(row))
        for row in get_recent_signals(HISTORY_SIGNALS_LIMIT)
    ]
    stats = get_stats()
    simulation = get_simulation_stats()
    weights = load_weights()
    best_signal = _pick_best_signal(latest_signals)
    storage_backend = get_storage_backend_name()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return render_template_string(
        HTML,
        latest_signals=latest_signals,
        history_signals=history_signals,
        stats=stats,
        simulation=simulation,
        weights=weights,
        best_signal=best_signal,
        storage_backend=storage_backend,
        fmt_money=_format_money,
        fmt_inr=_format_inr,
        fmt_compact=_format_compact,
        fmt_pct=_format_pct,
        fmt_age=_format_age,
        fmt_timestamp=_format_timestamp,
        now=now,
    )


@app.route("/api/signals")
def api_signals():
    limit = request.args.get("limit", default=LATEST_SIGNALS_LIMIT, type=int)
    pool = [
        enrich_signal_simulation(_normalise_signal(row))
        for row in get_recent_signals(max(limit * 5, 40))
    ]
    return jsonify([row for row in pool if _is_live_trade_signal(row)][:limit])


@app.route("/api/history")
def api_history():
    limit = request.args.get("limit", default=HISTORY_SIGNALS_LIMIT, type=int)
    offset = request.args.get("offset", default=0, type=int)
    outcome = request.args.get("outcome", default=None, type=str)
    return jsonify(
        [
            enrich_signal_simulation(_normalise_signal(row))
            for row in get_recent_signals(limit=limit, offset=offset, outcome=outcome)
        ]
    )


@app.route("/api/stats")
def api_stats():
    payload = get_stats()
    payload["simulation"] = get_simulation_stats()
    return jsonify(payload)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "storage_backend": get_storage_backend_name()})


def _pick_best_signal(signals):
    pending = [signal for signal in signals if signal.get("outcome") == "PENDING"]
    pool = pending or signals
    if not pool:
        return None

    def rank(signal):
        action_priority = {"BUY": 2, "SHORT": 2, "HOLD": 1, "AVOID": 0}.get(signal.get("ai_action"), 0)
        bias_priority = {
            "LONG": 2,
            "SHORT": 2,
            "WAIT": 1,
            "UNAVAILABLE": 0,
        }.get(signal.get("futures_bias"), 0)
        return (
            action_priority,
            bias_priority,
            float(signal.get("aggregate_score") or 0),
            float(signal.get("futures_score") or 0),
            float(signal.get("quality_score") or 0),
            float(signal.get("confidence", 0)),
            float(signal.get("pump_score", 0)),
            signal.get("timestamp", ""),
        )

    return max(pool, key=rank)


def start_dashboard() -> None:
    """Start the Flask dashboard in a background daemon thread."""

    def _run():
        import logging as _logging

        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="dashboard")
    thread.start()
    logger.info("Dashboard started at http://%s:%d", DASHBOARD_HOST, DASHBOARD_PORT)
