"""
dashboard.py — Lightweight Flask web dashboard.

Serves a live-updating HTML page showing:
  - Agent stats (win rate, total signals)
  - Latest signals table
  - Current weights

Runs on its own thread so it doesn't block the main scan loop.
"""

import logging
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template_string

from learning_engine import get_recent_signals, get_stats, load_weights
from config import DASHBOARD_HOST, DASHBOARD_PORT

logger = logging.getLogger(__name__)
app = Flask(__name__)

# ── HTML template (single-file, auto-refresh every 30s) ───────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta http-equiv="refresh" content="30"/>
  <title>🤖 Crypto AI Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0d1117; color: #c9d1d9;
      padding: 24px;
    }
    h1 { color: #58a6ff; font-size: 1.6rem; margin-bottom: 4px; }
    .subtitle { color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }
    .warning {
      background: #3d1f00; border: 1px solid #f0883e;
      color: #f0883e; padding: 10px 16px; border-radius: 8px;
      margin-bottom: 20px; font-size: 0.85rem;
    }
    .stats-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px; margin-bottom: 24px;
    }
    .stat-card {
      background: #161b22; border: 1px solid #30363d;
      border-radius: 10px; padding: 16px; text-align: center;
    }
    .stat-card .val { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
    .stat-card .lbl { font-size: 0.75rem; color: #8b949e; margin-top: 4px; }
    .win  { color: #3fb950 !important; }
    .loss { color: #f85149 !important; }
    .hold { color: #d29922 !important; }
    table {
      width: 100%; border-collapse: collapse;
      background: #161b22; border-radius: 10px; overflow: hidden;
      font-size: 0.82rem;
    }
    th {
      background: #21262d; color: #8b949e;
      padding: 10px 12px; text-align: left;
      border-bottom: 1px solid #30363d;
    }
    td { padding: 9px 12px; border-bottom: 1px solid #21262d; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #1c2128; }
    .badge {
      padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
    }
    .badge-buy    { background: #0d4429; color: #3fb950; }
    .badge-avoid  { background: #3d0b0b; color: #f85149; }
    .badge-hold   { background: #2d2106; color: #d29922; }
    .badge-win    { background: #0d4429; color: #3fb950; }
    .badge-loss   { background: #3d0b0b; color: #f85149; }
    .badge-pending{ background: #1c2631; color: #58a6ff; }
    .badge-neutral{ background: #21262d; color: #8b949e; }
    .section-title {
      color: #8b949e; font-size: 0.8rem; font-weight: 600;
      text-transform: uppercase; letter-spacing: 1px;
      margin: 24px 0 10px;
    }
    .weights { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
    .weight-pill {
      background: #21262d; border: 1px solid #30363d;
      border-radius: 20px; padding: 4px 14px; font-size: 0.78rem;
    }
    .weight-pill span { color: #58a6ff; font-weight: 700; }
    .hero-card {
      background: linear-gradient(135deg, #182235, #142033);
      border: 1px solid #2f4868; border-radius: 12px;
      padding: 18px; margin-bottom: 24px;
    }
    .hero-title {
      color: #8b949e; font-size: 0.78rem; text-transform: uppercase;
      letter-spacing: 1px; margin-bottom: 10px; font-weight: 700;
    }
    .hero-coin {
      display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .hero-coin strong { font-size: 1.2rem; color: #f0f6fc; }
    .hero-metrics {
      display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px;
      color: #c9d1d9; font-size: 0.88rem;
    }
    .hero-reason { color: #8b949e; font-size: 0.84rem; line-height: 1.5; }
    .refresh-note { color: #484f58; font-size: 0.75rem; margin-top: 20px; }
  </style>
</head>
<body>
  <h1>🤖 Crypto AI Agent</h1>
  <p class="subtitle">Live market analysis · Auto-refreshes every 30 seconds</p>

  <div class="warning">
    ⚠️ <strong>DISCLAIMER:</strong> This tool provides analysis ONLY and does NOT execute trades.
    All signals are suggestions. Crypto is high risk — always do your own research.
  </div>

  <!-- Stats -->
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

  <!-- Adaptive weights -->
  <div class="section-title">📊 Adaptive Scoring Weights</div>
  <div class="weights">
    {% for k, v in weights.items() %}
    <div class="weight-pill">{{ k.replace('_', ' ').title() }}: <span>{{ v }}</span></div>
    {% endfor %}
  </div>

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
      <span>Entry: ${{ "%.4f"|format(best_signal.entry_price) }}</span>
      <span>Target: ${{ "%.4f"|format(best_signal.target_price) }}</span>
      <span>Confidence: {{ best_signal.confidence | int }}%</span>
      <span>Score: {{ best_signal.pump_score | int }}/100</span>
      <span>Quality: {{ (best_signal.quality_score or best_signal.pump_score) | int }}/100</span>
      <span>Final: {{ (best_signal.aggregate_score or best_signal.quality_score or best_signal.pump_score) | int }}/100</span>
      <span>Regime: {{ best_signal.market_regime or "UNKNOWN" }}</span>
    </div>
    <p class="hero-reason">{{ best_signal.ai_reason }}</p>
  </div>
  {% endif %}

  <!-- Signals table -->
  <div class="section-title">🔔 Latest Signals</div>
  {% if signals %}
  <table>
    <thead>
      <tr>
        <th>Coin</th>
        <th>Action</th>
        <th>Entry</th>
        <th>Target</th>
        <th>Stop Loss</th>
        <th>Confidence</th>
        <th>Score</th>
        <th>Quality</th>
        <th>Final</th>
        <th>Regime</th>
        <th>Outcome</th>
        <th>Time (UTC)</th>
      </tr>
    </thead>
    <tbody>
      {% for s in signals %}
      <tr>
        <td><strong>{{ s.symbol }}</strong><br><small style="color:#8b949e">{{ s.coin }}</small></td>
        <td>
          <span class="badge badge-{{ s.ai_action | lower }}">{{ s.ai_action }}</span>
        </td>
        <td>${{ "%.4f"|format(s.entry_price) }}</td>
        <td>${{ "%.4f"|format(s.target_price) }}</td>
        <td>${{ "%.4f"|format(s.stop_loss) }}</td>
        <td>{{ s.confidence | int }}%</td>
        <td>{{ s.pump_score | int }}/100</td>
        <td>{{ (s.quality_score or s.pump_score) | int }}/100</td>
        <td>{{ (s.aggregate_score or s.quality_score or s.pump_score) | int }}/100</td>
        <td>{{ s.market_regime or "UNKNOWN" }}</td>
        <td>
          <span class="badge badge-{{ s.outcome | lower }}">{{ s.outcome }}</span>
        </td>
        <td style="color:#8b949e;font-size:0.75rem">{{ s.timestamp[:19] }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <p style="color:#8b949e; padding:20px 0">
    No signals yet — the agent is scanning the market. Check back in a few minutes.
  </p>
  {% endif %}

  <p class="refresh-note">Last render: {{ now }} UTC · Page auto-refreshes every 30s</p>
</body>
</html>
"""


@app.route("/")
def index():
    signals = get_recent_signals(50)
    stats   = get_stats()
    weights = load_weights()
    best_signal = _pick_best_signal(signals)
    now     = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return render_template_string(
        HTML,
        signals=signals,
        stats=stats,
        weights=weights,
        best_signal=best_signal,
        now=now,
    )


@app.route("/api/signals")
def api_signals():
    return jsonify(get_recent_signals(50))


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


def _pick_best_signal(signals):
    pending = [s for s in signals if s.get("outcome") == "PENDING"]
    pool = pending or signals
    if not pool:
        return None

    def rank(signal):
        action_priority = {"BUY": 2, "HOLD": 1, "AVOID": 0}.get(signal.get("ai_action"), 0)
        return (
            action_priority,
            float(signal.get("aggregate_score") or 0),
            float(signal.get("quality_score") or 0),
            float(signal.get("confidence", 0)),
            float(signal.get("pump_score", 0)),
            signal.get("timestamp", ""),
        )

    return max(pool, key=rank)


def start_dashboard() -> None:
    """Start the Flask dashboard in a background daemon thread."""
    def _run():
        import os
        # Suppress Flask startup messages in production
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
        app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True, name="dashboard")
    t.start()
    logger.info("Dashboard started at http://%s:%d", DASHBOARD_HOST, DASHBOARD_PORT)
