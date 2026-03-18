import streamlit as st
import streamlit.components.v1
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os, time, urllib.request, xml.etree.ElementTree as ET
import yfinance as yf
import sqlite3, hashlib, json, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

from data_loader import load_data
from preprocessing import scale_data, create_sequences
from model import build_model
from train import train_model
from predict import make_predictions, next_day_prediction, forecast_n_days


# ══════════════════════════════════════════════════════════════════
#  PERSISTENT STORAGE — SQLite
# ══════════════════════════════════════════════════════════════════
import tempfile, platform
# Cloud-compatible paths — use temp dir if home not writable
_BASE = Path.home() / ".tradevision"
try:
    _BASE.mkdir(exist_ok=True)
    (_BASE / "test").touch()
    (_BASE / "test").unlink()
except:
    _BASE = Path(tempfile.gettempdir()) / ".tradevision"
    _BASE.mkdir(exist_ok=True)

DB_PATH = str(_BASE / "tradevision.db")

def init_db():
    """Create tables if not exist."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY, ticker TEXT UNIQUE)""")
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY, ticker TEXT UNIQUE,
        shares REAL, buy_price REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY, ticker TEXT,
        price REAL, direction TEXT,
        triggered INTEGER DEFAULT 0,
        created TEXT, email TEXT DEFAULT '')""")
    conn.commit(); conn.close()

def db_get_watchlist():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT ticker FROM watchlist").fetchall()
    conn.close()
    return [r[0] for r in rows]

def db_add_watchlist(ticker):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT OR IGNORE INTO watchlist (ticker) VALUES (?)", (ticker,))
        conn.commit()
    except: pass
    conn.close()

def db_remove_watchlist(ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))
    conn.commit(); conn.close()

def db_get_portfolio():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT ticker, shares, buy_price FROM portfolio").fetchall()
    conn.close()
    return {r[0]: {"shares": r[1], "buy_price": r[2]} for r in rows}

def db_add_portfolio(ticker, shares, buy_price):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT OR REPLACE INTO portfolio (ticker, shares, buy_price)
        VALUES (?,?,?)""", (ticker, shares, buy_price))
    conn.commit(); conn.close()

def db_remove_portfolio(ticker):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM portfolio WHERE ticker=?", (ticker,))
    conn.commit(); conn.close()

def db_get_alerts():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, ticker, price, direction, triggered, created, email FROM alerts"
    ).fetchall()
    conn.close()
    return [{"id":r[0],"ticker":r[1],"price":r[2],"direction":r[3],
             "triggered":bool(r[4]),"created":r[5],"email":r[6]} for r in rows]

def db_add_alert(ticker, price, direction, email=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO alerts (ticker, price, direction, created, email)
        VALUES (?,?,?,?,?)""",
        (ticker, price, direction, datetime.now().strftime("%d %b %H:%M"), email))
    conn.commit(); conn.close()

def db_trigger_alert(alert_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE alerts SET triggered=1 WHERE id=?", (alert_id,))
    conn.commit(); conn.close()

def db_delete_alert(alert_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    conn.commit(); conn.close()

def db_clear_alerts():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM alerts")
    conn.commit(); conn.close()


# ══════════════════════════════════════════════════════════════════
#  EMAIL ALERTS
# ══════════════════════════════════════════════════════════════════
def get_secret(key, default=""):
    """Get secret from Streamlit secrets OR environment variable."""
    try:
        val = st.secrets.get(key, "")
        if val: return val
    except: pass
    return os.environ.get(key, default)

def send_alert_email(to_email, ticker, direction, target_price, current_price, cur):
    """Send price alert email via Gmail SMTP."""
    gmail_user = get_secret("GMAIL_USER")
    gmail_pass = get_secret("GMAIL_APP_PASSWORD")
    if not gmail_user or not gmail_pass:
        return False, "GMAIL_USER ya GMAIL_APP_PASSWORD secrets.toml ma nathi"

    arrow = "▲" if direction == "above" else "▼"
    subject = f"🔔 TradeVision Alert: {ticker} {arrow} {cur}{target_price:,.2f}"
    body = f"""
    <html><body style="font-family:Arial;background:#0b111a;color:#f1f5f9;padding:24px;">
    <div style="max-width:500px;margin:0 auto;background:#131b27;border-radius:16px;
        padding:28px;border:1px solid rgba(0,255,157,0.2);">
        <div style="font-size:24px;font-weight:700;color:#00ff9d;margin-bottom:8px;">
            🔔 Price Alert Triggered!</div>
        <div style="font-size:14px;color:#94a3b8;margin-bottom:20px;">
            TradeVision AI · {datetime.now().strftime('%d %b %Y %H:%M')}</div>
        <div style="background:#0b111a;border-radius:12px;padding:20px;margin-bottom:16px;">
            <div style="font-size:28px;font-weight:700;color:#f1f5f9;">{ticker}</div>
            <div style="font-size:20px;color:{'#00ff9d' if direction=='above' else '#ff0055'};margin-top:4px;">
                {arrow} Crossed {cur}{target_price:,.2f}</div>
            <div style="font-size:14px;color:#64748b;margin-top:6px;">
                Current Price: {cur}{current_price:,.2f}</div>
        </div>
        <div style="font-size:11px;color:#475569;text-align:center;">
            This is an automated alert from TradeVision AI.<br>
            Not financial advice. Always DYOR.</div>
    </div></body></html>
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = gmail_user
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to_email, msg.as_string())
        return True, "✅ Email sent!"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Gmail login failed — App Password check karo"
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP Error: {str(e)[:80]}"
    except Exception as e:
        return False, f"❌ Error: {str(e)[:80]}"

# Initialise DB on startup
init_db()

# Sync DB → session state on first load
def sync_db_to_session():
    if not st.session_state.get("db_synced"):
        st.session_state.watchlist = db_get_watchlist()
        st.session_state.portfolio = db_get_portfolio()
        st.session_state.alerts    = db_get_alerts()
        st.session_state.db_synced = True

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TradeVision AI", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════
def _ss(k, v):
    if k not in st.session_state: st.session_state[k] = v
_ss("ticker",       "")
_ss("analyzed",     False)
_ss("model_type",   "SimpleRNN")
_ss("dark_mode",    True)
_ss("horizon_val",  5)
_ss("live_feed_on", False)
_ss("live_prices",  [])
_ss("live_refresh", 5)
_ss("watchlist",    [])
_ss("page",         "Dashboard")
_ss("portfolio",    {})
_ss("compare_b",    "")
_ss("compare_run",  False)
_ss("ticker_msg",   "")
_ss("alerts",       [])   # [{ticker, price, direction, triggered}]
_ss("chat_history", [])   # [{role, content}]
_ss("db_synced",    False)

# Sync persistent DB to session state
sync_db_to_session()

# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
def get_currency(t):
    t = t.upper()
    if t.endswith((".NS",".BO")): return "₹"
    if t.endswith(".L"):           return "£"
    if t.endswith((".PA",".DE",".AS")): return "€"
    return "$"

def smart_resolve_ticker(raw):
    """
    Smart ticker resolver:
    - User types 'NTPC'    → tries NTPC.NS, NTPC.BO, NTPC
    - User types 'TSLA'    → TSLA.NS fails, TSLA.BO fails, TSLA (US) works ✅
    - User types 'TSLA.NS' → validates, fails → tries TSLA (US) ✅
    - User types 'RELIANCE.NS' → validates, works ✅
    Returns (resolved_ticker, was_auto_fixed, message)
    """
    raw = raw.strip().upper()
    if not raw:
        return None, False, "Please enter a ticker."

    has_suffix = any(raw.endswith(s) for s in [".NS",".BO",".L",".PA",".DE",".AS"])

    if has_suffix:
        # User gave suffix — validate it first
        try:
            df_try = yf.Ticker(raw).history(period="5d")
            if df_try is not None and not df_try.empty and len(df_try) >= 2:
                return raw, False, ""  # valid, use as-is
        except:
            pass
        # Suffix given but invalid — strip suffix and try smart resolve
        base = raw.rsplit(".", 1)[0]
    else:
        base = raw

    # Try NSE → BSE → US (no suffix)
    candidates = [
        (base + ".NS", "NSE India"),
        (base + ".BO", "BSE India"),
        (base,         "US Market"),
    ]
    for ticker_try, market in candidates:
        # Skip if same as what user already tried (and failed)
        if has_suffix and ticker_try == raw:
            continue
        try:
            df_try = yf.Ticker(ticker_try).history(period="5d")
            if df_try is not None and not df_try.empty and len(df_try) >= 2:
                was_fixed = (ticker_try != raw)
                msg = f"Auto-detected as **{ticker_try}** ({market}) ✅" if was_fixed else ""
                return ticker_try, was_fixed, msg
        except:
            continue

    return raw, False, f"Could not find **{base}** on any exchange."

def parse_news_item(item):
    content = item.get("content", {})
    if content:
        title = (content.get("title") or "").strip()
        pub   = (content.get("provider",{}).get("displayName") or content.get("provider",{}).get("name") or "News Source").strip()
        link  = (content.get("canonicalUrl",{}).get("url") or content.get("clickThroughUrl",{}).get("url") or "").strip()
        raw   = content.get("pubDate") or content.get("displayTime") or ""
        dt    = "N/A"
        if raw:
            try: dt = datetime.strptime(raw[:19],"%Y-%m-%dT%H:%M:%S").strftime("%b %d, %H:%M")
            except: dt = raw[:10]
        if title: return title, pub, link, dt
    title = (item.get("title") or "").strip()
    pub   = (item.get("publisher") or "News Source").strip()
    link  = (item.get("link") or "").strip()
    pts   = item.get("providerPublishTime")
    dt    = "N/A"
    if pts:
        try: dt = datetime.fromtimestamp(int(pts)).strftime("%b %d, %H:%M")
        except: pass
    if title: return title, pub, link, dt
    return None

def fetch_live_price(ticker):
    try:
        df = yf.Ticker(ticker).history(period="1d", interval="1m")
        if df.empty: return None, None
        return round(float(df["Close"].iloc[-1]), 2), df.index[-1].to_pydatetime()
    except: return None, None

def fetch_google_news(query, stock_name):
    try:
        q       = urllib.request.quote(query)
        url     = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        req     = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            root = ET.fromstring(r.read())
        items = []
        for it in root.findall(".//item")[:6]:
            title = (it.findtext("title") or "").strip()
            link  = (it.findtext("link")  or "").strip()
            pub   = (it.findtext("source") or "Google News").strip()
            dt    = (it.findtext("pubDate") or "")[:16]
            if title: items.append({"title":title,"link":link,"publisher":pub,"providerPublishTime":dt})
        return items
    except: return []

# ══════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════
DARK = st.session_state.dark_mode
T = {
    "bg_main":    "#05070a"                if DARK else "#f0f4f8",
    "bg_sidebar": "#0b111a"                if DARK else "#ffffff",
    "bg_card":    "rgba(11,17,26,0.6)"     if DARK else "rgba(255,255,255,0.9)",
    "bg_card2":   "rgba(15,23,42,0.5)"     if DARK else "rgba(241,245,249,0.9)",
    "bg_card_h":  "rgba(11,17,26,0.8)"     if DARK else "rgba(255,255,255,1)",
    "bg_news":    "rgba(11,17,26,0.45)"    if DARK else "rgba(255,255,255,0.9)",
    "bg_news_h":  "rgba(11,17,26,0.65)"    if DARK else "rgba(241,245,249,1)",
    "border":     "rgba(255,255,255,0.06)" if DARK else "rgba(0,0,0,0.08)",
    "border_sb":  "#16202d"                if DARK else "#e2e8f0",
    "text_main":  "#e2e8f0"                if DARK else "#0f172a",
    "text_muted": "#94a3b8"                if DARK else "#64748b",
    "text_faint": "#64748b"                if DARK else "#94a3b8",
    "text_metric":"#ffffff"                if DARK else "#0f172a",
    "plot_bg":    "rgba(11,17,26,0.4)"     if DARK else "rgba(248,250,252,0.8)",
    "grid":       "rgba(255,255,255,0.04)" if DARK else "rgba(0,0,0,0.05)",
    "line_c":     "rgba(255,255,255,0.1)"  if DARK else "rgba(0,0,0,0.15)",
    "tick_c":     "#94a3b8"                if DARK else "#475569",
    "legend_bg":  "rgba(11,17,26,0.85)"    if DARK else "rgba(255,255,255,0.95)",
    "grad_ticker":"linear-gradient(135deg,#ffffff 0%,#00b8ff 100%)" if DARK else "linear-gradient(135deg,#0f172a 0%,#0369a1 100%)",
    "nav_bg":     "rgba(15,23,42,0.6)"     if DARK else "rgba(241,245,249,0.9)",
    "nav_active": "rgba(0,255,157,0.12)"   if DARK else "rgba(0,200,120,0.1)",
    "radial_bg":  "radial-gradient(circle at 15% 50%,rgba(0,255,157,0.03),transparent 25%),radial-gradient(circle at 85% 30%,rgba(191,0,255,0.03),transparent 25%)" if DARK else "none",
}

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;700&family=Bebas+Neue&family=Outfit:wght@300;400;500;700&display=swap');

html,body,[class*="css"]{{
    background-color:{T['bg_main']} !important;
    color:{T["text_main"]} !important;
    font-family:'Outfit',sans-serif !important;
}}

/* ── Keyframes ── */
@keyframes fadeIn{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes countUp{{from{{opacity:0;transform:scale(0.8)}}to{{opacity:1;transform:scale(1)}}}}
@keyframes pulseNeon{{0%{{box-shadow:0 0 4px #ff0055;opacity:1}}50%{{box-shadow:0 0 16px #ff0055;opacity:0.5}}100%{{box-shadow:0 0 4px #ff0055;opacity:1}}}}
@keyframes pulseGreen{{0%{{opacity:1}}50%{{opacity:0.4}}100%{{opacity:1}}}}
@keyframes slideIn{{from{{opacity:0;transform:translateX(-10px)}}to{{opacity:1;transform:translateX(0)}}}}

/* ── Sidebar ── */
[data-testid="stSidebar"]{{
    background-color:{T["bg_sidebar"]} !important;
    border-right:1px solid {T["border_sb"]} !important;
}}
[data-testid="stAppViewContainer"]{{
    background-image:{T["radial_bg"]} !important;
    background-color:{T['bg_main']} !important;
}}

/* ── Metric Cards ── */
[data-testid="metric-container"]{{
    background:{T['bg_card']} !important;
    border:1px solid {T['border']} !important;
    padding:20px !important;
    border-radius:16px !important;
    box-shadow:0 4px 24px rgba(0,0,0,{"0.35" if DARK else "0.06"}) !important;
    backdrop-filter:blur(16px) !important;
    transition:all 0.3s ease !important;
    animation:fadeIn 0.5s ease-out forwards;
}}
[data-testid="metric-container"]:hover{{
    transform:translateY(-4px) !important;
    border-color:rgba(0,255,157,0.35) !important;
    box-shadow:0 12px 36px rgba(0,255,157,0.12) !important;
}}
[data-testid="metric-container"] label{{
    color:{T['text_muted']} !important;font-size:10px !important;
    letter-spacing:2px !important;text-transform:uppercase !important;font-weight:700 !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{
    font-weight:700 !important;font-size:28px !important;
    color:{T['text_metric']} !important;
    animation:countUp 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards;
}}

/* ── Buttons ── */
[data-testid="stButton"]>button{{
    background:linear-gradient(135deg,#00ff9d 0%,#00b8ff 100%) !important;
    color:#000 !important;border:none !important;border-radius:10px !important;
    font-family:'Bebas Neue',sans-serif !important;font-size:20px !important;
    letter-spacing:3px !important;padding:14px 32px !important;width:100% !important;
    transition:all 0.25s ease !important;box-shadow:0 4px 15px rgba(0,255,157,0.25) !important;
}}
[data-testid="stButton"]>button:hover{{
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(0,255,157,0.45) !important;filter:brightness(1.08);
}}

/* ── Section Sub-headers ── */
.section-sub{{
    font-size:10px;color:{T['text_faint']};letter-spacing:4px;
    text-transform:uppercase;margin-bottom:10px;font-weight:700;opacity:0.9;
}}

/* ── Big ticker ── */
.big-ticker{{
    font-family:'Bebas Neue',sans-serif;font-size:80px;letter-spacing:6px;line-height:1;
    background:{T["grad_ticker"]};-webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation:fadeIn 0.8s ease-out;
}}
.current-price{{
    font-weight:700;font-size:48px;color:#00ff9d;text-align:right;
    text-shadow:0 0 24px rgba(0,255,157,0.3);animation:countUp 0.7s ease-out;
}}

/* ── Navigation Pills ── */
.nav-pill{{
    display:flex;align-items:center;gap:10px;
    padding:12px 16px;border-radius:12px;margin-bottom:6px;
    cursor:pointer;transition:all 0.2s ease;
    border:1px solid transparent;
    font-size:13px;font-weight:600;letter-spacing:0.5px;
    text-decoration:none;
}}
.nav-pill:hover{{
    background:rgba(0,255,157,0.08) !important;
    border-color:rgba(0,255,157,0.2) !important;
    transform:translateX(4px);
}}
.nav-pill.active{{
    background:{T['nav_active']} !important;
    border-color:rgba(0,255,157,0.4) !important;
    color:#00ff9d !important;
}}
.nav-icon{{font-size:16px;width:24px;text-align:center;}}

/* ── News Cards ── */
.news-card{{
    background:{T['bg_news']} !important;border:1px solid {T['border']} !important;
    padding:16px 20px !important;border-radius:14px !important;margin-bottom:12px !important;
    transition:all 0.25s ease !important;backdrop-filter:blur(8px);
    animation:slideIn 0.4s ease-out forwards;
}}
.news-card:hover{{
    border-color:rgba(0,184,255,0.35) !important;
    transform:translateX(4px) !important;
    background:{T['bg_news_h']} !important;
}}

/* ── Watchlist items ── */
.watchlist-item{{
    display:flex;justify-content:space-between;align-items:center;
    padding:10px 14px;border-radius:10px;margin-bottom:6px;
    background:{T['bg_card2']};border:1px solid {T['border']};
    cursor:pointer;transition:all 0.2s ease;
}}
.watchlist-item:hover{{border-color:rgba(0,255,157,0.3);background:{T['bg_card_h']};}}

/* ── Alert Badges ── */
.alert-badge{{
    display:inline-flex;align-items:center;gap:4px;
    padding:3px 10px;border-radius:20px;font-size:10px;
    font-weight:700;letter-spacing:1px;margin-left:8px;
}}
.badge-danger{{background:rgba(255,0,85,0.15);color:#ff0055;border:1px solid rgba(255,0,85,0.3);}}
.badge-warning{{background:rgba(227,179,65,0.15);color:#e3b341;border:1px solid rgba(227,179,65,0.3);}}
.badge-safe{{background:rgba(0,255,157,0.1);color:#00ff9d;border:1px solid rgba(0,255,157,0.25);}}

/* ── Footer ── */
.footer{{padding:40px 0 30px 0;margin-top:60px;border-top:1px solid {T['border']};color:{T['text_faint']};font-size:12px;text-align:center;}}
.footer-links a{{color:{T['text_muted']};text-decoration:none;margin:0 12px;transition:color 0.2s;}}
.footer-links a:hover{{color:#00ff9d;}}
.info-badge{{display:inline-block;background:{"rgba(0,255,157,0.07)" if DARK else "rgba(0,180,140,0.1)"};border:1px solid {"rgba(0,255,157,0.15)" if DARK else "rgba(0,180,140,0.25)"};color:{"#00ff9d" if DARK else "#059669"};padding:3px 12px;border-radius:20px;font-size:9px;letter-spacing:2px;font-weight:700;margin:0 3px;}}
.disclaimer{{font-size:11px;color:{T['text_faint']};text-align:center;padding:14px;border:1px solid rgba(255,80,80,0.1);border-radius:10px;background:rgba(255,0,85,0.02);}}
.stTooltipIcon{{color:#00b8ff !important;}}

/* ── Tooltips ── */
.tv-tooltip {{
    position:relative; display:inline-block; cursor:help;
}}
.tv-tooltip .tv-tip {{
    visibility:hidden; opacity:0; background:#1e293b; color:#cbd5e1;
    border:1px solid rgba(0,255,157,0.2); border-radius:10px;
    padding:8px 12px; font-size:11px; line-height:1.5; width:200px;
    position:absolute; z-index:999; bottom:130%; left:50%;
    transform:translateX(-50%); transition:opacity 0.2s;
    box-shadow:0 4px 20px rgba(0,0,0,0.5); pointer-events:none;
}}
.tv-tooltip:hover .tv-tip {{ visibility:visible; opacity:1; }}

/* ── Analyze button ── */
div[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button {{
    background: linear-gradient(135deg,#00ff9d,#00b8ff) !important;
    color: #05070a !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    border: none !important;
}}

/* ── Mobile Responsive ── */
@media (max-width: 768px) {{
    .big-ticker {{ font-size: 48px !important; letter-spacing: 3px !important; }}
    .current-price {{ font-size: 32px !important; }}
    [data-testid="stSidebar"] {{ min-width: 260px !important; }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{ font-size: 20px !important; }}
    .news-card {{ padding: 12px !important; }}
}}

/* ── Sidebar radio hide default ── */
[data-testid="stSidebar"] [data-testid="stRadio"]>div>div{{display:none;}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T['plot_bg'],
    font=dict(family="Outfit, sans-serif", color=T['tick_c'], size=12),
    xaxis=dict(gridcolor=T['grid'], linecolor=T['line_c'], tickfont=dict(size=10, family="Fira Code, monospace", color=T['tick_c']), showgrid=True),
    yaxis=dict(gridcolor=T['grid'], linecolor=T['line_c'], tickfont=dict(size=10, family="Fira Code, monospace", color=T['tick_c']), showgrid=True),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor=T['legend_bg'], bordercolor=T['border'], borderwidth=1, font=dict(size=11, color=T['tick_c']), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

# ══════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════
import pickle, importlib

MODELS_DIR = _BASE / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

def _model_cache_key(stock_name, model_type, window, horizon):
    """Unique key per stock+model+settings. Expires after 1 day."""
    today = datetime.now().strftime("%Y-%m-%d")
    raw   = f"{stock_name}_{model_type}_{window}_{horizon}_{today}"
    return hashlib.md5(raw.encode()).hexdigest()

def _save_result(key, result):
    try:
        # Save keras model separately
        model_path = MODELS_DIR / f"{key}.keras"
        result["model"].save(str(model_path))
        # Save rest with pickle
        result_copy = {k:v for k,v in result.items() if k != "model"}
        result_copy["model_path"] = str(model_path)
        with open(MODELS_DIR / f"{key}.pkl", "wb") as f:
            pickle.dump(result_copy, f)
    except Exception as e:
        pass  # Cache fail is non-fatal

def _load_result(key):
    try:
        pkl_path = MODELS_DIR / f"{key}.pkl"
        if not pkl_path.exists(): return None
        with open(pkl_path, "rb") as f:
            result = pickle.load(f)
        # Reload keras model
        from tensorflow.keras.models import load_model
        model_path = result.get("model_path","")
        if model_path and Path(model_path).exists():
            result["model"] = load_model(model_path)
            return result
    except:
        pass
    return None

@st.cache_resource(show_spinner=False)
def run_pipeline(stock_name, epochs=20, batch_size=32, window=60, horizon=5, model_type="SimpleRNN"):

    # ── Check disk cache first ────────────────────────────────────
    cache_key = _model_cache_key(stock_name, model_type, window, horizon)
    cached    = _load_result(cache_key)
    if cached:
        return cached, None

    # ── Full training pipeline ────────────────────────────────────
    data, info = load_data(stock_name)
    if data is None or data.empty:
        return None, "No data found for this ticker."

    close_prices, scaled_data, train_data, test_data, scaler, close_scaler, training_data_len = scale_data(
        data, window_size=window, use_features=True, ticker_symbol=stock_name)

    X_train, y_train = create_sequences(train_data, window_size=window)
    X_test,  y_test  = create_sequences(test_data,  window_size=window)

    n_features = X_train.shape[2]

    model   = build_model((window, n_features), model_type=model_type, units=64, dropout=0.2)
    history = train_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)

    train_pred, test_pred, total_actual, rmse = make_predictions(
        model, X_train, X_test, y_train, y_test, scaler, close_scaler)

    next_price = next_day_prediction(model, scaled_data, close_scaler, window_size=window)

    forecast_dates, forecast_prices = forecast_n_days(
        model, scaled_data, close_scaler, data.index[-1],
        window_size=window, forecast_days=horizon)

    result = dict(
        data=data, close_prices=close_prices, total_actual=total_actual,
        train_pred=train_pred, test_pred=test_pred, training_data_len=training_data_len,
        history=history, rmse=rmse, current_price=close_prices[-1][0],
        next_price=next_price, forecast_dates=forecast_dates, forecast_prices=forecast_prices,
        stock_name=stock_name, info=info, model_type=model_type, model=model,
    )

    # ── Save to disk for next time ────────────────────────────────
    _save_result(cache_key, result)

    return result, None

# ══════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════

def chart_candlestick(data, stock_name, cur):
    """OHLC Candlestick with Volume — like Zerodha / Groww"""
    df = data.copy().tail(120)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#00ff9d", width=1), fillcolor="rgba(0,255,157,0.7)"),
        decreasing=dict(line=dict(color="#ff0055", width=1), fillcolor="rgba(255,0,85,0.7)"),
    ), row=1, col=1)

    # 20 EMA
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", line=dict(color="#00b8ff", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(color="#bf00ff", width=1.5, dash="dash")), row=1, col=1)

    # Volume bars colored by direction
    colors = ["rgba(0,255,157,0.5)" if c >= o else "rgba(255,0,85,0.5)" for c, o in zip(df["Close"], df["Open"])]
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, marker_line_width=0), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T['plot_bg'],
        font=dict(family="Outfit, sans-serif", color=T['tick_c'], size=12),
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
        legend=dict(bgcolor=T['legend_bg'], bordercolor=T['border'], borderwidth=1,
            font=dict(size=11, color=T['tick_c']), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=f"{stock_name} — Candlestick (120 Days) + Volume", font=dict(size=14, color=T['text_muted'])),
        xaxis_rangeslider_visible=False,
        xaxis=dict(gridcolor=T['grid'], linecolor=T['line_c'], tickfont=dict(size=10, color=T['tick_c']), showgrid=True),
        yaxis=dict(title=f"Price ({cur})", gridcolor=T['grid'], linecolor=T['line_c'], tickprefix=cur, tickfont=dict(size=10, color=T['tick_c'])),
        yaxis2=dict(title="Volume", gridcolor=T['grid'], linecolor=T['line_c'], tickfont=dict(size=9, color=T['tick_c'])),
        height=500,
    )
    return fig

def chart_rsi_gauge(rsi_value):
    """RSI Speedometer Gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rsi_value,
        title=dict(text="RSI (14-Day)", font=dict(size=14, color=T['text_muted'], family="Outfit, sans-serif")),
        number=dict(font=dict(size=36, color=T['text_metric'], family="Bebas Neue"), suffix=""),
        delta=dict(reference=50, increasing=dict(color="#ff0055"), decreasing=dict(color="#00ff9d")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=T['tick_c'], tickfont=dict(size=10, color=T['tick_c'])),
            bar=dict(color="#00b8ff", thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  30], color="rgba(0,255,157,0.15)"),
                dict(range=[30, 70], color="rgba(148,163,184,0.08)"),
                dict(range=[70,100], color="rgba(255,0,85,0.15)"),
            ],
            threshold=dict(
                line=dict(color="#e3b341", width=3),
                thickness=0.75,
                value=rsi_value,
            ),
        ),
    ))
    # Labels
    for val, label, color in [(15,"OVERSOLD","#00ff9d"), (50,"NEUTRAL",T['text_faint']), (85,"OVERBOUGHT","#ff0055")]:
        fig.add_annotation(x=0.5, y=-0.15, xref="paper", yref="paper",
            text=f'<span style="color:{color};font-size:11px;letter-spacing:2px;">{label if abs(rsi_value-val)<20 else ""}</span>',
            showarrow=False, font=dict(size=11))

    zone = "🟢 OVERSOLD — Potential Buy" if rsi_value < 30 else ("🔴 OVERBOUGHT — Potential Sell" if rsi_value > 70 else "🟡 NEUTRAL ZONE")
    zcolor = "#00ff9d" if rsi_value < 30 else ("#ff0055" if rsi_value > 70 else "#e3b341")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color=T['tick_c']),
        height=260,
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[dict(
            x=0.5, y=-0.05, xref="paper", yref="paper",
            text=f"<b style='color:{zcolor};font-size:12px;'>{zone}</b>",
            showarrow=False,
        )]
    )
    return fig

def chart_sparkline(prices, color="#00ff9d"):
    """Tiny inline sparkline — no axes, just the line"""
    fill = "rgba(0,255,157,0.1)" if color == "#00ff9d" else "rgba(255,0,85,0.1)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=prices, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=fill,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0), height=50,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

def chart_historical(data, stock_name):
    data = data.copy()
    data["MA50"]      = data["Close"].rolling(50).mean()
    data["MA200"]     = data["Close"].rolling(200).mean()
    data["STD20"]     = data["Close"].rolling(20).std()
    data["UpperBand"] = data["Close"].rolling(20).mean() + data["STD20"]*2
    data["LowerBand"] = data["Close"].rolling(20).mean() - data["STD20"]*2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"].squeeze(), mode="lines", name="Close", line=dict(color="#00ff9d", width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["UpperBand"], mode="lines", showlegend=False, line=dict(color="rgba(0,184,255,0.2)", width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data["LowerBand"], mode="lines", name="Bollinger Bands", fill="tonexty", fillcolor="rgba(0,184,255,0.05)", line=dict(color="rgba(0,184,255,0.2)", width=1)))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"].squeeze(),  mode="lines", name="MA50",  line=dict(color="#00b8ff", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA200"].squeeze(), mode="lines", name="MA200", line=dict(color="#bf00ff", width=1.5, dash="dash")))
    fig.update_layout(**PL, title=dict(text=f"{stock_name} — Historical + Bollinger Bands", font=dict(size=14, color=T['text_muted'])), height=420)
    return fig

def chart_indicators(data):
    data  = data.copy()
    delta = data["Close"].diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    data["RSI"] = 100 - (100/(1+gain/loss))
    data["MACD"]   = data["Close"].ewm(span=12).mean() - data["Close"].ewm(span=26).mean()
    data["Signal"] = data["MACD"].ewm(span=9).mean()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.4,0.3,0.3],
        subplot_titles=["RSI (14)", "MACD", "Volume"])
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI", line=dict(color="#00ff9d", width=1.5)), row=1, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff0055", row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00b8ff", row=1, col=1)
    fig.add_hrect(y0=70,y1=100, fillcolor="rgba(255,0,85,0.05)", layer="below", line_width=0, row=1, col=1)
    fig.add_hrect(y0=0, y1=30,  fillcolor="rgba(0,255,157,0.05)",layer="below", line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"],   name="MACD",   line=dict(color="#00b8ff",width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["Signal"], name="Signal", line=dict(color="#e3b341",width=1.5,dash="dot")), row=2, col=1)
    macd_hist = data["MACD"] - data["Signal"]
    fig.add_trace(go.Bar(x=data.index, y=macd_hist, name="Histogram",
        marker_color=["rgba(0,255,157,0.5)" if v>=0 else "rgba(255,0,85,0.5)" for v in macd_hist]), row=2, col=1)
    if "Volume" in data.columns:
        fig.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume", marker_color="rgba(0,184,255,0.35)"), row=3, col=1)
    fig.update_layout(**PL, height=550, showlegend=True,
        title=dict(text="Technical Indicators — RSI · MACD · Volume", font=dict(size=14, color=T['text_muted'])))
    fig.update_annotations(font=dict(color=T['text_muted'], size=11))
    return fig

def chart_train_test(total_actual, train_pred, test_pred, training_data_len):
    n  = len(total_actual)
    tx = list(range(60, 60+len(train_pred)))
    ex = list(range(60+len(train_pred), 60+len(train_pred)+len(test_pred)))
    sx = training_data_len - 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(n)), y=total_actual.flatten(), mode="lines", name="Actual", line=dict(color="#4a5568", width=1.5)))
    fig.add_trace(go.Scatter(x=tx, y=train_pred.flatten(), mode="lines", name="Train Pred", line=dict(color="#00b8ff", width=2)))
    fig.add_trace(go.Scatter(x=ex, y=test_pred.flatten(),  mode="lines", name="Test Pred",  line=dict(color="#00ff9d", width=2.5)))
    fig.add_vline(x=sx, line_dash="dash", line_color="#e3b341", line_width=1.5,
        annotation_text="TRAIN / TEST SPLIT", annotation_font=dict(color="#e3b341", size=10, family="Fira Code, monospace"), annotation_position="top right")
    fig.add_vrect(x0=0, x1=sx, fillcolor="rgba(88,166,255,0.03)", layer="below", line_width=0)
    fig.add_vrect(x0=sx, x1=n, fillcolor="rgba(46,160,67,0.03)",  layer="below", line_width=0)
    fig.update_layout(**PL, title=dict(text="Actual vs Predicted", font=dict(size=14, color=T['text_muted'])), height=400)
    return fig

def chart_residuals(total_actual, train_pred, test_pred, _):
    n_tr  = len(train_pred.flatten()); n_te = len(test_pred.flatten())
    ta    = total_actual.flatten()[-n_te:]
    tra   = total_actual.flatten()[-(n_tr+n_te):-n_te]
    tr_r  = tra - train_pred.flatten(); te_r = ta - test_pred.flatten()
    fig   = make_subplots(rows=2, cols=1, subplot_titles=["Train Residuals","Test Residuals"], vertical_spacing=0.14)
    fig.add_trace(go.Bar(x=list(range(len(tr_r))), y=tr_r, marker_color=["#00b8ff" if v>=0 else "#ff0055" for v in tr_r], marker_line_width=0), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(len(te_r))), y=te_r, marker_color=["#00ff9d" if v>=0 else "#ff0055" for v in te_r], marker_line_width=0), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#3a4a5a", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#3a4a5a", row=2, col=1)
    fig.update_layout(**PL, height=400, showlegend=False, title=dict(text="Residual Analysis", font=dict(size=14, color=T['text_muted'])))
    fig.update_annotations(font=dict(color=T['text_muted'], size=11))
    return fig

def chart_loss_curve(history):
    ep  = list(range(1, len(history.history["loss"])+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=history.history["loss"], mode="lines+markers", name="Train Loss", line=dict(color="#00b8ff",width=2), marker=dict(size=4), fill="tozeroy", fillcolor="rgba(0,184,255,0.06)"))
    val = history.history.get("val_loss",[])
    if val: fig.add_trace(go.Scatter(x=ep, y=val, mode="lines+markers", name="Val Loss", line=dict(color="#00ff9d",width=2), marker=dict(size=4)))
    fig.update_layout(**PL, title=dict(text="Training Loss Curve", font=dict(size=14, color=T['text_muted'])), height=300)
    return fig

def chart_scatter(total_actual, train_pred, test_pred):
    n_tr = len(train_pred.flatten()); n_te = len(test_pred.flatten())
    ta   = total_actual.flatten()[-n_te:]; tra = total_actual.flatten()[-(n_tr+n_te):-n_te]
    mn, mx = np.concatenate([tra,ta]).min(), np.concatenate([tra,ta]).max()
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=tra, y=train_pred.flatten(), mode="markers", name="Train", marker=dict(color="#00b8ff",size=4,opacity=0.6)))
    fig.add_trace(go.Scatter(x=ta,  y=test_pred.flatten(),  mode="markers", name="Test",  marker=dict(color="#00ff9d",size=5,opacity=0.7)))
    fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines", name="Perfect Fit", line=dict(color="#e3b341",dash="dash",width=1.5)))
    fig.update_layout(**PL, title=dict(text="Actual vs Predicted (Scatter)", font=dict(size=14, color=T['text_muted'])), height=360)
    return fig

def chart_forecast(data, forecast_dates, forecast_prices, current_price, stock_name, horizon, cur):
    recent = data["Close"].squeeze()[-100:]
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines", name="Recent", line=dict(color="#4a5568",width=1.5)))
    fig.add_trace(go.Scatter(x=[recent.index[-1],forecast_dates[0]], y=[current_price,forecast_prices[0][0]], mode="lines", showlegend=False, line=dict(color="#00ff9d",width=1.5,dash="dot")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=[p[0] for p in forecast_prices], mode="lines+markers", name=f"{horizon}-Day Forecast",
        line=dict(color="#00ff9d",width=2.5,dash="dash"), marker=dict(size=8,color="#00ff9d",symbol="circle",line=dict(color="#05070a",width=2)),
        fill="tonexty", fillcolor="rgba(0,255,157,0.07)"))
    fig.add_trace(go.Scatter(x=[recent.index[-1],recent.index[-1]], y=[recent.min()*0.95, max([p[0] for p in forecast_prices])*1.05],
        mode="lines", showlegend=False, line=dict(color="#e3b341",width=1,dash="dash")))
    fig.add_annotation(x=recent.index[-1], y=max([p[0] for p in forecast_prices])*1.05, text="TODAY", showarrow=False,
        font=dict(color="#e3b341",size=9,family="Fira Code, monospace"), yshift=10)
    fig.update_layout(**PL, title=dict(text=f"{stock_name} — {horizon}-Day Forecast", font=dict(size=14, color=T['text_muted'])),
        height=380)
    return fig

def compute_signals(data):
    df = data.copy()
    df["EMA9"]  = df["Close"].ewm(span=9).mean()
    df["EMA21"] = df["Close"].ewm(span=21).mean()
    delta = df["Close"].diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    df["RSI"] = 100 - (100/(1+gain/loss))
    df["MACD"]   = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["Signal_MACD"] = df["MACD"].ewm(span=9).mean()
    df["buy_signal"]  = ((df["EMA9"] > df["EMA21"]) & (df["EMA9"].shift(1) <= df["EMA21"].shift(1)) & (df["RSI"] < 65))
    df["sell_signal"] = ((df["EMA9"] < df["EMA21"]) & (df["EMA9"].shift(1) >= df["EMA21"].shift(1)) & (df["RSI"] > 35))
    return df

def chart_signals(data, stock_name, cur):
    df  = compute_signals(data).tail(180)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.7, 0.3], subplot_titles=["Price + EMA Signals", "RSI (14)"])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(), mode="lines",
        name="Price", line=dict(color="#4a5568", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA9"],  name="EMA 9",
        line=dict(color="#00b8ff", width=1.5, dash="dot")),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA21"], name="EMA 21",
        line=dict(color="#bf00ff", width=1.5, dash="dash")), row=1, col=1)
    buys  = df[df["buy_signal"]]
    sells = df[df["sell_signal"]]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"].squeeze(), mode="markers",
        name="BUY", marker=dict(symbol="triangle-up", size=14, color="#00ff9d",
        line=dict(color="#05070a", width=1.5))), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"].squeeze(), mode="markers",
        name="SELL", marker=dict(symbol="triangle-down", size=14, color="#ff0055",
        line=dict(color="#05070a", width=1.5))), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#e3b341", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff0055", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00ff9d", row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,0,85,0.05)",  layer="below", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,157,0.05)", layer="below", line_width=0, row=2, col=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T['plot_bg'],
        font=dict(family="Outfit, sans-serif", color=T['tick_c'], size=12),
        title=dict(text=f"{stock_name} - Buy/Sell Signals (EMA Crossover + RSI)", font=dict(size=14, color=T['text_muted'])),
        xaxis2=dict(gridcolor=T['grid'], tickfont=dict(size=10, color=T['tick_c'])),
        yaxis=dict(gridcolor=T['grid'], tickprefix=cur, tickfont=dict(size=10, color=T['tick_c'])),
        yaxis2=dict(gridcolor=T['grid'], tickfont=dict(size=10, color=T['tick_c']), range=[0,100]),
        margin=dict(l=50,r=20,t=50,b=40), hovermode="x unified", height=520,
        legend=dict(bgcolor=T['legend_bg'], bordercolor=T['border'], borderwidth=1,
            font=dict(size=11, color=T['tick_c']), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_annotations(font=dict(color=T['text_muted'], size=11))
    return fig, df

def chart_comparison(data_a, data_b, name_a, name_b):
    close_a = data_a["Close"].squeeze()
    close_b = data_b["Close"].squeeze()
    common  = close_a.index.intersection(close_b.index)
    if len(common) < 10: return None
    a = close_a.loc[common]; b = close_b.loc[common]
    a_norm = (a / a.iloc[0]) * 100
    b_norm = (b / b.iloc[0]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a_norm.index, y=a_norm.values, mode="lines",
        name=name_a, line=dict(color="#00ff9d", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,255,157,0.04)"))
    fig.add_trace(go.Scatter(x=b_norm.index, y=b_norm.values, mode="lines",
        name=name_b, line=dict(color="#00b8ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,184,255,0.04)"))
    fig.add_hline(y=100, line_dash="dot", line_color="#e3b341", line_width=1,
        annotation_text="BASELINE", annotation_font=dict(color="#e3b341", size=9))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T['plot_bg'],
        font=dict(family="Outfit, sans-serif", color=T['tick_c'], size=12),
        title=dict(text=f"Performance: {name_a} vs {name_b} (Normalized to 100)", font=dict(size=14, color=T['text_muted'])),
        xaxis=dict(gridcolor=T['grid'], tickfont=dict(size=10, color=T['tick_c'])),
        yaxis=dict(gridcolor=T['grid'], ticksuffix="", tickfont=dict(size=10, color=T['tick_c'])),
        margin=dict(l=50,r=20,t=50,b=40), hovermode="x unified", height=400,
        legend=dict(bgcolor=T['legend_bg'], bordercolor=T['border'], borderwidth=1,
            font=dict(size=12, color=T['tick_c']), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_live_feed(live_prices, cur, ticker):
    if len(live_prices) < 2: return None
    times  = [p[0] for p in live_prices]; prices = [p[1] for p in live_prices]; first = prices[0]
    lc     = "#00ff9d" if prices[-1] >= first else "#ff0055"
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=prices, mode="lines+markers", line=dict(color=lc,width=2.5),
        marker=dict(size=4, color=["#00ff9d" if p>=first else "#ff0055" for p in prices]),
        fill="tozeroy", fillcolor="rgba(0,255,157,0.04)" if prices[-1]>=first else "rgba(255,0,85,0.04)",
        hovertemplate=f"{cur}%{{y:,.2f}}<br>%{{x|%H:%M:%S}}<extra></extra>"))
    fig.add_hline(y=first, line_dash="dot", line_color="#e3b341", line_width=1,
        annotation_text="SESSION OPEN", annotation_font=dict(color="#e3b341",size=9,family="Fira Code, monospace"), annotation_position="bottom right")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T['plot_bg'],
        font=dict(family="Outfit, sans-serif", color=T['tick_c'], size=12),
        title=dict(text=f"{ticker} — Live Price Feed", font=dict(size=14, color=T['text_muted'])),
        xaxis=dict(title="Time", gridcolor=T['grid'], tickfont=dict(size=10, color=T['tick_c'])),
        yaxis=dict(title=f"Price ({cur})", tickprefix=cur, gridcolor=T['grid'], tickfont=dict(size=10, color=T['tick_c'])),
        margin=dict(l=50,r=20,t=50,b=40), hovermode="x unified", height=340, showlegend=False,
    )
    return fig

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Theme Toggle ──────────────────────────────────────────────
    tc1, tc2 = st.columns([4,1])
    with tc1:
        st.markdown(f"<div style='font-size:10px;color:{T['text_faint']};letter-spacing:2px;font-weight:700;padding-top:8px;'>{'🌙 DARK MODE' if DARK else '☀️ LIGHT MODE'}</div>", unsafe_allow_html=True)
    with tc2:
        if st.button("⇄", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    st.markdown("<hr style='margin:8px 0;opacity:0.1;'>", unsafe_allow_html=True)

    # ── Logo ─────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex;align-items:center;margin-bottom:16px;margin-top:8px;'>
        <svg width="36" height="36" viewBox="0 0 100 100" fill="none">
            <path d="M20 80L40 40L60 60L80 20" stroke="url(#g1)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="20" cy="80" r="5" fill="#00ff9d"/><circle cx="40" cy="40" r="5" fill="#00b8ff"/>
            <circle cx="60" cy="60" r="5" fill="#00ff9d"/><circle cx="80" cy="20" r="5" fill="#00b8ff"/>
            <defs><linearGradient id="g1" x1="20" y1="80" x2="80" y2="20" gradientUnits="userSpaceOnUse">
                <stop stop-color="#00ff9d"/><stop offset="1" stop-color="#00b8ff"/></linearGradient></defs>
        </svg>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:28px;letter-spacing:4px;
            background:linear-gradient(135deg,#00ff9d,#00b8ff);-webkit-background-clip:text;
            -webkit-text-fill-color:transparent;margin-left:10px;'>TradeVision AI</div>
    </div>
    <div style='font-size:9px;color:#64748b;letter-spacing:3px;margin-bottom:24px;'>// RECURRENT NEURAL ENGINE</div>
    """, unsafe_allow_html=True)

    # ── Ticker Input ─────────────────────────────────────────────
    st.markdown("<div class='section-sub'>// STOCK INPUT</div>", unsafe_allow_html=True)
    ticker_input = st.text_input("Ticker", value=st.session_state.ticker,
        placeholder="e.g. RELIANCE.NS / AAPL", label_visibility="collapsed")

    # Quick example chips
    st.markdown("""<div style='margin:6px 0 10px;'>
        <span style='font-size:9px;color:#3a4a5a;letter-spacing:2px;'>TRY: </span>
        <span style='font-size:9px;color:#e3b341;background:rgba(227,179,65,0.08);
            border:1px solid rgba(227,179,65,0.2);padding:2px 8px;border-radius:6px;margin:2px;'>RELIANCE.NS</span>
        <span style='font-size:9px;color:#e3b341;background:rgba(227,179,65,0.08);
            border:1px solid rgba(227,179,65,0.2);padding:2px 8px;border-radius:6px;margin:2px;'>TCS.NS</span>
        <span style='font-size:9px;color:#e3b341;background:rgba(227,179,65,0.08);
            border:1px solid rgba(227,179,65,0.2);padding:2px 8px;border-radius:6px;margin:2px;'>AAPL</span>
    </div>""", unsafe_allow_html=True)

    if st.button("⚡ ANALYZE →", use_container_width=True) and ticker_input.strip():
        raw_t = ticker_input.strip().upper()
        with st.spinner("🔍 Looking up ticker..."):
            resolved, auto_fixed, msg = smart_resolve_ticker(raw_t)

        if auto_fixed:
            st.session_state.ticker_msg = msg
        elif "Could not find" in msg:
            st.error(f"😕 {msg}\n\n💡 Try: `RELIANCE.NS`, `TCS.NS`, `AAPL`, `TSLA`")
            st.stop()

        t = resolved
        st.session_state.ticker      = t
        st.session_state.analyzed    = True
        st.session_state.live_prices = []
        st.session_state.page        = "Dashboard"
        if t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
            db_add_watchlist(t)
        st.rerun()

    # Show auto-fix message if ticker was resolved
    if st.session_state.get("ticker_msg"):
        st.success(st.session_state.ticker_msg)
        st.session_state.ticker_msg = ""
    st.markdown(f"<div style='margin-top:6px;font-size:9px;color:#3a4a5a;line-height:2;'>NSE → <span style='color:#e3b341;'>.NS</span> &nbsp;|&nbsp; BSE → <span style='color:#e3b341;'>.BO</span> &nbsp;|&nbsp; US → no suffix</div>", unsafe_allow_html=True)

    # ── Watchlist ─────────────────────────────────────────────────
    if st.session_state.watchlist:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>// WATCHLIST</div>", unsafe_allow_html=True)
        for wt in st.session_state.watchlist:
            wcur = get_currency(wt)
            is_active = wt == st.session_state.ticker
            bg = "rgba(0,255,157,0.1)" if is_active else T['bg_card2']
            bdr = "rgba(0,255,157,0.4)" if is_active else T['border']
            col_w1, col_w2 = st.columns([3,1])
            with col_w1:
                if st.button(f"{'▶ ' if is_active else ''}{wt}", key=f"wl_{wt}", use_container_width=True):
                    st.session_state.ticker      = wt
                    st.session_state.analyzed    = True
                    st.session_state.live_prices = []
                    st.session_state.page        = "Dashboard"
                    st.rerun()
            with col_w2:
                if st.button("✕", key=f"rm_{wt}"):
                    st.session_state.watchlist.remove(wt)
                    db_remove_watchlist(wt)
                    if st.session_state.ticker == wt:
                        st.session_state.analyzed = False
                        st.session_state.ticker   = ""
                    st.rerun()

    # ── Navigation ───────────────────────────────────────────────
    if st.session_state.analyzed:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>// NAVIGATION</div>", unsafe_allow_html=True)

        NAV_ITEMS = [
            ("📊", "Dashboard",           "Dashboard"),
            ("🕯️", "Candlestick",         "Candlestick"),
            ("🔮", f"{st.session_state.horizon_val}-Day Forecast", f"{st.session_state.horizon_val}-Day Forecast"),
            ("📈", "Technical",           "Technical"),
            ("🟢", "Buy/Sell Signals",    "Buy/Sell Signals"),
            ("⚖️", "Compare Stocks",      "Compare Stocks"),
            ("💼", "Portfolio",           "Portfolio"),
            ("🧠", "Model Performance",   "Model Performance"),
            ("📡", "Live Feed",           "🔴 Live Feed"),
            ("📰", "Sentiment",           "Sentiment"),
            ("⏱️", "Backtesting",         "Backtesting"),
            ("📄", "PDF Report",          "PDF Report"),
            ("🔍", "Stock Screener",      "Stock Screener"),
            ("ℹ️",  "About",              "About"),
        ]

        # Keyboard shortcuts
        st.markdown("""<script>
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const map = {'d':'Dashboard','c':'Candlestick','t':'Technical',
                         'f':'Forecast','p':'Portfolio','l':'Live Feed','s':'Sentiment'};
            if (map[e.key.toLowerCase()]) {
                const btns = parent.document.querySelectorAll('button');
                btns.forEach(b => { if(b.innerText.includes(map[e.key.toLowerCase()])) b.click(); });
            }
        });
        </script>""", unsafe_allow_html=True)

        # CSS for pill buttons — no dots, full glow
        st.markdown(f"""<style>
        div[data-testid="stSidebar"] div[data-testid="stButton"] button {{
            width: 100% !important;
            text-align: left !important;
            padding: 9px 14px !important;
            border-radius: 10px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
            border: 1px solid {T['border']} !important;
            background: {T['bg_card2']} !important;
            color: {T['text_muted']} !important;
            transition: all 0.18s ease !important;
            margin-bottom: 3px !important;
            box-shadow: none !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
            background: rgba(0,255,157,0.08) !important;
            border-color: rgba(0,255,157,0.35) !important;
            color: #ffffff !important;
            box-shadow: 0 0 12px rgba(0,255,157,0.15) !important;
            transform: translateX(3px) !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stButton"] button.nav-active,
        div[data-testid="stSidebar"] div[data-testid="stButton"] button:focus {{
            background: rgba(0,255,157,0.12) !important;
            border-color: rgba(0,255,157,0.5) !important;
            color: #00ff9d !important;
            box-shadow: 0 0 16px rgba(0,255,157,0.2) !important;
        }}
        </style>""", unsafe_allow_html=True)

        cur_page = st.session_state.page
        for icon, label, key in NAV_ITEMS:
            is_active = cur_page == key
            # Active pill: green bg + glow via markdown overlay
            if is_active:
                st.markdown(f"""<div style='background:rgba(0,255,157,0.1);
                    border:1px solid rgba(0,255,157,0.45);border-radius:10px;
                    padding:9px 14px;margin-bottom:3px;
                    box-shadow:0 0 14px rgba(0,255,157,0.18);'>
                    <span style='font-size:13px;'>{icon}</span>
                    <span style='font-size:12px;font-weight:700;color:#00ff9d;
                        margin-left:8px;letter-spacing:0.3px;'>{label}</span>
                    <span style='float:right;width:6px;height:6px;border-radius:50%;
                        background:#00ff9d;display:inline-block;margin-top:4px;
                        box-shadow:0 0 6px #00ff9d;'></span>
                </div>""", unsafe_allow_html=True)
            else:
                if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                    st.session_state.page = key
                    st.rerun()
    else:
        pass  # no nav when not analyzed

    st.markdown("<hr style='margin:16px 0;opacity:0.1;'>", unsafe_allow_html=True)

    # ── Model Selector ───────────────────────────────────────────
    st.markdown("<div class='section-sub'>// MODEL ARCHITECTURE</div>", unsafe_allow_html=True)
    MODEL_INFO = {
        "SimpleRNN": {"icon":"⚡", "color":"#00b8ff", "tag":"BASELINE"},
        "LSTM":      {"icon":"🧠", "color":"#00ff9d", "tag":"RECOMMENDED"},
        "GRU":       {"icon":"🚀", "color":"#bf00ff", "tag":"EFFICIENT"},
    }
    for mtype, mi in MODEL_INFO.items():
        is_sel = st.session_state.model_type == mtype
        bg  = f"rgba(0,255,157,0.1)"  if is_sel else T['bg_card2']
        bdr = mi["color"]             if is_sel else T['border']
        clr = mi["color"]             if is_sel else T['text_muted']
        st.markdown(f"""
        <div style='background:{bg};border:1px solid {bdr};border-radius:12px;
            padding:10px 14px;margin-bottom:2px;{"box-shadow:0 0 10px "+mi["color"]+"22;" if is_sel else ""}'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-size:13px;font-weight:700;color:{clr};'>{mi["icon"]}  {mtype}</span>
                <span style='font-size:8px;color:{mi["color"]};letter-spacing:2px;font-weight:700;
                    background:{mi["color"]}18;border:1px solid {mi["color"]}44;
                    padding:2px 7px;border-radius:8px;'>{mi["tag"]}</span>
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"{'✓ ACTIVE' if is_sel else f'SELECT {mtype}'}", key=f"btn_{mtype}", use_container_width=True):
            st.session_state.model_type = mtype
            st.rerun()

    st.markdown("<hr style='margin:12px 0;opacity:0.1;'>", unsafe_allow_html=True)

    # ── Hyperparameters ──────────────────────────────────────────
    st.markdown("<div class='section-sub'>// HYPERPARAMETERS</div>", unsafe_allow_html=True)
    epochs_input = st.slider("Epochs", 5, 100, 20, 5)
    batch_input  = st.selectbox("Batch Size", [16,32,64,128], index=1)
    window_input = st.slider("Lookback Window (Days)", 30, 120, 60, 10)

    st.markdown("<div class='section-sub' style='margin-top:10px;'>// FORECAST HORIZON</div>", unsafe_allow_html=True)
    hq = st.select_slider("Horizon", options=[5,7,14,30], value=st.session_state.horizon_val,
        format_func=lambda x: f"{x} Days", label_visibility="collapsed")
    if hq != st.session_state.horizon_val:
        st.session_state.horizon_val = hq
        st.rerun()
    horizon_input = hq

# ══════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════
if not st.session_state.analyzed:

    # ── CSS + Animations ─────────────────────────────────────────
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem !important; }

    /* Animated SVG chart bg */
    .landing-bg {
        position:fixed;top:0;left:0;width:100%;height:100%;
        z-index:0;pointer-events:none;overflow:hidden;opacity:0.07;
    }
    .chart-line {
        stroke:#00ff9d;stroke-width:2;fill:none;
        stroke-dasharray:2000;stroke-dashoffset:2000;
        animation:drawLine 4s ease forwards;
    }
    .chart-line2 {
        stroke:#00b8ff;stroke-width:1.5;fill:none;
        stroke-dasharray:2000;stroke-dashoffset:2000;
        animation:drawLine 5s ease 0.5s forwards;
    }
    @keyframes drawLine { to { stroke-dashoffset:0; } }

    /* Glass hero */
    .glass-hero {
        background:rgba(11,17,26,0.55);
        backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
        border:1px solid rgba(0,255,157,0.15);border-radius:28px;
        padding:52px 48px;text-align:center;position:relative;z-index:1;
        box-shadow:0 8px 64px rgba(0,255,157,0.07),0 2px 16px rgba(0,0,0,0.4),
            inset 0 1px 0 rgba(255,255,255,0.06);
        margin-bottom:24px;
        animation:heroFadeIn 0.8s ease-out forwards;
    }
    @keyframes heroFadeIn {
        from{opacity:0;transform:translateY(24px);}
        to{opacity:1;transform:translateY(0);}
    }
    .hero-title {
        font-family:"Bebas Neue",Impact,sans-serif;
        font-size:88px;letter-spacing:14px;line-height:1;
        background:linear-gradient(135deg,#00ff9d 0%,#00b8ff 40%,#bf00ff 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin-bottom:10px;
        animation:titlePulse 3s ease-in-out infinite alternate;
    }
    @keyframes titlePulse {
        from{filter:brightness(1);}
        to{filter:brightness(1.15) drop-shadow(0 0 20px rgba(0,255,157,0.3));}
    }
    .hero-sub {
        font-size:11px;color:#475569;letter-spacing:5px;
        margin-bottom:32px;font-family:"Fira Code",monospace;
    }
    .glass-card {
        background:rgba(15,23,42,0.5);backdrop-filter:blur(16px);
        -webkit-backdrop-filter:blur(16px);
        border:1px solid rgba(255,255,255,0.08);border-radius:18px;
        padding:20px 16px;text-align:center;
        transition:all 0.3s ease;
        animation:cardSlideUp 0.6s ease-out forwards;opacity:0;
    }
    .glass-card:hover {
        border-color:rgba(0,255,157,0.3);
        box-shadow:0 0 24px rgba(0,255,157,0.1);
        transform:translateY(-4px);
    }
    @keyframes cardSlideUp {
        from{opacity:0;transform:translateY(20px);}
        to{opacity:1;transform:translateY(0);}
    }
    .idx-card {
        background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,0.06);border-radius:14px;
        padding:14px 10px;text-align:center;transition:all 0.25s ease;
    }
    .idx-card:hover{transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,0.3);}
    .trend-pill {
        display:inline-block;background:rgba(0,255,157,0.07);
        border:1px solid rgba(0,255,157,0.2);color:#00ff9d;
        padding:6px 18px;border-radius:24px;font-size:11px;font-weight:700;
        margin:4px;letter-spacing:1px;cursor:pointer;transition:all 0.2s ease;
    }
    .trend-pill:hover{background:rgba(0,255,157,0.15);transform:scale(1.05);}
    @keyframes tickerScroll{0%{transform:translateX(0);}100%{transform:translateX(-33.33%);}}
    </style>

    <!-- Animated chart background -->
    <div class="landing-bg">
      <svg viewBox="0 0 1400 600" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <polyline class="chart-line" points="0,450 60,420 120,440 180,380 240,360 300,390
          360,320 420,280 480,310 540,260 600,240 660,270 720,210 780,190 840,220
          900,170 960,150 1020,180 1080,130 1140,110 1200,140 1260,90 1320,70 1400,100"/>
        <polyline class="chart-line2" points="0,500 80,480 160,490 240,460 320,470
          400,440 480,450 560,410 640,420 720,380 800,390 880,350 960,360
          1040,320 1120,330 1200,290 1280,300 1400,260"/>
        <line x1="0" y1="150" x2="1400" y2="150" stroke="#00ff9d" stroke-width="0.5" opacity="0.3"/>
        <line x1="0" y1="300" x2="1400" y2="300" stroke="#00ff9d" stroke-width="0.5" opacity="0.2"/>
        <line x1="0" y1="450" x2="1400" y2="450" stroke="#00ff9d" stroke-width="0.5" opacity="0.1"/>
      </svg>
    </div>
    """, unsafe_allow_html=True)

    # ── Live Ticker Bar ───────────────────────────────────────────
    MARKET_TICKERS = [
        ("NIFTY 50","^NSEI"),("SENSEX","^BSESN"),("BANKNIFTY","^NSEBANK"),
        ("NIFTY IT","^CNXIT"),("GOLD","GC=F"),("CRUDE OIL","CL=F"),
        ("USD/INR","USDINR=X"),("S&P 500","^GSPC"),("NASDAQ","^IXIC"),
    ]
    ticker_parts = []
    for label, sym in MARKET_TICKERS:
        try:
            df_t = yf.Ticker(sym).history(period="2d")
            if not df_t.empty and len(df_t)>=2:
                cp=float(df_t["Close"].iloc[-1]); pp=float(df_t["Close"].iloc[-2])
                ch=cp-pp; chp=ch/pp*100
                col2="#00ff9d" if ch>=0 else "#ff0055"
                arr="▲" if ch>=0 else "▼"
                ticker_parts.append(
                    f"<span style='color:#64748b;margin-right:5px;font-size:10px;'>{label}</span>"
                    f"<span style='color:#f1f5f9;font-weight:700;font-size:11px;margin-right:4px;'>{cp:,.2f}</span>"
                    f"<span style='color:{col2};font-size:10px;margin-right:28px;'>{arr} {abs(chp):.2f}%</span>"
                )
        except: pass

    if ticker_parts:
        ticker_html = "".join(ticker_parts * 3)
        st.markdown(f"""
        <div style='overflow:hidden;background:rgba(11,17,26,0.8);backdrop-filter:blur(12px);
            border:1px solid rgba(255,255,255,0.06);border-radius:12px;
            padding:10px 0;margin-bottom:20px;'>
            <div style='display:inline-block;white-space:nowrap;
                animation:tickerScroll 50s linear infinite;'>{ticker_html}</div>
        </div>""", unsafe_allow_html=True)

    # ── Glass Hero Section ────────────────────────────────────────
    st.markdown("""
    <div class="glass-hero">
        <div style='font-size:11px;color:#00ff9d;letter-spacing:6px;font-weight:700;
            margin-bottom:18px;font-family:"Fira Code",monospace;'>
            ◈ AI-POWERED &nbsp;·&nbsp; DEEP LEARNING &nbsp;·&nbsp; REAL-TIME
        </div>
        <div class="hero-title">TRADEVISION AI</div>
        <div class="hero-sub">
            RECURRENT NEURAL NETWORK &nbsp;·&nbsp; DEEP LEARNING &nbsp;·&nbsp; REAL-TIME ANALYSIS
        </div>
        <div style='font-size:15px;color:#94a3b8;line-height:1.9;'>
            India's most advanced stock analysis platform.<br>
            Enter any ticker in the sidebar and click
            <span style='color:#00ff9d;font-weight:700;letter-spacing:1px;
                text-shadow:0 0 12px rgba(0,255,157,0.5);'>⚡ ANALYZE →</span>
        </div>
        <div style='position:absolute;top:-40px;left:-40px;width:200px;height:200px;
            background:radial-gradient(circle,rgba(0,255,157,0.08),transparent 70%);
            border-radius:50%;pointer-events:none;'></div>
        <div style='position:absolute;bottom:-40px;right:-40px;width:200px;height:200px;
            background:radial-gradient(circle,rgba(191,0,255,0.08),transparent 70%);
            border-radius:50%;pointer-events:none;'></div>
    </div>""", unsafe_allow_html=True)

    # ── Trending Stocks ───────────────────────────────────────────
    TRENDING = ["RELIANCE","TCS","HDFCBANK","INFY","WIPRO","SBIN","TATAMOTORS","BAJFINANCE"]
    pills = "".join([f"<span class='trend-pill'>{t}</span>" for t in TRENDING])
    st.markdown(f"""
    <div style='text-align:center;margin-bottom:28px;'>
        <span style='font-size:10px;color:#475569;letter-spacing:3px;font-weight:700;
            margin-right:10px;font-family:"Fira Code",monospace;'>🔥 TRENDING</span>
        {pills}
    </div>""", unsafe_allow_html=True)

    # ── Market Pulse Header ───────────────────────────────────────
    st.markdown("""<div style='text-align:center;margin-bottom:18px;'>
        <span style='font-size:10px;color:#475569;letter-spacing:4px;font-weight:700;
            font-family:"Fira Code",monospace;'>// TODAY\'S MARKET PULSE</span>
    </div>""", unsafe_allow_html=True)

    # ── Fetch Market Data ─────────────────────────────────────────
    with st.spinner(""):
        INDICES = [
            ("NIFTY 50","^NSEI","₹"),("SENSEX","^BSESN","₹"),
            ("BANKNIFTY","^NSEBANK","₹"),("NIFTY IT","^CNXIT","₹"),
            ("S&P 500","^GSPC","$"),("NASDAQ","^IXIC","$"),
        ]
        idx_data = []
        for name, sym, cur_i in INDICES:
            try:
                df_i = yf.Ticker(sym).history(period="5d")
                if not df_i.empty and len(df_i)>=2:
                    cp=float(df_i["Close"].iloc[-1]); pp=float(df_i["Close"].iloc[-2])
                    ch=cp-pp; chp=ch/pp*100
                    spark=df_i["Close"].tolist()[-5:]
                    idx_data.append((name,cp,ch,chp,spark,cur_i))
            except: pass

        NIFTY50_SAMPLE = [
            "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
            "HINDUNILVR.NS","ITC.NS","SBIN.NS","BAJFINANCE.NS","KOTAKBANK.NS",
            "LT.NS","AXISBANK.NS","WIPRO.NS","MARUTI.NS","ASIANPAINT.NS",
            "SUNPHARMA.NS","TATAMOTORS.NS","TITAN.NS","NESTLEIND.NS","ULTRACEMCO.NS"
        ]
        advances=declines=unchanged=0
        top_gainers=[]; top_losers=[]
        for sym in NIFTY50_SAMPLE:
            try:
                df_b=yf.Ticker(sym).history(period="2d")
                if not df_b.empty and len(df_b)>=2:
                    cp2=float(df_b["Close"].iloc[-1]); pp2=float(df_b["Close"].iloc[-2])
                    chp2=(cp2-pp2)/pp2*100
                    if chp2>0.1: advances+=1
                    elif chp2<-0.1: declines+=1
                    else: unchanged+=1
                    n2=sym.replace(".NS","")
                    top_gainers.append((n2,chp2,cp2))
                    top_losers.append((n2,chp2,cp2))
            except: pass
        top_gainers=sorted(top_gainers,key=lambda x:x[1],reverse=True)[:5]
        top_losers=sorted(top_losers,key=lambda x:x[1])[:5]

    # ── Index Cards ───────────────────────────────────────────────
    if idx_data:
        cols_idx = st.columns(len(idx_data))
        for col_i,(name,cp,ch,chp,spark,cur_i) in zip(cols_idx,idx_data):
            color="#00ff9d" if ch>=0 else "#ff0055"
            arrow="▲" if ch>=0 else "▼"
            bg_bdr="rgba(0,255,157,0.2)" if ch>=0 else "rgba(255,0,85,0.2)"
            svg=""
            if len(spark)>1:
                mn=min(spark); mx=max(spark); rng=mx-mn or 1
                pts=" ".join([f"{int(i*(44/(len(spark)-1)))},{int(28-(v-mn)/rng*24)}" for i,v in enumerate(spark)])
                svg=f"<svg width='44' height='28' style='display:block;margin:6px auto 0;'><polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2'/></svg>"
            col_i.markdown(f"""
            <div class='idx-card' style='border-color:{bg_bdr};margin-bottom:12px;'>
                <div style='font-size:8px;color:#475569;letter-spacing:2px;font-weight:700;margin-bottom:5px;'>{name}</div>
                <div style='font-size:15px;font-weight:700;color:#f1f5f9;'>{cur_i}{cp:,.0f}</div>
                <div style='font-size:11px;color:{color};font-weight:700;'>{arrow} {abs(chp):.2f}%</div>
                {svg}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Breadth + Gainers + Losers ────────────────────────────────
    b_col,g_col,l_col = st.columns([1.2,1.4,1.4])
    total_b = advances+declines+unchanged or 1
    adv_pct=advances/total_b*100; dec_pct=declines/total_b*100
    if adv_pct>=60:   slbl,scol="BULLISH 🐂","#00ff9d"
    elif dec_pct>=60: slbl,scol="BEARISH 🐻","#ff0055"
    else:             slbl,scol="NEUTRAL ➡️","#e3b341"

    with b_col:
        st.markdown(f"""
        <div style='background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
            border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:20px;'>
            <div style='font-size:9px;color:#475569;letter-spacing:3px;font-weight:700;
                margin-bottom:14px;font-family:"Fira Code",monospace;'>MARKET BREADTH</div>
            <div style='font-size:18px;font-weight:700;color:{scol};text-align:center;margin-bottom:16px;'>{slbl}</div>
            <div style='margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                    <span style='font-size:11px;color:#00ff9d;font-weight:600;'>▲ Advances</span>
                    <span style='font-size:12px;font-weight:700;color:#00ff9d;'>{advances}</span>
                </div>
                <div style='background:rgba(255,255,255,0.05);border-radius:4px;height:5px;'>
                    <div style='width:{adv_pct:.0f}%;height:5px;background:#00ff9d;border-radius:4px;'></div>
                </div>
            </div>
            <div style='margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                    <span style='font-size:11px;color:#ff0055;font-weight:600;'>▼ Declines</span>
                    <span style='font-size:12px;font-weight:700;color:#ff0055;'>{declines}</span>
                </div>
                <div style='background:rgba(255,255,255,0.05);border-radius:4px;height:5px;'>
                    <div style='width:{dec_pct:.0f}%;height:5px;background:#ff0055;border-radius:4px;'></div>
                </div>
            </div>
            <div>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                    <span style='font-size:11px;color:#e3b341;font-weight:600;'>◆ Unchanged</span>
                    <span style='font-size:12px;font-weight:700;color:#e3b341;'>{unchanged}</span>
                </div>
                <div style='background:rgba(255,255,255,0.05);border-radius:4px;height:5px;'>
                    <div style='width:{unchanged/total_b*100:.0f}%;height:5px;background:#e3b341;border-radius:4px;'></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    with g_col:
        st.markdown("""
        <div style='background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
            border:1px solid rgba(0,255,157,0.12);border-radius:18px;padding:20px;'>
            <div style='font-size:9px;color:#00ff9d;letter-spacing:3px;font-weight:700;
                margin-bottom:14px;font-family:"Fira Code",monospace;'>🚀 TOP GAINERS</div>""",
            unsafe_allow_html=True)
        for n2,chp2,cp2 in top_gainers:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>
                <span style='font-size:12px;font-weight:700;color:#f1f5f9;'>{n2}</span>
                <div>
                    <span style='font-size:12px;font-weight:700;color:#00ff9d;'>▲ {chp2:.2f}%</span>
                    <span style='font-size:10px;color:#475569;margin-left:8px;'>₹{cp2:,.1f}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with l_col:
        st.markdown("""
        <div style='background:rgba(15,23,42,0.6);backdrop-filter:blur(12px);
            border:1px solid rgba(255,0,85,0.12);border-radius:18px;padding:20px;'>
            <div style='font-size:9px;color:#ff0055;letter-spacing:3px;font-weight:700;
                margin-bottom:14px;font-family:"Fira Code",monospace;'>📉 TOP LOSERS</div>""",
            unsafe_allow_html=True)
        for n2,chp2,cp2 in top_losers:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>
                <span style='font-size:12px;font-weight:700;color:#f1f5f9;'>{n2}</span>
                <div>
                    <span style='font-size:12px;font-weight:700;color:#ff0055;'>▼ {abs(chp2):.2f}%</span>
                    <span style='font-size:10px;color:#475569;margin-left:8px;'>₹{cp2:,.1f}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── How It Works ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style='text-align:center;margin-bottom:18px;'>
        <span style='font-size:10px;color:#475569;letter-spacing:4px;font-weight:700;
            font-family:"Fira Code",monospace;'>// HOW IT WORKS</span>
    </div>""", unsafe_allow_html=True)
    h1,h2,h3,h4 = st.columns(4)
    for col_h,num_h,icon_h,title_h,desc_h,delay in [
        (h1,"01","🎯","Enter Ticker",  "Type any NSE/BSE/US ticker in sidebar","0.1s"),
        (h2,"02","🧠","Choose Model",  "SimpleRNN · LSTM · GRU architecture","0.2s"),
        (h3,"03","⚡","Click Analyze", "AI trains & generates forecast in 60s","0.3s"),
        (h4,"04","📊","Explore Results","15+ analysis pages await you","0.4s"),
    ]:
        col_h.markdown(f"""
        <div class="glass-card" style='animation-delay:{delay};'>
            <div style='font-size:28px;margin-bottom:10px;'>{icon_h}</div>
            <div style='font-size:22px;font-weight:700;color:#00ff9d;opacity:0.3;
                font-family:"Bebas Neue",sans-serif;letter-spacing:3px;margin-bottom:6px;'>{num_h}</div>
            <div style='font-size:13px;font-weight:700;color:#f1f5f9;margin-bottom:5px;'>{title_h}</div>
            <div style='font-size:10px;color:#475569;line-height:1.6;'>{desc_h}</div>
        </div>""", unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════
#  RUN PIPELINE
# ══════════════════════════════════════════════════════════════════
stock_name = st.session_state.ticker
CUR        = get_currency(stock_name)
page       = st.session_state.page

# Animated progress steps
_prog = st.empty()
_prog.markdown(f"""
<div style='background:{T["bg_card2"]};border:1px solid {T["border"]};border-radius:16px;
    padding:28px 32px;text-align:center;margin:20px 0;'>
    <div style='font-size:13px;color:#00ff9d;font-weight:700;letter-spacing:2px;margin-bottom:20px;'>
        ⚡ ANALYZING {stock_name}</div>
    <div style='display:flex;justify-content:center;gap:8px;flex-wrap:wrap;'>
        {"".join([f"<div style='font-size:10px;color:#e3b341;background:rgba(227,179,65,0.08);border:1px solid rgba(227,179,65,0.2);padding:6px 14px;border-radius:20px;'>{'✓' if False else '⏳'} {s}</div>" for s in ["Fetching Data","Preprocessing","Training "+st.session_state.model_type,"Generating Forecast","Computing Signals"]])}
    </div>
</div>""", unsafe_allow_html=True)

result, error = run_pipeline(stock_name, epochs=epochs_input, batch_size=batch_input,
    window=window_input, horizon=horizon_input, model_type=st.session_state.model_type)
_prog.empty()

if error:
    # Friendly error messages
    err_lower = str(error).lower()
    if "no data" in err_lower or "empty" in err_lower or "period" in err_lower:
        friendly = f"😕 Couldn't find **{stock_name}**. Check the ticker symbol and try again.\n\n💡 **Tips:**\n- NSE stocks: add `.NS` → `RELIANCE.NS`\n- BSE stocks: add `.BO` → `RELIANCE.BO`\n- US stocks: no suffix → `AAPL`, `TSLA`"
    elif "timeout" in err_lower or "connection" in err_lower or "network" in err_lower:
        friendly = f"🌐 Network problem. Please check your internet connection and try again."
    elif "insufficient" in err_lower or "not enough" in err_lower:
        friendly = f"📊 Not enough historical data for **{stock_name}**. Try a more established stock."
    else:
        friendly = f"⚠️ Something went wrong with **{stock_name}**.\n\nError: `{error}`\n\nTry a different ticker or refresh the page."
    st.error(friendly)
    st.stop()

data              = result["data"]
close_prices      = result["close_prices"]
total_actual      = result["total_actual"]
train_pred        = result["train_pred"]
test_pred         = result["test_pred"]
training_data_len = result["training_data_len"]
history           = result["history"]
rmse              = result["rmse"]
current_price     = result["current_price"]
next_price        = result["next_price"]
forecast_dates    = result["forecast_dates"]
forecast_prices   = result["forecast_prices"]
info              = result.get("info", {}) or {}
active_model      = result.get("model_type", "SimpleRNN")

prev_close        = close_prices[-2][0] if len(close_prices)>1 else current_price
price_change      = current_price - prev_close
price_change_pct  = (price_change/prev_close)*100
forecast_change   = forecast_prices[-1][0] - current_price
forecast_flat     = [p[0] for p in forecast_prices]
sign              = "+" if price_change>=0 else ""
fsign             = "+" if forecast_change>=0 else ""

MODEL_COLORS = {"SimpleRNN":"#00b8ff","LSTM":"#00ff9d","GRU":"#bf00ff"}
active_color  = MODEL_COLORS.get(active_model,"#00ff9d")

# 52W proximity badges
w52h = info.get("fiftyTwoWeekHigh")
w52l = info.get("fiftyTwoWeekLow")
badge_html = ""
if isinstance(w52h,(int,float)) and w52h>0:
    pct_from_high = (w52h - current_price)/w52h*100
    if pct_from_high < 5:
        badge_html += "<span class='alert-badge badge-danger'>🔴 NEAR 52W HIGH</span>"
    elif pct_from_high < 15:
        badge_html += "<span class='alert-badge badge-warning'>🟡 APPROACHING HIGH</span>"
if isinstance(w52l,(int,float)) and w52l>0:
    pct_from_low = (current_price - w52l)/w52l*100
    if pct_from_low < 5:
        badge_html += "<span class='alert-badge badge-safe'>🟢 NEAR 52W LOW</span>"

# ══════════════════════════════════════════════════════════════════
#  HEADER  (always visible)
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:flex-end;
    margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid {T['border']};'>
    <div>
        <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
            <div style='font-size:11px;color:{T['text_muted']};letter-spacing:3px;text-transform:uppercase;'>
                {info.get("shortName","TICKER ANALYSIS")}</div>
            <div style='font-size:9px;font-weight:700;letter-spacing:2px;color:{active_color};
                background:rgba(0,0,0,0.3);border:1px solid {active_color}44;
                padding:2px 8px;border-radius:20px;'>{active_model}</div>
            {badge_html}
        </div>
        <div class='big-ticker'>{stock_name}</div>
    </div>
    <div style='text-align:right;'>
        <div class='current-price'>{CUR}{current_price:,.2f}</div>
        <div style='font-family:"Fira Code",monospace;color:{"#00ff9d" if price_change>=0 else "#ff0055"};font-size:16px;'>
            {sign}{CUR}{abs(price_change):.2f} &nbsp; ({sign}{price_change_pct:.2f}%)
        </div>
        <div style='font-size:10px;color:{T['text_faint']};margin-top:4px;'>
            {datetime.now().strftime("%d %b %Y · %H:%M")}
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# Session metrics
if info:
    last  = data.iloc[-1]
    d_o   = last.get("Open",0); d_h=last.get("High",0); d_l=last.get("Low",0); d_c=last.get("Close",0)
    mcap  = info.get("marketCap","N/A")
    if isinstance(mcap,(int,float)):
        mcap = f"{CUR}{mcap/1e12:.2f}T" if mcap>=1e12 else (f"{CUR}{mcap/1e9:.2f}B" if mcap>=1e9 else f"{CUR}{mcap/1e6:.2f}M")

    # Sparkline last 30 days
    spark_prices = close_prices[-30:].flatten().tolist()
    spark_color  = "#00ff9d" if spark_prices[-1]>=spark_prices[0] else "#ff0055"

    r1c1,r1c2,r1c3,r1c4,r1c5,r1c6 = st.columns(6)
    r1c1.metric("Open",       f"{CUR}{d_o:.2f}")
    r1c2.metric("High",       f"{CUR}{d_h:.2f}", f"+{d_h-d_o:.2f}")
    r1c3.metric("Low",        f"{CUR}{d_l:.2f}",  f"{d_l-d_o:.2f}", delta_color="inverse")
    r1c4.metric("Close",      f"{CUR}{d_c:.2f}")
    r1c5.metric("Market Cap", mcap)
    r1c6.metric("P/E Ratio",  f"{info.get('trailingPE','N/A')}")

    # 52W alert row
    if isinstance(w52h,(int,float)) and isinstance(w52l,(int,float)):
        pct_h = (w52h-current_price)/w52h*100
        pct_l = (current_price-w52l)/w52l*100
        st.markdown(f"""
        <div style='display:flex;gap:12px;margin:8px 0 4px;flex-wrap:wrap;'>
            <div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:10px;
                padding:8px 16px;font-size:11px;flex:1;'>
                <span style='color:{T['text_faint']};'>52W HIGH</span>
                <span style='color:{T['text_metric']};font-weight:700;margin-left:8px;'>{CUR}{w52h:,.2f}</span>
                <span class='alert-badge {"badge-danger" if pct_h<5 else "badge-warning" if pct_h<15 else "badge-safe"}'>
                    {pct_h:.1f}% away
                </span>
            </div>
            <div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:10px;
                padding:8px 16px;font-size:11px;flex:1;'>
                <span style='color:{T['text_faint']};'>52W LOW</span>
                <span style='color:{T['text_metric']};font-weight:700;margin-left:8px;'>{CUR}{w52l:,.2f}</span>
                <span class='alert-badge {"badge-safe" if pct_l<5 else "badge-warning" if pct_l<15 else "badge-safe"}'>
                    {pct_l:.1f}% above
                </span>
            </div>
            <div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:10px;
                padding:8px 16px;font-size:11px;flex:1;'>
                <span style='color:{T['text_faint']};'>TOMORROW RNN</span>
                <span style='color:{"#00ff9d" if next_price>=current_price else "#ff0055"};font-weight:700;margin-left:8px;'>
                    {CUR}{next_price:,.2f}
                </span>
                <span class='alert-badge {"badge-safe" if next_price>=current_price else "badge-danger"}'>
                    {"▲" if next_price>=current_price else "▼"} {abs(next_price-current_price)/current_price*100:.2f}%
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

st.divider()

# Forecast summary cards
c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.metric("Tomorrow",      f"{CUR}{next_price:,.2f}", f"{'▲' if price_change>=0 else '▼'} {abs(price_change_pct):.2f}%", delta_color="normal" if price_change>=0 else "inverse")
with c2: st.metric("Test RMSE",     f"{CUR}{rmse:.2f}",       "Prediction Error",   delta_color="off")
with c3: st.metric("Forecast High", f"{CUR}{max(forecast_flat):,.2f}", "Peak",       delta_color="normal")
with c4: st.metric("Forecast Low",  f"{CUR}{min(forecast_flat):,.2f}", "Trough",     delta_color="inverse")
with c5:
    nc = (forecast_flat[-1]-current_price)/current_price*100
    st.metric(f"{horizon_input}-Day Change", f"{CUR}{fsign}{forecast_change:.2f}", f"{fsign}{nc:.2f}%", delta_color="normal" if nc>=0 else "inverse")

st.divider()

# ══════════════════════════════════════════════════════════════════
#  PAGE ROUTING
# ══════════════════════════════════════════════════════════════════

# ── Dashboard ─────────────────────────────────────────────────────
if page == "Dashboard":
    col_l, col_r = st.columns([1.6, 1])

    with col_l:
        # Company profile
        summary = info.get("longBusinessSummary","")
        if summary:
            st.markdown("<div class='section-sub'>// COMPANY PROFILE</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:13px;color:{T['text_muted']};line-height:1.65;background:{T['bg_card2']};padding:18px;border-radius:12px;border:1px solid {T['border']};margin-bottom:16px;'>{summary[:500]}{'...' if len(summary)>500 else ''}</div>", unsafe_allow_html=True)

        # News
        st.markdown("<div class='section-sub'>// LATEST HEADLINES</div>", unsafe_allow_html=True)
        news_list = info.get("news", [])
        if not news_list:
            short = info.get("shortName", stock_name)
            news_list = fetch_google_news(f"{short} stock", stock_name)

        if news_list:
            shown = 0
            for item in news_list:
                parsed = parse_news_item(item)
                if not parsed:
                    t2 = (item.get("title","")).strip()
                    if not t2: continue
                    parsed = (t2, item.get("publisher","News Source"), item.get("link",""), str(item.get("providerPublishTime","N/A")))
                title, pub, link, dt = parsed
                href = link if link.startswith("http") else "#"
                st.markdown(f"""
                <div class='news-card'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                        <span style='font-size:10px;color:#00b8ff;font-weight:600;letter-spacing:1px;'>{pub.upper()}</span>
                        <span style='font-size:9px;color:{T['text_faint']};'>{dt}</span>
                    </div>
                    <div style='font-size:13px;font-weight:600;color:{"#f1f5f9" if DARK else "#0f172a"};line-height:1.5;'>
                        <a href="{href}" target="_blank" style="text-decoration:none;color:inherit;">{title}</a>
                    </div>
                </div>""", unsafe_allow_html=True)
                shown += 1
                if shown >= 6: break
        else:
            qe = urllib.request.quote(f"{stock_name} stock news")
            st.markdown(f"""
            <div style='padding:24px;border-radius:14px;border:1px solid {T['border']};background:{T['bg_card2']};text-align:center;'>
                <div style='font-size:28px;margin-bottom:8px;'>📰</div>
                <div style='font-size:12px;color:{T['text_muted']};margin-bottom:14px;'>No headlines found for {stock_name}</div>
                <a href="https://www.google.com/search?q={qe}" target="_blank"
                   style='padding:8px 20px;border-radius:8px;background:linear-gradient(135deg,#00ff9d,#00b8ff);
                   color:#000;font-weight:700;font-size:11px;text-decoration:none;letter-spacing:1px;'>🔍 SEARCH GOOGLE</a>
            </div>""", unsafe_allow_html=True)

    with col_r:
        # Analyst
        rec    = (info.get("recommendationKey") or "").replace("_"," ").upper().strip()
        target = info.get("targetMeanPrice")
        n_ana  = info.get("numberOfAnalystOpinions")
        rec_display = rec if rec and rec not in ("NONE","N/A","") else "N/A"
        rec_color   = "#00ff9d" if any(x in rec_display for x in ("BUY","OUTPERFORM","OVERWEIGHT")) else ("#ff0055" if any(x in rec_display for x in ("SELL","UNDERPERFORM","UNDERWEIGHT")) else ("#e3b341" if "HOLD" in rec_display else T['text_faint']))

        st.markdown("<div class='section-sub'>// ANALYST CONSENSUS</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:{T['bg_card2']};padding:20px;border-radius:14px;border:1px solid {T['border']};text-align:center;margin-bottom:14px;'>
            <div style='font-size:9px;color:#64748b;letter-spacing:2px;margin-bottom:8px;'>CONSENSUS RATING</div>
            <div style='font-size:30px;font-weight:700;color:{rec_color};letter-spacing:1px;'>{rec_display}</div>
            {f"<div style='font-size:10px;color:{T['text_faint']};margin-top:4px;'>{n_ana} analysts</div>" if n_ana else ""}
            <hr style='margin:14px 0;opacity:0.08;'>
            <div style='font-size:9px;color:#64748b;letter-spacing:2px;margin-bottom:6px;'>MEAN TARGET</div>
            <div style='font-size:22px;font-weight:700;color:{T['text_metric']};'>{f"{CUR}{target:,.2f}" if isinstance(target,(int,float)) else "N/A"}</div>
            {f"<div style='font-size:11px;color:{'#00ff9d' if isinstance(target,(int,float)) and target>current_price else '#ff0055'};margin-top:4px;'>{'▲' if isinstance(target,(int,float)) and target>current_price else '▼'} {abs((target-current_price)/current_price*100):.1f}% from current</div>" if isinstance(target,(int,float)) else ""}
        </div>""", unsafe_allow_html=True)

        # Profitability
        roe    = info.get("returnOnEquity")
        margin = info.get("profitMargins")
        eps    = info.get("trailingEps")
        divi   = info.get("dividendYield")
        st.markdown("<div class='section-sub'>// KEY RATIOS</div>", unsafe_allow_html=True)
        ratios = [
            ("ROE",             f"{roe*100:.2f}%"    if isinstance(roe,(int,float))    else "N/A"),
            ("Profit Margin",   f"{margin*100:.2f}%" if isinstance(margin,(int,float)) else "N/A"),
            ("EPS (TTM)",       f"{CUR}{eps:.2f}"    if isinstance(eps,(int,float))    else "N/A"),
            ("Dividend Yield",  f"{divi*100:.2f}%"   if isinstance(divi,(int,float))   else "N/A"),
        ]
        for label, val in ratios:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                padding:9px 14px;margin-bottom:5px;border-radius:9px;
                background:{T['bg_card2']};border:1px solid {T['border']};'>
                <span style='font-size:11px;color:{T['text_faint']};'>{label}</span>
                <span style='font-size:12px;font-weight:700;color:{T['text_metric']};'>{val}</span>
            </div>""", unsafe_allow_html=True)

# ── Candlestick ───────────────────────────────────────────────────
elif page == "Candlestick":
    st.markdown("<div class='section-sub'>// OHLC CANDLESTICK CHART</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{T['text_faint']};margin-bottom:16px;'>Green candle = Close > Open (Bullish) &nbsp;|&nbsp; Red candle = Close &lt; Open (Bearish) &nbsp;|&nbsp; EMA lines for trend</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_candlestick(data, stock_name, CUR), use_container_width=True)

    # Mini sparkline summary cards
    st.markdown("<div class='section-sub' style='margin-top:8px;'>// PRICE SPARKLINES</div>", unsafe_allow_html=True)
    periods = [("7D", 7), ("1M", 30), ("3M", 90), ("1Y", 252)]
    sp_cols = st.columns(4)
    for col, (label, days) in zip(sp_cols, periods):
        pts = close_prices[-days:].flatten()
        chg = (pts[-1]-pts[0])/pts[0]*100
        col.markdown(f"""
        <div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:14px;padding:14px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>
                <span style='font-size:11px;font-weight:700;color:{T['text_muted']};letter-spacing:2px;'>{label}</span>
                <span style='font-size:11px;font-weight:700;color:{"#00ff9d" if chg>=0 else "#ff0055"};'>
                    {"+" if chg>=0 else ""}{chg:.1f}%
                </span>
            </div>
            <div style='font-size:18px;font-weight:700;color:{T['text_metric']};'>{CUR}{pts[-1]:,.2f}</div>
        </div>""", unsafe_allow_html=True)
        col.plotly_chart(chart_sparkline(pts.tolist(), "#00ff9d" if chg>=0 else "#ff0055"), use_container_width=True)

# ── Forecast ──────────────────────────────────────────────────────
elif page == f"{horizon_input}-Day Forecast":
    st.markdown(f"<div class='section-sub'>// {horizon_input}-DAY PRICE FORECAST</div>", unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)
    fdf = pd.DataFrame({"Date":[d.strftime("%Y-%m-%d") for d in forecast_dates], f"Predicted ({CUR})":[round(p[0],2) for p in forecast_prices], "Day":[f"Day {i+1}" for i in range(len(forecast_dates))]})
    with ec1:
        st.download_button("📥 DOWNLOAD CSV",   data=fdf.to_csv(index=False).encode(), file_name=f"{stock_name}_{horizon_input}day.csv",  mime="text/csv", use_container_width=True)
    with ec2:
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w: fdf.to_excel(w, index=False, sheet_name="Forecast")
        st.download_button("📊 DOWNLOAD EXCEL", data=buf.getvalue(), file_name=f"{stock_name}_{horizon_input}day.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    st.plotly_chart(chart_forecast(data, forecast_dates, forecast_prices, current_price, stock_name, horizon_input, CUR), use_container_width=True)

    st.markdown("<div class='section-sub' style='margin-top:8px;'>// DAILY BREAKDOWN</div>", unsafe_allow_html=True)
    prev = current_price
    for row_start in range(0, horizon_input, 7):
        chunk = list(zip(forecast_dates, forecast_prices))[row_start:row_start+7]
        cols  = st.columns(len(chunk))
        for i,(col,(date,pa)) in enumerate(zip(cols,chunk)):
            price=pa[0]; chg=price-prev; chgp=chg/prev*100
            col.metric(f"Day {row_start+i+1}·{date.strftime('%b %d')}", f"{CUR}{price:,.2f}", f"{'▲' if chg>=0 else '▼'} {abs(chgp):.1f}%", delta_color="normal" if chg>=0 else "inverse")
            prev=price

# ── Technical Analysis ────────────────────────────────────────────
elif page == "Technical":
    # RSI Gauge
    data_rsi = data.copy()
    delta_rsi = data_rsi["Close"].diff()
    gain_rsi  = delta_rsi.where(delta_rsi>0,0).rolling(14).mean()
    loss_rsi  = (-delta_rsi.where(delta_rsi<0,0)).rolling(14).mean()
    rsi_series = 100-(100/(1+gain_rsi/loss_rsi))
    current_rsi = float(rsi_series.dropna().iloc[-1])

    g1, g2 = st.columns([1,2])
    with g1:
        st.markdown("<div class='section-sub'>// RSI GAUGE</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_rsi_gauge(current_rsi), use_container_width=True)
    with g2:
        st.markdown("<div class='section-sub'>// RSI · MACD · VOLUME</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_indicators(data), use_container_width=True)

    st.markdown("<div class='section-sub' style='margin-top:16px;'>// HISTORICAL + BOLLINGER BANDS</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_historical(data, stock_name), use_container_width=True)

    with st.expander("📋 Raw Data Table (Last 30 Days)"):
        st.dataframe(data[["Open","High","Low","Close","Volume"]].tail(30).style.format("{:,.2f}"), use_container_width=True)

# ── Model Performance ─────────────────────────────────────────────
elif page == "Model Performance":
    st.markdown(f"<div class='section-sub'>// {active_model} — PREDICTIONS vs ACTUAL</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_train_test(total_actual, train_pred, test_pred, training_data_len), use_container_width=True)

    ca,cb,cc = st.columns(3)
    ca.metric("Training Samples", f"{len(train_pred):,}")
    cb.metric("Testing Samples",  f"{len(test_pred):,}")
    cc.metric("Test RMSE",        f"{CUR}{rmse:.2f}")

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-sub' style='margin-top:16px;'>// TRAINING LOSS CURVE</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_loss_curve(history), use_container_width=True)
        lv = history.history["loss"]
        ll1,ll2,ll3 = st.columns(3)
        ll1.metric("Initial Loss", f"{lv[0]:.5f}")
        ll2.metric("Final Loss",   f"{lv[-1]:.5f}")
        ll3.metric("Improvement",  f"{(lv[0]-lv[-1])/lv[0]*100:.1f}%")
    with c2:
        st.markdown("<div class='section-sub' style='margin-top:16px;'>// ACTUAL vs PREDICTED SCATTER</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_scatter(total_actual, train_pred, test_pred), use_container_width=True)

    st.markdown("<div class='section-sub' style='margin-top:16px;'>// RESIDUAL ANALYSIS</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_residuals(total_actual, train_pred, test_pred, training_data_len), use_container_width=True)
    n_te  = len(test_pred.flatten())
    te_r  = total_actual.flatten()[-n_te:] - test_pred.flatten()
    cr1,cr2,cr3,cr4 = st.columns(4)
    cr1.metric("Test MAE",          f"{np.mean(np.abs(te_r)):.2f}")
    cr2.metric("Test RMSE",         f"{np.sqrt(np.mean(te_r**2)):.2f}")
    cr3.metric("Max Over-predict",  f"{te_r.max():.2f}")
    cr4.metric("Max Under-predict", f"{abs(te_r.min()):.2f}")


# ── Buy/Sell Signals ──────────────────────────────────────────────
elif page == "Buy/Sell Signals":
    st.markdown("<div class='section-sub'>// BUY / SELL SIGNAL GENERATOR</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{T['text_faint']};margin-bottom:16px;'>Signals based on EMA 9/21 crossover + RSI confirmation. Green ▲ = BUY, Red ▼ = SELL. Not financial advice.</div>", unsafe_allow_html=True)

    sig_fig, sig_df = chart_signals(data, stock_name, CUR)
    st.plotly_chart(sig_fig, use_container_width=True)

    # Current signal status
    last_row   = sig_df.iloc[-1]
    cur_rsi    = float(sig_df["RSI"].dropna().iloc[-1])
    ema9_now   = float(sig_df["EMA9"].iloc[-1])
    ema21_now  = float(sig_df["EMA21"].iloc[-1])
    trend      = "BULLISH" if ema9_now > ema21_now else "BEARISH"
    trend_col  = "#00ff9d" if trend == "BULLISH" else "#ff0055"
    rsi_zone   = "OVERBOUGHT" if cur_rsi>70 else ("OVERSOLD" if cur_rsi<30 else "NEUTRAL")
    rsi_col    = "#ff0055" if rsi_zone=="OVERBOUGHT" else ("#00ff9d" if rsi_zone=="OVERSOLD" else "#e3b341")

    # Overall signal
    if trend=="BULLISH" and rsi_zone!="OVERBOUGHT":
        overall, ocol, oicon = "BUY", "#00ff9d", "▲"
    elif trend=="BEARISH" and rsi_zone!="OVERSOLD":
        overall, ocol, oicon = "SELL", "#ff0055", "▼"
    else:
        overall, ocol, oicon = "HOLD", "#e3b341", "◆"

    s1,s2,s3,s4 = st.columns(4)
    s1.markdown(f"""<div style='background:{T['bg_card2']};border:2px solid {ocol};border-radius:16px;padding:20px;text-align:center;'>
        <div style='font-size:10px;color:{T['text_faint']};letter-spacing:2px;margin-bottom:6px;'>OVERALL SIGNAL</div>
        <div style='font-size:36px;font-weight:700;color:{ocol};'>{oicon} {overall}</div>
    </div>""", unsafe_allow_html=True)
    s2.markdown(f"""<div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:16px;padding:20px;text-align:center;'>
        <div style='font-size:10px;color:{T['text_faint']};letter-spacing:2px;margin-bottom:6px;'>EMA TREND</div>
        <div style='font-size:24px;font-weight:700;color:{trend_col};'>{trend}</div>
        <div style='font-size:10px;color:{T['text_faint']};margin-top:4px;'>EMA9 {">" if ema9_now>ema21_now else "<"} EMA21</div>
    </div>""", unsafe_allow_html=True)
    s3.markdown(f"""<div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:16px;padding:20px;text-align:center;'>
        <div style='font-size:10px;color:{T['text_faint']};letter-spacing:2px;margin-bottom:6px;'>RSI ZONE</div>
        <div style='font-size:24px;font-weight:700;color:{rsi_col};'>{cur_rsi:.1f}</div>
        <div style='font-size:10px;color:{rsi_col};margin-top:4px;'>{rsi_zone}</div>
    </div>""", unsafe_allow_html=True)
    s4.markdown(f"""<div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:16px;padding:20px;text-align:center;'>
        <div style='font-size:10px;color:{T['text_faint']};letter-spacing:2px;margin-bottom:6px;'>SIGNAL COUNT (180D)</div>
        <div style='font-size:20px;font-weight:700;color:#00ff9d;'>▲ {sig_df["buy_signal"].sum()}</div>
        <div style='font-size:20px;font-weight:700;color:#ff0055;'>▼ {sig_df["sell_signal"].sum()}</div>
    </div>""", unsafe_allow_html=True)

    # Recent signal history table
    st.markdown(f"<br><div class='section-sub'>// RECENT SIGNALS</div>", unsafe_allow_html=True)
    buy_dates  = sig_df[sig_df["buy_signal"]].tail(5)
    sell_dates = sig_df[sig_df["sell_signal"]].tail(5)
    all_sigs   = []
    for idx, row in buy_dates.iterrows():
        all_sigs.append({"Date": str(idx)[:10], "Signal": "▲ BUY", "Price": f"{CUR}{float(row['Close']):.2f}", "RSI": f"{float(row['RSI']):.1f}"})
    for idx, row in sell_dates.iterrows():
        all_sigs.append({"Date": str(idx)[:10], "Signal": "▼ SELL", "Price": f"{CUR}{float(row['Close']):.2f}", "RSI": f"{float(row['RSI']):.1f}"})
    if all_sigs:
        all_sigs = sorted(all_sigs, key=lambda x: x["Date"], reverse=True)[:8]
        for sig in all_sigs:
            is_buy = "BUY" in sig["Signal"]
            sc2 = "#00ff9d" if is_buy else "#ff0055"
            st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                padding:10px 16px;margin-bottom:5px;border-radius:10px;
                background:{T['bg_card2']};border:1px solid {"rgba(0,255,157,0.2)" if is_buy else "rgba(255,0,85,0.2)"};'>
                <span style='font-size:12px;font-weight:700;color:{sc2};'>{sig["Signal"]}</span>
                <span style='font-size:12px;color:{T['text_muted']};'>{sig["Date"]}</span>
                <span style='font-size:12px;font-weight:700;color:{T['text_metric']};'>{sig["Price"]}</span>
                <span style='font-size:11px;color:#e3b341;'>RSI {sig["RSI"]}</span>
            </div>""", unsafe_allow_html=True)

# ── Compare Stocks ─────────────────────────────────────────────────
elif page == "Compare Stocks":
    st.markdown("<div class='section-sub'>// STOCK COMPARISON MODE</div>", unsafe_allow_html=True)

    ca_col, cb_col = st.columns([3,1])
    with ca_col:
        b_input = st.text_input("Compare with (ticker)", value=st.session_state.compare_b,
            placeholder="e.g. TCS.NS / GOOGL", label_visibility="collapsed")
    with cb_col:
        if st.button("COMPARE →", key="cmp_btn") and b_input.strip():
            st.session_state.compare_b   = b_input.strip().upper()
            st.session_state.compare_run = True
            st.rerun()

    if st.session_state.compare_run and st.session_state.compare_b:
        ticker_b = st.session_state.compare_b
        cur_b    = get_currency(ticker_b)
        with st.spinner(f"Loading {ticker_b}..."):
            data_b, info_b = load_data(ticker_b)

        if data_b is None or data_b.empty:
            st.error(f"Could not load data for {ticker_b}")
        else:
            # Header comparison
            price_a = float(data["Close"].squeeze().iloc[-1])
            price_b = float(data_b["Close"].squeeze().iloc[-1])
            chg_a   = (price_a - float(data["Close"].squeeze().iloc[-2]))/float(data["Close"].squeeze().iloc[-2])*100
            chg_b   = (price_b - float(data_b["Close"].squeeze().iloc[-2]))/float(data_b["Close"].squeeze().iloc[-2])*100

            h1,hm,h2 = st.columns([2,1,2])
            with h1:
                st.markdown(f"""<div style='background:{T['bg_card2']};border:2px solid #00ff9d;border-radius:16px;padding:20px;text-align:center;'>
                    <div style='font-size:10px;color:#00ff9d;letter-spacing:3px;margin-bottom:4px;'>STOCK A</div>
                    <div style='font-size:28px;font-weight:700;color:{T['text_metric']};'>{stock_name}</div>
                    <div style='font-size:22px;font-weight:700;color:#00ff9d;margin-top:4px;'>{CUR}{price_a:,.2f}</div>
                    <div style='font-size:12px;color:{"#00ff9d" if chg_a>=0 else "#ff0055"};'>{"▲" if chg_a>=0 else "▼"} {abs(chg_a):.2f}%</div>
                </div>""", unsafe_allow_html=True)
            with hm:
                st.markdown(f"""<div style='text-align:center;padding-top:30px;'>
                    <div style='font-size:28px;color:{T['text_faint']};font-weight:700;'>VS</div>
                </div>""", unsafe_allow_html=True)
            with h2:
                st.markdown(f"""<div style='background:{T['bg_card2']};border:2px solid #00b8ff;border-radius:16px;padding:20px;text-align:center;'>
                    <div style='font-size:10px;color:#00b8ff;letter-spacing:3px;margin-bottom:4px;'>STOCK B</div>
                    <div style='font-size:28px;font-weight:700;color:{T['text_metric']};'>{ticker_b}</div>
                    <div style='font-size:22px;font-weight:700;color:#00b8ff;margin-top:4px;'>{cur_b}{price_b:,.2f}</div>
                    <div style='font-size:12px;color:{"#00ff9d" if chg_b>=0 else "#ff0055"};'>{"▲" if chg_b>=0 else "▼"} {abs(chg_b):.2f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Normalized chart
            cmp_fig = chart_comparison(data, data_b, stock_name, ticker_b)
            if cmp_fig:
                st.plotly_chart(cmp_fig, use_container_width=True)

            # Metrics table
            st.markdown("<div class='section-sub'>// KEY METRICS COMPARISON</div>", unsafe_allow_html=True)
            def safe(info, key, fmt=None, mul=1):
                v = info.get(key)
                if not isinstance(v,(int,float)): return "N/A"
                v = v * mul
                return fmt.format(v) if fmt else str(round(v,2))

            rows = [
                ("Market Cap",     safe(info,  "marketCap",          "{:.2f}B", 1/1e9), safe(info_b,"marketCap",          "{:.2f}B",1/1e9)),
                ("P/E Ratio",      safe(info,  "trailingPE",         "{:.2f}"),         safe(info_b,"trailingPE",         "{:.2f}")),
                ("EPS (TTM)",      safe(info,  "trailingEps",        "{:.2f}"),         safe(info_b,"trailingEps",        "{:.2f}")),
                ("Profit Margin",  safe(info,  "profitMargins",      "{:.1f}%", 100),   safe(info_b,"profitMargins",      "{:.1f}%",100)),
                ("ROE",            safe(info,  "returnOnEquity",     "{:.1f}%", 100),   safe(info_b,"returnOnEquity",     "{:.1f}%",100)),
                ("Dividend Yield", safe(info,  "dividendYield",      "{:.2f}%", 100),   safe(info_b,"dividendYield",      "{:.2f}%",100)),
                ("52W High",       safe(info,  "fiftyTwoWeekHigh",   "{:.2f}"),         safe(info_b,"fiftyTwoWeekHigh",   "{:.2f}")),
                ("52W Low",        safe(info,  "fiftyTwoWeekLow",    "{:.2f}"),         safe(info_b,"fiftyTwoWeekLow",    "{:.2f}")),
            ]
            tc1,tc2,tc3 = st.columns([2,2,2])
            tc1.markdown(f"<div style='font-size:11px;color:{T['text_faint']};font-weight:700;padding:8px 0;'>METRIC</div>", unsafe_allow_html=True)
            tc2.markdown(f"<div style='font-size:11px;color:#00ff9d;font-weight:700;padding:8px 0;'>{stock_name}</div>", unsafe_allow_html=True)
            tc3.markdown(f"<div style='font-size:11px;color:#00b8ff;font-weight:700;padding:8px 0;'>{ticker_b}</div>", unsafe_allow_html=True)
            for label, val_a, val_b in rows:
                rc1,rc2,rc3 = st.columns([2,2,2])
                rc1.markdown(f"<div style='padding:7px 0;font-size:11px;color:{T['text_faint']};border-bottom:1px solid {T['border']};'>{label}</div>", unsafe_allow_html=True)
                rc2.markdown(f"<div style='padding:7px 0;font-size:12px;font-weight:700;color:{T['text_metric']};border-bottom:1px solid {T['border']};'>{val_a}</div>", unsafe_allow_html=True)
                rc3.markdown(f"<div style='padding:7px 0;font-size:12px;font-weight:700;color:{T['text_metric']};border-bottom:1px solid {T['border']};'>{val_b}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='text-align:center;padding:60px;background:{T['bg_card2']};
            border:1px solid {T['border']};border-radius:16px;'>
            <div style='font-size:40px;margin-bottom:12px;'>⚖️</div>
            <div style='font-size:16px;color:{T['text_muted']};'>Enter a second ticker above and click COMPARE</div>
            <div style='font-size:11px;color:{T['text_faint']};margin-top:8px;'>e.g. compare MRF.NS with HEROMOTOCO.NS</div>
        </div>""", unsafe_allow_html=True)

# ── Portfolio ──────────────────────────────────────────────────────
elif page == "Portfolio":
    st.markdown("<div class='section-sub'>// PORTFOLIO TRACKER</div>", unsafe_allow_html=True)

    # Add stock form
    with st.expander("➕ ADD STOCK TO PORTFOLIO", expanded=len(st.session_state.portfolio)==0):
        pf1,pf2,pf3,pf4 = st.columns([2,1,1,1])
        with pf1:
            p_ticker = st.text_input("Ticker", placeholder="RELIANCE.NS", key="pf_ticker", label_visibility="collapsed")
        with pf2:
            p_shares = st.number_input("Shares", min_value=0.01, value=1.0, step=1.0, key="pf_shares", label_visibility="collapsed")
        with pf3:
            p_price  = st.number_input("Buy Price", min_value=0.01, value=100.0, step=10.0, key="pf_price", label_visibility="collapsed")
        with pf4:
            if st.button("ADD →", key="pf_add") and p_ticker.strip():
                st.session_state.portfolio[p_ticker.strip().upper()] = {
                    "shares": p_shares, "buy_price": p_price
                }
                st.rerun()

    if not st.session_state.portfolio:
        st.markdown(f"""<div style='text-align:center;padding:60px;background:{T['bg_card2']};
            border:1px solid {T['border']};border-radius:16px;'>
            <div style='font-size:40px;margin-bottom:12px;'>💼</div>
            <div style='font-size:15px;color:{T['text_muted']};'>Portfolio is empty — ADD stock above</div>
        </div>""", unsafe_allow_html=True)
    else:
        total_invested = 0; total_current = 0
        rows_data = []

        for pticker, pdata in st.session_state.portfolio.items():
            pcur   = get_currency(pticker)
            shares = pdata["shares"]
            bp     = pdata["buy_price"]
            # Fetch current price
            try:
                pdf = yf.Ticker(pticker).history(period="2d")
                cp  = float(pdf["Close"].iloc[-1]) if not pdf.empty else bp
            except:
                cp = bp
            invested  = shares * bp
            current   = shares * cp
            pnl       = current - invested
            pnl_pct   = (pnl / invested) * 100
            total_invested += invested
            total_current  += current
            rows_data.append({
                "ticker": pticker, "cur": pcur, "shares": shares,
                "buy_price": bp, "current_price": cp,
                "invested": invested, "current": current,
                "pnl": pnl, "pnl_pct": pnl_pct
            })

        total_pnl     = total_current - total_invested
        total_pnl_pct = (total_pnl/total_invested*100) if total_invested>0 else 0

        # Summary cards
        sm1,sm2,sm3,sm4 = st.columns(4)
        sm1.metric("Total Invested",  f"{total_invested:,.0f}")
        sm2.metric("Current Value",   f"{total_current:,.0f}",  f"{'▲' if total_pnl>=0 else '▼'} {abs(total_pnl_pct):.2f}%", delta_color="normal" if total_pnl>=0 else "inverse")
        sm3.metric("Total P&L",       f"{'+'if total_pnl>=0 else ''}{total_pnl:,.0f}", delta_color="normal" if total_pnl>=0 else "inverse")
        sm4.metric("Stocks",          f"{len(rows_data)}")

        st.markdown("<br><div class='section-sub'>// HOLDINGS</div>", unsafe_allow_html=True)

        # Portfolio pie chart
        if len(rows_data) > 1:
            pie_fig = go.Figure(go.Pie(
                labels=[r["ticker"] for r in rows_data],
                values=[r["current"] for r in rows_data],
                hole=0.55,
                marker=dict(colors=["#00ff9d","#00b8ff","#bf00ff","#e3b341","#ff0055","#22c55e"],
                    line=dict(color=T['bg_main'], width=2)),
                textfont=dict(size=12, color="#ffffff"),
                hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
            ))
            pie_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color=T['tick_c']),
                height=280, margin=dict(l=0,r=0,t=20,b=0),
                legend=dict(bgcolor=T['legend_bg'], font=dict(size=11, color=T['tick_c']),
                    orientation="v", x=1, y=0.5),
                showlegend=True,
            )
            pc1, pc2 = st.columns([1,2])
            with pc1:
                st.plotly_chart(pie_fig, use_container_width=True)
            with pc2:
                for r in rows_data:
                    pnl_col = "#00ff9d" if r["pnl"]>=0 else "#ff0055"
                    arrow   = "▲" if r["pnl"]>=0 else "▼"
                    rm_key  = f"rm_pf_{r['ticker']}"
                    col_a, col_b = st.columns([4,1])
                    with col_a:
                        st.markdown(f"""<div style='background:{T['bg_card2']};border:1px solid {T['border']};
                            border-radius:12px;padding:12px 16px;margin-bottom:6px;'>
                            <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <div>
                                    <div style='font-size:14px;font-weight:700;color:{T['text_metric']};'>{r["ticker"]}</div>
                                    <div style='font-size:10px;color:{T['text_faint']};'>{r["shares"]} shares @ {r["cur"]}{r["buy_price"]:,.2f}</div>
                                </div>
                                <div style='text-align:right;'>
                                    <div style='font-size:14px;font-weight:700;color:{T['text_metric']};'>{r["cur"]}{r["current_price"]:,.2f}</div>
                                    <div style='font-size:12px;color:{pnl_col};'>{arrow} {r["cur"]}{abs(r["pnl"]):,.0f} ({arrow}{abs(r["pnl_pct"]):.1f}%)</div>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with col_b:
                        if st.button("✕", key=rm_key):
                            del st.session_state.portfolio[r["ticker"]]
                            db_remove_portfolio(r["ticker"])
                            st.rerun()
        else:
            for r in rows_data:
                pnl_col = "#00ff9d" if r["pnl"]>=0 else "#ff0055"
                arrow   = "▲" if r["pnl"]>=0 else "▼"
                col_a, col_b = st.columns([5,1])
                with col_a:
                    st.markdown(f"""<div style='background:{T['bg_card2']};border:1px solid {T['border']};
                        border-radius:12px;padding:14px 18px;margin-bottom:6px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;'>
                            <div>
                                <div style='font-size:15px;font-weight:700;color:{T['text_metric']};'>{r["ticker"]}</div>
                                <div style='font-size:10px;color:{T['text_faint']};'>{r["shares"]} shares @ {r["cur"]}{r["buy_price"]:,.2f}</div>
                            </div>
                            <div style='text-align:right;'>
                                <div style='font-size:16px;font-weight:700;color:{T['text_metric']};'>{r["cur"]}{r["current_price"]:,.2f}</div>
                                <div style='font-size:13px;color:{pnl_col};'>{arrow} {r["cur"]}{abs(r["pnl"]):,.0f} ({arrow}{abs(r["pnl_pct"]):.1f}%)</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    if st.button("✕", key=f"rm_pf_{r['ticker']}"):
                        del st.session_state.portfolio[r["ticker"]]
                        db_remove_portfolio(r["ticker"])
                        st.rerun()

# ── Live Feed ─────────────────────────────────────────────────────
elif page == "🔴 Live Feed":
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
        <div class='section-sub' style='margin:0;'>// REAL-TIME PRICE FEED</div>
        <div style='width:7px;height:7px;border-radius:50%;background:#ff0055;
            animation:pulseNeon 1.2s ease-in-out infinite;'></div>
        <div style='font-size:9px;color:#ff0055;letter-spacing:2px;font-weight:700;'>LIVE</div>
    </div>
    <div style='font-size:11px;color:{T['text_faint']};margin-bottom:16px;'>
        Polls Yahoo Finance every few seconds. Best during market hours.</div>""", unsafe_allow_html=True)

    ctrl1,ctrl2,ctrl3 = st.columns([1,1,2])
    with ctrl1:
        if st.button("⏹ STOP" if st.session_state.live_feed_on else "▶ START", key="live_toggle"):
            st.session_state.live_feed_on = not st.session_state.live_feed_on
            if not st.session_state.live_feed_on: st.session_state.live_prices=[]
            st.rerun()
    with ctrl2:
        if st.button("🗑 CLEAR", key="live_clear"):
            st.session_state.live_prices=[]
            st.rerun()
    with ctrl3:
        rs = st.select_slider("Refresh", options=[5,10,15,30,60], value=st.session_state.live_refresh, format_func=lambda x:f"Every {x}s")
        st.session_state.live_refresh = rs

    st.markdown("<br>", unsafe_allow_html=True)
    ph_price = st.empty(); ph_chart = st.empty(); ph_status = st.empty()

    ip, its = fetch_live_price(stock_name)
    if ip and (not st.session_state.live_prices or st.session_state.live_prices[-1][1]!=its):
        st.session_state.live_prices.append((its, ip))

    if st.session_state.live_prices:
        lp=st.session_state.live_prices[-1][1]; lts=st.session_state.live_prices[-1][0]
        fp=st.session_state.live_prices[0][1]; sc=lp-fp; sp=(sc/fp*100) if fp else 0
        cc2="#00ff9d" if sc>=0 else "#ff0055"; ar="▲" if sc>=0 else "▼"
        ph_price.markdown(f"""
        <div style='background:{T['bg_card2']};border:1px solid {cc2}33;border-radius:18px;
            padding:24px 32px;display:flex;justify-content:space-between;align-items:center;
            box-shadow:0 0 24px {cc2}0d;'>
            <div>
                <div style='font-size:10px;color:{T['text_faint']};letter-spacing:3px;margin-bottom:4px;'>LIVE · {stock_name}</div>
                <div style='font-family:"Bebas Neue",sans-serif;font-size:64px;letter-spacing:3px;
                    color:{cc2};line-height:1;text-shadow:0 0 24px {cc2}44;'>{CUR}{lp:,.2f}</div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:24px;font-weight:700;color:{cc2};'>{ar} {CUR}{abs(sc):.2f}</div>
                <div style='font-size:14px;color:{cc2};opacity:0.8;'>{ar} {abs(sp):.2f}%</div>
                <div style='font-size:10px;color:{T['text_faint']};margin-top:6px;'>
                    {lts.strftime("%H:%M:%S") if hasattr(lts,"strftime") else str(lts)}</div>
                <div style='font-size:10px;color:{T['text_faint']};'>Ticks: {len(st.session_state.live_prices)}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if len(st.session_state.live_prices)>=2:
            po=[p[1] for p in st.session_state.live_prices]
            s1,s2,s3,s4=st.columns(4)
            s1.metric("Open",  f"{CUR}{fp:,.2f}")
            s2.metric("High",  f"{CUR}{max(po):,.2f}")
            s3.metric("Low",   f"{CUR}{min(po):,.2f}")
            s4.metric("Ticks", f"{len(po)}", f"/{rs}s")
            fl=chart_live_feed(st.session_state.live_prices,CUR,stock_name)
            if fl: ph_chart.plotly_chart(fl,use_container_width=True)
    else:
        ph_price.markdown(f"""
        <div style='background:{T['bg_card2']};border:1px solid {T['border']};border-radius:18px;
            padding:48px;text-align:center;'>
            <div style='font-size:36px;margin-bottom:10px;'>📡</div>
            <div style='font-size:14px;color:{T['text_muted']};'>Press <strong style="color:#00ff9d;">▶ START</strong> to begin</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.live_feed_on:
        ph_status.markdown(f"<div style='margin-top:10px;font-size:10px;color:{T['text_faint']};'><span style='color:#00ff9d;'>●</span> ACTIVE · every {rs}s</div>", unsafe_allow_html=True)
        time.sleep(rs)
        np2,nts=fetch_live_price(stock_name)
        if np2 and nts:
            lt=st.session_state.live_prices[-1][1] if st.session_state.live_prices else None
            if nts!=lt:
                st.session_state.live_prices.append((nts,np2))
                if len(st.session_state.live_prices)>200:
                    st.session_state.live_prices=st.session_state.live_prices[-200:]
        st.rerun()
    else:
        if st.session_state.live_prices:
            ph_status.markdown(f"<div style='margin-top:10px;font-size:10px;color:{T['text_faint']};'>⏸ PAUSED · {len(st.session_state.live_prices)} ticks</div>", unsafe_allow_html=True)


# ── Sentiment Analysis ────────────────────────────────────────────
elif page == "Sentiment":
    st.markdown("<div class='section-sub'>// NEWS SENTIMENT ANALYSIS</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{T['text_faint']};margin-bottom:16px;'>AI-powered keyword sentiment — no API key needed. 100+ positive/negative financial terms.</div>", unsafe_allow_html=True)

    # ── Keyword-based sentiment scorer (no API needed) ────────────
    POSITIVE = [
        "surge","rally","gain","rise","jump","soar","high","record","profit","beat",
        "growth","strong","upgrade","buy","bullish","positive","revenue","outperform",
        "dividend","acquisition","expansion","robust","recovery","momentum","breakout",
        "earnings beat","upside","target raised","overweight","accumulate","recommend",
        "milestone","boost","turnaround","optimistic","uptrend","above estimate",
        "strong buy","market share","new high","order win","contract","partnership"
    ]
    NEGATIVE = [
        "fall","drop","crash","loss","decline","weak","downgrade","sell","bearish",
        "negative","lawsuit","fraud","miss","cut","layoff","recession","default",
        "penalty","probe","investigation","below estimate","earnings miss","downside",
        "underweight","reduce","caution","warning","concern","risk","volatile",
        "slump","plunge","disappoint","struggle","pressure","headwind","debt",
        "deficit","slowdown","underperform","target cut","sell-off","correction"
    ]

    def score_headline(text):
        t = text.lower()
        pos = sum(1 for w in POSITIVE if w in t)
        neg = sum(1 for w in NEGATIVE if w in t)
        raw = pos - neg
        score = max(-1.0, min(1.0, raw * 0.4))
        if score > 0.1:   return score, "POSITIVE", "Positive indicators found"
        elif score < -0.1: return score, "NEGATIVE", "Negative indicators found"
        else:              return score, "NEUTRAL",  "No strong signal"

    # Fetch news
    news_list_s = info.get("news", [])
    if not news_list_s:
        short_name = info.get("shortName", stock_name)
        news_list_s = fetch_google_news(f"{short_name} stock", stock_name)

    if not news_list_s:
        st.warning("No news found for sentiment analysis.")
    else:
        parsed_news = []
        for item in news_list_s[:10]:
            parsed = parse_news_item(item)
            if not parsed:
                t2 = (item.get("title","")).strip()
                if t2: parsed = (t2, item.get("publisher",""), item.get("link",""), "")
            if parsed:
                parsed_news.append(parsed)

        if parsed_news:
            # Score all headlines
            scored = []
            for pn in parsed_news:
                title, pub, link, dt = pn
                sc, sent_label, reason = score_headline(title)
                scored.append((title, pub, link, dt, sc, sent_label, reason))

            # Overall score
            scores     = [s[4] for s in scored]
            overall_sc = float(sum(scores)/len(scores)) if scores else 0.0
            positives  = sum(1 for s in scores if s > 0.1)
            negatives  = sum(1 for s in scores if s < -0.1)
            neutrals   = len(scores) - positives - negatives

            if overall_sc > 0.1:   overall_sent, sent_color, sent_icon = "BULLISH", "#00ff9d", "📈"
            elif overall_sc < -0.1: overall_sent, sent_color, sent_icon = "BEARISH", "#ff0055", "📉"
            else:                   overall_sent, sent_color, sent_icon = "NEUTRAL", "#e3b341", "➡️"

            # Summary text
            summary = f"{stock_name} news sentiment is currently {overall_sent.lower()}. "
            summary += f"Out of {len(scored)} recent headlines, {positives} are positive, {negatives} negative, and {neutrals} neutral. "
            if overall_sc > 0.3:   summary += "Strong buying interest and positive market momentum detected."
            elif overall_sc > 0.1: summary += "Mild positive outlook with some growth indicators."
            elif overall_sc < -0.3: summary += "Significant bearish signals — caution advised."
            elif overall_sc < -0.1: summary += "Some negative news — monitor closely."
            else:                   summary += "Market is in a wait-and-watch mode with mixed signals."

            # ── Gauge chart ───────────────────────────────────────
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(overall_sc * 100, 1),
                title=dict(text="Market Sentiment Score", font=dict(size=14, color=T['text_muted'])),
                number=dict(font=dict(size=32, color=sent_color)),
                gauge=dict(
                    axis=dict(range=[-100,100], tickfont=dict(size=10, color=T['tick_c'])),
                    bar=dict(color=sent_color, thickness=0.25),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    steps=[
                        dict(range=[-100,-30], color="rgba(255,0,85,0.12)"),
                        dict(range=[-30,30],   color="rgba(148,163,184,0.06)"),
                        dict(range=[30,100],   color="rgba(0,255,157,0.12)"),
                    ],
                ),
            ))
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, sans-serif", color=T['tick_c']),
                height=240, margin=dict(l=20,r=20,t=50,b=10),
            )

            g1, g2 = st.columns([1,2])
            with g1:
                st.plotly_chart(gauge_fig, use_container_width=True)
                st.markdown(f"""<div style='text-align:center;margin-top:-10px;'>
                    <span style='font-size:28px;'>{sent_icon}</span>
                    <div style='font-size:20px;font-weight:700;color:{sent_color};margin-top:4px;'>{overall_sent}</div>
                    <div style='font-size:11px;color:{T["text_faint"]};margin-top:4px;'>{len(scored)} headlines analysed</div>
                </div>""", unsafe_allow_html=True)

            with g2:
                st.markdown("<div class='section-sub'>// SENTIMENT SUMMARY</div>", unsafe_allow_html=True)
                st.markdown(f"""<div style='background:{T["bg_card2"]};border:1px solid {sent_color}44;
                    border-radius:14px;padding:18px;font-size:13px;color:{T["text_muted"]};
                    line-height:1.8;margin-bottom:14px;'>{summary}</div>""", unsafe_allow_html=True)

                # Breakdown counts
                cnt1, cnt2, cnt3 = st.columns(3)
                for col_c, label_c, count_c, color_c in [
                    (cnt1, "POSITIVE", positives, "#00ff9d"),
                    (cnt2, "NEUTRAL",  neutrals,  "#e3b341"),
                    (cnt3, "NEGATIVE", negatives, "#ff0055"),
                ]:
                    col_c.markdown(f"""<div style='background:{T["bg_card2"]};border:1px solid {color_c}44;
                        border-radius:10px;padding:12px;text-align:center;'>
                        <div style='font-size:22px;font-weight:700;color:{color_c};'>{count_c}</div>
                        <div style='font-size:9px;color:{T["text_faint"]};letter-spacing:2px;'>{label_c}</div>
                    </div>""", unsafe_allow_html=True)

            # ── Headline breakdown ────────────────────────────────
            st.markdown("<br><div class='section-sub'>// HEADLINE BREAKDOWN</div>", unsafe_allow_html=True)
            for title, pub, link, dt, sc, sent_label, reason in scored:
                sc_col  = "#00ff9d" if sc > 0.1 else ("#ff0055" if sc < -0.1 else "#e3b341")
                sc_icon = "📈" if sc > 0.1 else ("📉" if sc < -0.1 else "➡️")
                bar_w   = abs(sc) * 100
                bdr_col = f"{sc_col}33"
                st.markdown(f"""<div style='background:{T["bg_card2"]};border:1px solid {bdr_col};
                    border-radius:12px;padding:14px 16px;margin-bottom:8px;'>
                    <div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'>
                        <div style='flex:1;'>
                            <div style='font-size:10px;color:#00b8ff;font-weight:600;margin-bottom:4px;'>{pub.upper() if pub else "NEWS"}</div>
                            <div style='font-size:12px;font-weight:600;color:{T["text_metric"]};line-height:1.4;margin-bottom:6px;'>
                                <a href="{link}" target="_blank" style="text-decoration:none;color:inherit;">{title}</a>
                            </div>
                            <div style='font-size:10px;color:{T["text_faint"]};'>{reason}</div>
                            <div style='margin-top:8px;background:rgba(255,255,255,0.05);border-radius:4px;height:4px;'>
                                <div style='width:{bar_w:.0f}%;height:4px;background:{sc_col};border-radius:4px;'></div>
                            </div>
                        </div>
                        <div style='text-align:center;min-width:70px;'>
                            <div style='font-size:18px;'>{sc_icon}</div>
                            <div style='font-size:10px;font-weight:700;color:{sc_col};'>{sent_label}</div>
                            <div style='font-size:13px;color:{sc_col};font-weight:700;'>{sc:+.2f}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)




# ── Backtesting Engine ────────────────────────────────────────────
elif page == "Backtesting":
    st.markdown("<div class='section-sub'>// BACKTESTING ENGINE</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{T['text_faint']};margin-bottom:16px;'> </div>", unsafe_allow_html=True)

    # Settings
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        bt_capital = st.number_input("Initial Capital (₹)", min_value=1000, value=10000, step=1000, key="bt_cap")
    with bc2:
        bt_period = st.selectbox("Backtest Period", ["1 Year","2 Years","3 Years","5 Years"], index=1, key="bt_period")
    with bc3:
        bt_strategy = st.selectbox("Strategy", ["EMA Crossover + RSI","EMA Crossover Only","RSI Only"], key="bt_strat")

    period_days = {"1 Year":252,"2 Years":504,"3 Years":756,"5 Years":1260}[bt_period]

    # Run backtest
    df_bt = data.copy().tail(period_days)
    close_bt = df_bt["Close"].squeeze()

    # Compute signals
    ema9  = close_bt.ewm(span=9).mean()
    ema21 = close_bt.ewm(span=21).mean()
    delta = close_bt.diff()
    gain  = delta.where(delta>0,0).rolling(14).mean()
    loss  = (-delta.where(delta<0,0)).rolling(14).mean()
    rsi   = 100 - (100/(1+gain/(loss+1e-10)))

    # Generate signals based on strategy
    if bt_strategy == "EMA Crossover + RSI":
        buy_sig  = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1)) & (rsi < 65)
        sell_sig = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1)) & (rsi > 35)
    elif bt_strategy == "EMA Crossover Only":
        buy_sig  = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
        sell_sig = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))
    else:  # RSI Only
        buy_sig  = (rsi < 30) & (rsi.shift(1) >= 30)
        sell_sig = (rsi > 70) & (rsi.shift(1) <= 70)

    # Simulate trades
    capital    = float(bt_capital)
    shares     = 0.0
    in_trade   = False
    trades     = []
    equity     = []
    buy_price  = 0.0

    for i in range(len(df_bt)):
        price = float(close_bt.iloc[i])
        date  = df_bt.index[i]
        if not in_trade and buy_sig.iloc[i]:
            shares    = capital / price
            buy_price = price
            in_trade  = True
            trades.append({"date": str(date)[:10], "type": "BUY", "price": price, "shares": round(shares,4)})
        elif in_trade and sell_sig.iloc[i]:
            capital   = shares * price
            pnl       = capital - (shares * buy_price)
            pnl_pct   = (pnl / (shares * buy_price)) * 100
            trades.append({"date": str(date)[:10], "type": "SELL", "price": price,
                          "shares": round(shares,4), "pnl": round(pnl,2), "pnl_pct": round(pnl_pct,2)})
            shares   = 0.0
            in_trade = False
        # Current equity
        curr_val = capital + (shares * price if in_trade else 0)
        equity.append(curr_val)

    # Final value if still holding
    final_price = float(close_bt.iloc[-1])
    final_val   = capital + (shares * final_price if in_trade else 0)
    total_ret   = ((final_val - bt_capital) / bt_capital) * 100
    buy_hold_ret = ((final_price - float(close_bt.iloc[0])) / float(close_bt.iloc[0])) * 100

    # Metrics
    sell_trades = [t for t in trades if t["type"]=="SELL"]
    wins  = [t for t in sell_trades if t.get("pnl",0) > 0]
    losses= [t for t in sell_trades if t.get("pnl",0) <= 0]
    win_rate = (len(wins)/len(sell_trades)*100) if sell_trades else 0

    # Summary cards
    m1,m2,m3,m4,m5 = st.columns(5)
    for col_m, label_m, val_m, color_m in [
        (m1, "Initial Capital",  f"{CUR}{bt_capital:,.0f}",          T['text_metric']),
        (m2, "Final Value",      f"{CUR}{final_val:,.0f}",           "#00ff9d" if final_val>bt_capital else "#ff0055"),
        (m3, "Strategy Return",  f"{'+'if total_ret>=0 else ''}{total_ret:.1f}%", "#00ff9d" if total_ret>=0 else "#ff0055"),
        (m4, "Buy & Hold",       f"{'+'if buy_hold_ret>=0 else ''}{buy_hold_ret:.1f}%", "#00b8ff"),
        (m5, "Win Rate",         f"{win_rate:.0f}%",                 "#00ff9d" if win_rate>50 else "#e3b341"),
    ]:
        col_m.markdown(f"""<div style='background:{T["bg_card2"]};border:1px solid {T["border"]};
            border-radius:12px;padding:14px;text-align:center;'>
            <div style='font-size:9px;color:{T["text_faint"]};letter-spacing:2px;margin-bottom:6px;'>{label_m}</div>
            <div style='font-size:20px;font-weight:700;color:{color_m};'>{val_m}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity curve chart
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        x=list(df_bt.index), y=equity, mode="lines", name="Strategy",
        line=dict(color="#00ff9d", width=2),
        fill="tozeroy", fillcolor="rgba(0,255,157,0.05)"
    ))
    # Buy & Hold line
    bh_vals = [bt_capital * (float(close_bt.iloc[i])/float(close_bt.iloc[0])) for i in range(len(df_bt))]
    eq_fig.add_trace(go.Scatter(
        x=list(df_bt.index), y=bh_vals, mode="lines", name="Buy & Hold",
        line=dict(color="#00b8ff", width=1.5, dash="dash")
    ))
    # Buy/Sell markers
    for t in trades:
        color_t = "#00ff9d" if t["type"]=="BUY" else "#ff0055"
        sym_t   = "triangle-up" if t["type"]=="BUY" else "triangle-down"
        # Find price in equity at that date
        eq_fig.add_trace(go.Scatter(
            x=[t["date"]], y=[t["price"]*bt_capital/float(close_bt.iloc[0])],
            mode="markers", name=t["type"],
            marker=dict(symbol=sym_t, size=10, color=color_t),
            showlegend=False
        ))
    eq_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T["plot_bg"],
        font=dict(family="Outfit, sans-serif", color=T["tick_c"], size=12),
        title=dict(text=f"Equity Curve — {bt_strategy} vs Buy & Hold", font=dict(size=14, color=T["text_muted"])),
        xaxis=dict(gridcolor=T["grid"], tickfont=dict(size=10, color=T["tick_c"])),
        yaxis=dict(gridcolor=T["grid"], tickprefix=CUR, tickfont=dict(size=10, color=T["tick_c"])),
        margin=dict(l=50,r=20,t=50,b=40), hovermode="x unified", height=380,
        legend=dict(bgcolor=T["legend_bg"], bordercolor=T["border"], borderwidth=1,
            font=dict(size=11, color=T["tick_c"]), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # Stats row
    st.markdown("<div class='section-sub'>// TRADE STATISTICS</div>", unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Total Trades",   len(sell_trades))
    s2.metric("Winning Trades", len(wins))
    s3.metric("Losing Trades",  len(losses))
    s4.metric("vs Buy & Hold",  f"{'+'if (total_ret-buy_hold_ret)>=0 else ''}{(total_ret-buy_hold_ret):.1f}%",
        delta_color="normal" if total_ret>=buy_hold_ret else "inverse")

    # Trade log
    if sell_trades:
        st.markdown("<br><div class='section-sub'>// TRADE LOG</div>", unsafe_allow_html=True)
        for t in sell_trades[-10:]:
            pnl_c = "#00ff9d" if t.get("pnl",0)>0 else "#ff0055"
            st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                padding:9px 16px;margin-bottom:4px;background:{T["bg_card2"]};
                border:1px solid {pnl_c}33;border-radius:10px;'>
                <span style='font-size:11px;color:{T["text_faint"]};'>{t["date"]}</span>
                <span style='font-size:12px;font-weight:700;color:{T["text_metric"]};'>{CUR}{t["price"]:,.2f}</span>
                <span style='font-size:12px;font-weight:700;color:{pnl_c};'>
                    {"+" if t.get("pnl",0)>0 else ""}{CUR}{abs(t.get("pnl",0)):,.0f} ({t.get("pnl_pct",0):+.1f}%)</span>
            </div>""", unsafe_allow_html=True)

# ── PDF Report ────────────────────────────────────────────────────
elif page == "PDF Report":
    tf_p = T["text_faint"]; tm_p = T["text_muted"]; bc2_p = T["bg_card2"]; bdr_p = T["border"]
    st.markdown("<div class='section-sub'>// PDF REPORT EXPORT</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{tf_p};margin-bottom:20px;'>Professional PDF report — download in one click !</div>", unsafe_allow_html=True)

    sig_df_pdf  = compute_signals(data)
    cur_rsi_pdf = float(sig_df_pdf["RSI"].dropna().iloc[-1])
    ema9_pdf    = float(sig_df_pdf["EMA9"].iloc[-1])
    ema21_pdf   = float(sig_df_pdf["EMA21"].iloc[-1])
    trend_pdf   = "Bullish" if ema9_pdf > ema21_pdf else "Bearish"
    overall_sig = "BUY" if (trend_pdf=="Bullish" and cur_rsi_pdf<65) else ("SELL" if (trend_pdf=="Bearish" and cur_rsi_pdf>35) else "HOLD")
    sig_col_r   = (0,200,81) if overall_sig=="BUY" else ((255,68,68) if overall_sig=="SELL" else (255,136,0))

    def generate_pdf():
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        import io

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            rightMargin=15*mm, leftMargin=15*mm,
            topMargin=15*mm, bottomMargin=15*mm)

        C_BG    = colors.HexColor("#0b111a")
        C_GREEN = colors.HexColor("#00ff9d")
        C_BLUE  = colors.HexColor("#00b8ff")
        C_WHITE = colors.white
        C_GRAY  = colors.HexColor("#94a3b8")
        C_LIGHT = colors.HexColor("#f8fafc")
        C_BORDER= colors.HexColor("#e2e8f0")
        C_RED   = colors.HexColor("#ff0055")
        C_YELLOW= colors.HexColor("#e3b341")
        C_SIG   = colors.Color(sig_col_r[0]/255, sig_col_r[1]/255, sig_col_r[2]/255)
        C_CHNG  = C_GREEN if price_change >= 0 else C_RED
        styles  = getSampleStyleSheet()

        def sty(size=10, color=None, bold=False, align=TA_LEFT):
            return ParagraphStyle("s", parent=styles["Normal"],
                fontSize=size, textColor=color or colors.black,
                fontName="Helvetica-Bold" if bold else "Helvetica",
                alignment=align, leading=size*1.5)

        def section_title(text):
            story.append(HRFlowable(width="100%", thickness=1, color=C_GREEN, spaceAfter=3))
            story.append(Paragraph(f"<font size='9' color='#475569'><b>{text}</b></font>", sty(9, colors.HexColor("#475569"), bold=True)))
            story.append(Spacer(1, 3*mm))

        def metric_tbl(rows, cols=3):
            while len(rows) % cols != 0: rows.append(("","","#1e293b"))
            cells = []
            for i in range(0, len(rows), cols):
                row_cells = []
                for label, val, vc in rows[i:i+cols]:
                    row_cells.append(Paragraph(
                        f"<font size='8' color='#94a3b8'>{label}</font><br/>"
                        f"<font size='13' color='{vc}'><b>{val}</b></font>",
                        sty(13, align=TA_CENTER)))
                cells.append(row_cells)
            t = Table(cells, colWidths=[f"{100//cols}%" for _ in range(cols)])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),C_LIGHT),
                ("BOX",(0,0),(-1,-1),0.5,C_BORDER),
                ("INNERGRID",(0,0),(-1,-1),0.5,C_BORDER),
                ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
                ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ]))
            return t

        story = []

        # Header
        h = Table([[
            Paragraph("<font color='#00ff9d' size='20'><b>TRADEVISION AI</b></font><br/>"
                      "<font color='#94a3b8' size='8'>DEEP LEARNING STOCK ANALYSIS REPORT</font>",
                      sty(20, C_GREEN, bold=True)),
            Paragraph(f"<font color='#94a3b8' size='8'>Generated<br/>{datetime.now().strftime('%d %B %Y, %I:%M %p')}</font>",
                      sty(8, C_GRAY, align=TA_RIGHT)),
        ]], colWidths=["65%","35%"])
        h.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),C_BG),
            ("TOPPADDING",(0,0),(-1,-1),14),("BOTTOMPADDING",(0,0),(-1,-1),14),
            ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(h)
        story.append(Spacer(1,4*mm))

        # Stock + Price
        arr = "▲" if price_change>=0 else "▼"
        p = Table([[
            Paragraph(f"<font size='16' color='white'><b>{info.get('shortName',stock_name)}</b></font><br/>"
                      f"<font size='9' color='#64748b'>{stock_name}</font>", sty(16, C_WHITE, bold=True)),
            Paragraph(f"<font size='22' color='#00ff9d'><b>{CUR}{current_price:,.2f}</b></font><br/>"
                      f"<font size='10' color='{'#00ff9d' if price_change>=0 else '#ff0055'}'>{arr} {CUR}{abs(price_change):.2f} ({price_change_pct:+.2f}%)</font>",
                      sty(22, C_GREEN, bold=True, align=TA_RIGHT)),
        ]], colWidths=["50%","50%"])
        p.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#131b27")),
            ("TOPPADDING",(0,0),(-1,-1),12),("BOTTOMPADDING",(0,0),(-1,-1),12),
            ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(p)
        story.append(Spacer(1,5*mm))

        

        # Key Metrics
        section_title("// KEY METRICS")
        rsi_c = "#FF4444" if cur_rsi_pdf>70 else ("#00C851" if cur_rsi_pdf<30 else "#FF8800")
        story.append(metric_tbl([
            ("CURRENT PRICE",f"{CUR}{current_price:,.2f}","#1e293b"),
            ("NEXT DAY PRED",f"{CUR}{next_price:,.2f}","#00C851" if next_price>=current_price else "#FF4444"),
            ("TEST RMSE",f"{CUR}{rmse:.2f}","#475569"),
            ("RSI (14)",f"{cur_rsi_pdf:.1f}",rsi_c),
            ("EMA TREND",trend_pdf,"#00C851" if trend_pdf=="Bullish" else "#FF4444"),
            ("MODEL",active_model,"#00b8ff"),
        ]))
        story.append(Spacer(1,5*mm))

        # Forecast Table
        section_title(f"// {horizon_input}-DAY PRICE FORECAST")
        fc_rows = [[
            Paragraph("<font size='9' color='white'><b>DAY</b></font>",sty(9,C_WHITE,bold=True,align=TA_CENTER)),
            Paragraph("<font size='9' color='white'><b>DATE</b></font>",sty(9,C_WHITE,bold=True,align=TA_CENTER)),
            Paragraph("<font size='9' color='white'><b>PREDICTED PRICE</b></font>",sty(9,C_WHITE,bold=True,align=TA_CENTER)),
            Paragraph("<font size='9' color='white'><b>CHANGE</b></font>",sty(9,C_WHITE,bold=True,align=TA_CENTER)),
        ]]
        prev_p = current_price
        for i,(fd,fp) in enumerate(zip(forecast_dates,forecast_prices)):
            chg = fp[0]-prev_p; pct = chg/prev_p*100
            pc  = "#00C851" if fp[0]>=current_price else "#FF4444"
            fc_rows.append([
                Paragraph(f"<font size='10'>Day {i+1}</font>",sty(10,align=TA_CENTER)),
                Paragraph(f"<font size='10'>{fd.strftime('%d %b %Y')}</font>",sty(10,align=TA_CENTER)),
                Paragraph(f"<font size='11' color='{pc}'><b>{CUR}{fp[0]:,.2f}</b></font>",sty(11,align=TA_CENTER)),
                Paragraph(f"<font size='10' color='{pc}'>{'▲' if chg>=0 else '▼'} {abs(pct):.2f}%</font>",sty(10,align=TA_CENTER)),
            ])
            prev_p = fp[0]
        fc_t = Table(fc_rows, colWidths=["15%","30%","30%","25%"])
        fc_t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),C_BG),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_LIGHT,C_WHITE]),
            ("BOX",(0,0),(-1,-1),0.5,C_BORDER),("INNERGRID",(0,0),(-1,-1),0.5,C_BORDER),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(fc_t)
        story.append(Spacer(1,5*mm))

        # Company Info
        section_title("// COMPANY INFORMATION")
        mcap = info.get("marketCap")
        story.append(metric_tbl([
            ("MARKET CAP", f"{CUR}{mcap/1e9:.1f}B" if isinstance(mcap,(int,float)) and mcap>0 else "N/A","#1e293b"),
            ("P/E RATIO",  str(round(float(info.get("trailingPE",0)),1)) if info.get("trailingPE") else "N/A","#1e293b"),
            ("52W HIGH",   f"{CUR}{info.get('fiftyTwoWeekHigh','N/A')}","#1e293b"),
            ("52W LOW",    f"{CUR}{info.get('fiftyTwoWeekLow','N/A')}","#1e293b"),
            ("SECTOR",     str(info.get("sector","N/A")),"#00b8ff"),
            ("ANALYST",    (info.get("recommendationKey") or "N/A").upper(),"#00C851"),
        ]))
        story.append(Spacer(1,6*mm))

        # Disclaimer
        d = Table([[Paragraph(
            "<font size='8' color='#92400e'><b>⚠ DISCLAIMER:</b> TradeVision AI educational purposes only. "
            "NOT financial advice. Consult SEBI-registered advisor before investing. "
            "Past performance does not guarantee future results.</font>",
            sty(8, colors.HexColor("#92400e")))]],colWidths=["100%"])
        d.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#fef3c7")),
            ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#f59e0b")),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
            ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12),
        ]))
        story.append(d)
        story.append(Spacer(1,4*mm))
        story.append(HRFlowable(width="100%",thickness=0.5,color=C_BORDER))
        story.append(Paragraph(
            f"<font size='8' color='#94a3b8'>TradeVision AI · {active_model} · "
            f"{datetime.now().strftime('%d %b %Y')} · Educational Use Only</font>",
            sty(8,C_GRAY,align=TA_CENTER)))

        doc.build(story)
        buf.seek(0)
        return buf.getvalue()

    # UI
    p1, p2 = st.columns(2)
    for col_p, icon_p, lbl_p, val_p, c_p in [
        (p1,"📊","Current Price",f"{CUR}{current_price:,.2f}","#00ff9d"),
        (p2,"🔮","Next Day Prediction",f"{CUR}{next_price:,.2f}","#00b8ff"),
    ]:
        col_p.markdown(f"""<div style='background:{bc2_p};border:1px solid {bdr_p};border-radius:14px;
            padding:20px;text-align:center;margin-bottom:16px;'>
            <div style='font-size:24px;margin-bottom:8px;'>{icon_p}</div>
            <div style='font-size:10px;color:{tf_p};letter-spacing:2px;margin-bottom:6px;'>{lbl_p}</div>
            <div style='font-size:22px;font-weight:700;color:{c_p};'>{val_p}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='background:{bc2_p};border:1px solid rgba(0,255,157,0.2);
        border-radius:14px;padding:18px;margin-bottom:20px;'>
        <div style='font-size:12px;color:{tm_p};line-height:1.9;'>
            📄 <strong style='color:#00ff9d;'>PDF includes:</strong> &nbsp;
            ✅ Stock overview &nbsp;·&nbsp;
            ✅ AI Signal &nbsp;·&nbsp;
            ✅ {horizon_input}-Day forecast table &nbsp;·&nbsp;
            ✅ Key metrics &nbsp;·&nbsp;
            ✅ Company info &nbsp;·&nbsp;
            ✅ Disclaimer
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("📥 GENERATE & DOWNLOAD PDF", use_container_width=True, key="gen_pdf"):
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_pdf()
                st.download_button(
                    label="✅ DOWNLOAD PDF NOW",
                    data=pdf_bytes,
                    file_name=f"{stock_name}_TradeVision_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("✅ PDF is ready! Click on Green button.")
            except ImportError:
                st.error("reportlab is not install . ")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Stock Screener ────────────────────────────────────────────────
elif page == "Stock Screener":
    st.markdown("<div class='section-sub'>// STOCK SCREENER</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{T['text_faint']};margin-bottom:16px;'>NIFTY 50 stocks filter karo — RSI, P/E, 52W range, market cap thi. Real-time data.</div>", unsafe_allow_html=True)

    # Filter controls
    fc1,fc2,fc3,fc4 = st.columns(4)
    with fc1:
        f_rsi_max = st.slider("RSI Max", 10, 100, 70, 5, key="scr_rsi")
        st.caption("RSI < value wala stocks")
    with fc2:
        f_pe_max = st.slider("P/E Max", 5, 100, 30, 5, key="scr_pe")
        st.caption("P/E < value wala stocks")
    with fc3:
        f_52w = st.slider("% from 52W Low (max)", 0, 100, 20, 5, key="scr_52w")
        st.caption("52W Low thi X% andar")
    with fc4:
        f_sector = st.selectbox("Sector", ["All","IT","Banking","Pharma","Auto","Energy","FMCG"], key="scr_sec")

    NIFTY50_STOCKS = {
        "RELIANCE.NS":"Energy","TCS.NS":"IT","HDFCBANK.NS":"Banking",
        "INFY.NS":"IT","ICICIBANK.NS":"Banking","HINDUNILVR.NS":"FMCG",
        "ITC.NS":"FMCG","SBIN.NS":"Banking","BAJFINANCE.NS":"Banking",
        "KOTAKBANK.NS":"Banking","LT.NS":"Engineering","AXISBANK.NS":"Banking",
        "WIPRO.NS":"IT","MARUTI.NS":"Auto","ASIANPAINT.NS":"FMCG",
        "SUNPHARMA.NS":"Pharma","TATAMOTORS.NS":"Auto","TITAN.NS":"FMCG",
        "ULTRACEMCO.NS":"Cement","ONGC.NS":"Energy","NTPC.NS":"Energy",
        "POWERGRID.NS":"Energy","TECHM.NS":"IT","HCLTECH.NS":"IT",
        "BAJAJFINSV.NS":"Banking","DIVISLAB.NS":"Pharma","DRREDDY.NS":"Pharma",
        "CIPLA.NS":"Pharma","EICHERMOT.NS":"Auto","HEROMOTOCO.NS":"Auto",
        "TATASTEEL.NS":"Metal","JSWSTEEL.NS":"Metal","HINDALCO.NS":"Metal",
        "COALINDIA.NS":"Energy","BRITANNIA.NS":"FMCG","NESTLEIND.NS":"FMCG",
        "APOLLOHOSP.NS":"Pharma","ADANIENT.NS":"Energy","ADANIPORTS.NS":"Infra",
        "BPCL.NS":"Energy",
    }

    if st.button("🔍 RUN SCREENER", use_container_width=False, key="run_screener"):
        results = []
        prog = st.progress(0, text="Scanning stocks...")
        stocks_list = [(t,s) for t,s in NIFTY50_STOCKS.items() if f_sector=="All" or s==f_sector]

        for i, (sym, sector) in enumerate(stocks_list):
            prog.progress((i+1)/len(stocks_list), text=f"Scanning {sym}...")
            try:
                tk_s  = yf.Ticker(sym)
                df_s  = tk_s.history(period="1y")
                if df_s.empty or len(df_s) < 50: continue
                info_s = {}
                try: info_s = tk_s.info or {}
                except: pass

                cp_s   = float(df_s["Close"].iloc[-1])
                pp_s   = float(df_s["Close"].iloc[-2])
                chp_s  = (cp_s-pp_s)/pp_s*100
                # RSI
                d_s    = df_s["Close"].diff()
                g_s    = d_s.where(d_s>0,0).rolling(14).mean()
                l_s    = (-d_s.where(d_s<0,0)).rolling(14).mean()
                rsi_s  = float((100-(100/(1+g_s/(l_s+1e-10)))).iloc[-1])
                # 52W
                high52 = float(df_s["Close"].max())
                low52  = float(df_s["Close"].min())
                pct_from_low = (cp_s-low52)/low52*100 if low52>0 else 100
                # PE
                pe_s   = info_s.get("trailingPE", None)

                # Apply filters
                if rsi_s > f_rsi_max: continue
                if f_pe_max < 100 and pe_s and float(pe_s) > f_pe_max: continue
                if pct_from_low > f_52w: continue

                results.append({
                    "ticker": sym.replace(".NS",""),
                    "full":   sym,
                    "sector": sector,
                    "price":  cp_s,
                    "change": chp_s,
                    "rsi":    rsi_s,
                    "pe":     round(float(pe_s),1) if pe_s else None,
                    "52w_low_pct": pct_from_low,
                    "52w_high": high52,
                    "52w_low":  low52,
                })
            except: continue

        prog.empty()

        if not results:
            st.warning("Koi stock match nathi thayo. Filters loosen karo.")
        else:
            st.markdown(f"<div style='font-size:12px;color:#00ff9d;font-weight:700;margin-bottom:12px;'>✅ {len(results)} stocks found</div>", unsafe_allow_html=True)
            # Header
            hc = st.columns([1.5,1,1,1,1,1,1])
            for col_h, lbl_h in zip(hc, ["STOCK","SECTOR","PRICE","CHANGE","RSI","P/E","52W LOW%"]):
                col_h.markdown(f"<div style='font-size:9px;color:{T['text_faint']};letter-spacing:2px;font-weight:700;padding:6px 0;border-bottom:1px solid {T['border']};'>{lbl_h}</div>", unsafe_allow_html=True)

            for r in sorted(results, key=lambda x: x["rsi"]):
                chg_col = "#00ff9d" if r["change"]>=0 else "#ff0055"
                rsi_col = "#00ff9d" if r["rsi"]<30 else ("#e3b341" if r["rsi"]<50 else T["text_muted"])
                rc = st.columns([1.5,1,1,1,1,1,1])
                rc[0].markdown(f"<div style='padding:8px 0;font-size:13px;font-weight:700;color:{T['text_metric']};'>{r['ticker']}</div>", unsafe_allow_html=True)
                rc[1].markdown(f"<div style='padding:8px 0;font-size:11px;color:{T['text_faint']};'>{r['sector']}</div>", unsafe_allow_html=True)
                rc[2].markdown(f"<div style='padding:8px 0;font-size:12px;font-weight:700;color:{T['text_metric']};'>₹{r['price']:,.1f}</div>", unsafe_allow_html=True)
                rc[3].markdown(f"<div style='padding:8px 0;font-size:12px;font-weight:700;color:{chg_col};'>{'▲' if r['change']>=0 else '▼'}{abs(r['change']):.1f}%</div>", unsafe_allow_html=True)
                rc[4].markdown(f"<div style='padding:8px 0;font-size:12px;font-weight:700;color:{rsi_col};'>{r['rsi']:.1f}</div>", unsafe_allow_html=True)
                rc[5].markdown(f"<div style='padding:8px 0;font-size:12px;color:{T['text_muted']};'>{r['pe'] if r['pe'] else 'N/A'}</div>", unsafe_allow_html=True)
                rc[6].markdown(f"<div style='padding:8px 0;font-size:12px;color:#e3b341;'>+{r['52w_low_pct']:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='text-align:center;padding:50px;background:{T["bg_card2"]};
            border:1px solid {T["border"]};border-radius:16px;'>
            <div style='font-size:40px;margin-bottom:12px;'>🔍</div>
            <div style='font-size:14px;color:{T["text_muted"]};'>Set Filters and Click on "RUN SCREENER" </div>
            <div style='font-size:11px;color:{T["text_faint"]};margin-top:8px;'>
                NIFTY 50 na {len(NIFTY50_STOCKS)} stocks scan karshhe
            </div>
        </div>""", unsafe_allow_html=True)

# ── About ─────────────────────────────────────────────────────────
elif page == "About":
    st.markdown("<div class='section-sub'>// ABOUT TRADEVISION AI</div>", unsafe_allow_html=True)

    a1, a2 = st.columns([2,1])
    with a1:
        st.markdown(f"""
        <div style='background:{T["bg_card2"]};border:1px solid {T["border"]};
            border-radius:16px;padding:28px;margin-bottom:16px;'>
            <div style='font-family:"Bebas Neue",sans-serif;font-size:36px;
                letter-spacing:4px;background:linear-gradient(135deg,#00ff9d,#00b8ff);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:8px;'>TRADEVISION AI</div>
            <div style='font-size:11px;color:{T["text_faint"]};letter-spacing:3px;margin-bottom:20px;'>
                v5.0.0 · RECURRENT NEURAL ENGINE</div>
            <div style='font-size:13px;color:{T["text_muted"]};line-height:1.9;'>
                TradeVision AI is an <strong style="color:#00ff9d;">educational stock analysis platform</strong>
                powered by deep learning. It uses Recurrent Neural Networks (RNN, LSTM, GRU)
                to analyze historical price data and generate forecasts.<br><br>
                Built with <strong style="color:#00b8ff;">Streamlit · TensorFlow · yFinance · Plotly</strong>
            </div>
        </div>""", unsafe_allow_html=True)

        # Keyboard shortcuts reference
        st.markdown("<div class='section-sub'>// KEYBOARD SHORTCUTS</div>", unsafe_allow_html=True)
        shortcuts = [
            ("D", "Dashboard"), ("C", "Candlestick"), ("T", "Technical"),
            ("F", "Forecast"), ("P", "Portfolio"), ("L", "Live Feed"), ("S", "Sentiment"),
        ]
        sc_cols = st.columns(4)
        for i, (key, page_s) in enumerate(shortcuts):
            sc_cols[i%4].markdown(f"""
            <div style='background:{T["bg_card2"]};border:1px solid {T["border"]};
                border-radius:10px;padding:10px;text-align:center;margin-bottom:8px;'>
                <div style='font-size:18px;font-weight:700;color:#00ff9d;
                    font-family:monospace;background:rgba(0,255,157,0.1);
                    border-radius:6px;padding:4px 10px;display:inline-block;
                    margin-bottom:4px;'>{key}</div>
                <div style='font-size:10px;color:{T["text_faint"]};'>{page_s}</div>
            </div>""", unsafe_allow_html=True)

    with a2:
        # Tech stack
        st.markdown("<div class='section-sub'>// TECH STACK</div>", unsafe_allow_html=True)
        stack = [
            ("🧠", "TensorFlow", "RNN · LSTM · GRU models", "#00ff9d"),
            ("📊", "Plotly",     "Interactive charts",       "#00b8ff"),
            ("🌐", "Streamlit",  "Web framework",            "#bf00ff"),
            ("📈", "yFinance",   "Market data API",          "#e3b341"),
            ("🐼", "Pandas",     "Data processing",          "#00ff9d"),
            ("🔢", "NumPy",      "Numerical computing",      "#00b8ff"),
        ]
        for icon, name, desc, color in stack:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:12px;padding:10px 14px;
                margin-bottom:6px;background:{T["bg_card2"]};border:1px solid {T["border"]};
                border-radius:10px;'>
                <span style='font-size:18px;'>{icon}</span>
                <div>
                    <div style='font-size:12px;font-weight:700;color:{color};'>{name}</div>
                    <div style='font-size:10px;color:{T["text_faint"]};'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # How to use guide
        st.markdown("<br><div class='section-sub'>// QUICK GUIDE</div>", unsafe_allow_html=True)
        steps = [
            ("1", "Enter ticker in sidebar", "e.g. RELIANCE.NS for NSE"),
            ("2", "Choose AI model",         "LSTM = best accuracy"),
            ("3", "Set epochs & horizon",    "More epochs = better model"),
            ("4", "Click ANALYZE →",         "Takes 30-60 seconds"),
            ("5", "Explore all pages",       "Charts · Signals · Portfolio"),
        ]
        for num, title, hint in steps:
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:flex-start;
                padding:8px 0;border-bottom:1px solid {T["border"]};'>
                <div style='font-size:14px;font-weight:700;color:#00ff9d;
                    background:rgba(0,255,157,0.1);border-radius:50%;
                    width:24px;height:24px;display:flex;align-items:center;
                    justify-content:center;flex-shrink:0;'>{num}</div>
                <div>
                    <div style='font-size:12px;font-weight:600;color:{T["text_metric"]};'>{title}</div>
                    <div style='font-size:10px;color:{T["text_faint"]};'>{hint}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:rgba(255,0,85,0.05);border:1px solid rgba(255,0,85,0.2);
        border-radius:14px;padding:20px 24px;'>
        <div style='font-size:12px;font-weight:700;color:#ff0055;
            letter-spacing:2px;margin-bottom:8px;'>⚠️ IMPORTANT DISCLAIMER</div>
        <div style='font-size:12px;color:{T["text_muted"]};line-height:1.8;'>
            TradeVision AI is built for <strong>educational and research purposes only</strong>.
            All predictions, signals, and analysis are generated by machine learning models
            and <strong>should NOT be considered as financial advice</strong>.<br><br>
            Stock markets are inherently unpredictable. Always do your own research (DYOR)
            and consult a SEBI-registered financial advisor before making any investment decisions.
            Past performance does not guarantee future results.
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class='footer'>
    <div style='margin-bottom:16px;'>
        <span class='info-badge'>v4.0.0</span>
        <span class='info-badge'>{active_model} × 3</span>
        <span class='info-badge'>ADAM</span>
        <span class='info-badge'>{"🌙 DARK" if DARK else "☀️ LIGHT"}</span>
    </div>
    <div class='footer-links'><a href='#'>Docs</a><a href='#'>Architecture</a><a href='#'>Disclaimer</a></div>
    <div style='margin-top:16px;opacity:0.5;font-size:11px;'>TradeVision AI © 2026 · Streamlit + TensorFlow · Educational Use Only</div>
</div>""", unsafe_allow_html=True)
st.divider()
st.markdown("""<div class='disclaimer'><strong style='letter-spacing:2px;display:block;margin-bottom:4px;'>⚠ DISCLAIMER</strong>
For educational and research purposes only. Not financial advice. Always consult a qualified advisor.</div>""", unsafe_allow_html=True)