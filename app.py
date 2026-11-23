import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Quotex AI Signal Pro", page_icon="📈", layout="wide")
st.title("QUOTEX AI SIGNAL PRO")
st.caption("Read-only research dashboard • No automatic trading • Historical results are not guarantees")

st.warning(
    "ডেটা সতর্কতা: Quotex OTC-এর আসল ক্যান্ডেল ডেটা ছাড়া OTC ফলাফল যাচাই করা যায় না। "
    "এই অ্যাপ নিজে থেকে Quotex-এ লগইন করে না এবং কোনো লাইভ/OTC ডেটা বানিয়ে দেখায় না।"
)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def enrich(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError("CSV-তে open, high, low, close কলাম থাকতে হবে।")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["ema9"] = df.close.ewm(span=9, adjust=False).mean()
    df["ema21"] = df.close.ewm(span=21, adjust=False).mean()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    df["rsi"] = rsi(df.close)
    fast = df.close.ewm(span=12, adjust=False).mean()
    slow = df.close.ewm(span=26, adjust=False).mean()
    df["macd"] = fast - slow
    df["macd_signal"] = df.macd.ewm(span=9, adjust=False).mean()
    df["body"] = (df.close - df.open).abs()
    df["range"] = (df.high - df.low).replace(0, np.nan)
    df["bull_engulf"] = (
        (df.close > df.open) & (df.close.shift(1) < df.open.shift(1)) &
        (df.close >= df.open.shift(1)) & (df.open <= df.close.shift(1))
    )
    df["bear_engulf"] = (
        (df.close < df.open) & (df.close.shift(1) > df.open.shift(1)) &
        (df.close <= df.open.shift(1)) & (df.open >= df.close.shift(1))
    )
    # Causal rolling levels: current candle is excluded from its own reference window.
    prev_high = df.high.shift(1).rolling(20).max()
    prev_low = df.low.shift(1).rolling(20).min()
    df["bull_sweep"] = (df.low < prev_low) & (df.close > prev_low)
    df["bear_sweep"] = (df.high > prev_high) & (df.close < prev_high)
    return df

def signals(df):
    d = df.copy()
    score = np.zeros(len(d), dtype=int)
    score += (d.close > d.ema21).fillna(False).astype(int)
    score += (d.ema9 > d.ema21).fillna(False).astype(int)
    score += (d.macd > d.macd_signal).fillna(False).astype(int)
    score += ((d.rsi >= 50) & (d.rsi <= 70)).fillna(False).astype(int)
    score += d.bull_engulf.fillna(False).astype(int)
    score += d.bull_sweep.fillna(False).astype(int)
    bear = np.zeros(len(d), dtype=int)
    bear += (d.close < d.ema21).fillna(False).astype(int)
    bear += (d.ema9 < d.ema21).fillna(False).astype(int)
    bear += (d.macd < d.macd_signal).fillna(False).astype(int)
    bear += ((d.rsi <= 50) & (d.rsi >= 30)).fillna(False).astype(int)
    bear += d.bear_engulf.fillna(False).astype(int)
    bear += d.bear_sweep.fillna(False).astype(int)
    d["bull_score"] = score
    d["bear_score"] = bear
    d["signal"] = np.where(score >= 4, "CALL", np.where(bear >= 4, "PUT", "WAIT"))
    d["confidence_score"] = np.where(d.signal == "CALL", score,
                              np.where(d.signal == "PUT", bear, np.maximum(score, bear)))
    return d

def backtest(d, max_rows, expiry):
    # Signal on candle i is evaluated against the close direction of candle i+1.
    x = d.tail(max_rows).copy().reset_index(drop=True)
    x["next_open"] = x.open.shift(-1)
    x["next_close"] = x.close.shift(-1)
    x["outcome"] = np.where(x.next_close > x.next_open, "GREEN",
                     np.where(x.next_close < x.next_open, "RED", "DOJI"))
    valid = x.signal.isin(["CALL", "PUT"]) & x.next_close.notna()
    trades = x.loc[valid].copy()
    trades["win"] = ((trades.signal == "CALL") & (trades.outcome == "GREEN")) | (
        (trades.signal == "PUT") & (trades.outcome == "RED"))
    return x, trades

uploaded = st.file_uploader("ঐতিহাসিক ক্যান্ডেলের CSV আপলোড করুন", type=["csv"])
st.caption("প্রয়োজনীয় কলাম: open, high, low, close। ঐচ্ছিক: timestamp বা time।")
with st.expander("CSV ফরম্যাটের উদাহরণ"):
    st.code("timestamp,open,high,low,close\n2026-01-01 10:00,1.1000,1.1010,1.0990,1.1005\n2026-01-01 10:01,1.1005,1.1020,1.1000,1.1015", language="csv")

c1, c2 = st.columns(2)
with c1:
    market = st.selectbox("Market type", ["LIVE historical feed", "Quotex OTC historical feed", "Other OHLC data"])
with c2:
    timeframe = st.selectbox("Timeframe", ["M1", "M2", "M5", "M15"])

max_rows = st.select_slider("সর্বোচ্চ ব্যাকটেস্ট ক্যান্ডেল", options=[100, 500, 2000, 3000, 10000, 20000, 30000, 50000], value=10000)
expiry = st.selectbox("Evaluation horizon", ["Next candle (1 candle)"], index=0)

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        data = signals(enrich(raw))
        st.success(f"ডেটা লোড হয়েছে: {len(data):,}টি পরিষ্কার ক্যান্ডেল")
        if market == "Quotex OTC historical feed":
            st.info("OTC নির্বাচন করা হয়েছে। নিশ্চিত করুন CSV-টি Quotex OTC থেকেই নেওয়া; অন্য ব্রোকারের ডেটা OTC-এর বিকল্প নয়।")
        tested, trades = backtest(data, max_rows, expiry)
        a, b, c, e = st.columns(4)
        wins = int(trades.win.sum()) if len(trades) else 0
        losses = int((~trades.win).sum()) if len(trades) else 0
        rate = 100*wins/len(trades) if len(trades) else 0
        a.metric("Signals", f"{len(trades):,}")
        b.metric("Wins", f"{wins:,}")
        c.metric("Losses", f"{losses:,}")
        e.metric("Win rate", f"{rate:.2f}%" if len(trades) else "N/A")
        st.caption("এটি পরবর্তী ক্যান্ডেলের GREEN/RED দিকের hit rate; payout, tie rules, latency ও execution slippage এতে অন্তর্ভুক্ত নয়।")
        st.subheader("সাম্প্রতিক সিগন্যাল")
        view_cols = [c for c in ["timestamp", "time", "open", "high", "low", "close", "rsi", "bull_score", "bear_score", "signal"] if c in tested.columns]
        st.dataframe(tested[view_cols].tail(30), use_container_width=True)
        st.subheader("Backtest outcomes")
        if len(trades):
            st.bar_chart(trades["win"].astype(int).rolling(50, min_periods=1).mean())
            st.dataframe(trades[[c for c in ["timestamp","time","signal","outcome","win","confidence_score"] if c in trades.columns]].tail(100),
                         use_container_width=True)
            st.download_button("Download tested signals CSV", tested.to_csv(index=False).encode("utf-8"),
                               "quotex_ai_tested_signals.csv", "text/csv")
        else:
            st.warning("এই ডেটা/সেটিংসে কোনো CALL বা PUT সিগন্যাল পাওয়া যায়নি।")
    except Exception as ex:
        st.error(f"ডেটা পড়তে সমস্যা: {ex}")
else:
    st.info("ব্যাকটেস্ট শুরু করতে CSV আপলোড করুন। ডেটা ছাড়া ১০,০০০–৫০,০০০ ক্যান্ডেলের পরীক্ষা বা নির্ভরযোগ্য Win Rate দাবি করা যাবে না।")

st.divider()
st.subheader("AI Trading Assistant — প্রাথমিক সহায়ক")
question = st.text_input("প্রশ্ন লিখুন", placeholder="যেমন: RSI 70 হলে কী বোঝায়?")
if question:
    q = question.lower()
    if "rsi" in q:
        answer = "RSI হলো momentum indicator। সাধারণভাবে 70-এর ওপরে overbought এবং 30-এর নিচে oversold বলা হয়; একে একা CALL/PUT সিদ্ধান্ত হিসেবে ব্যবহার করা ঠিক নয়।"
    elif "otc" in q:
        answer = "OTC-এর জন্য সেই নির্দিষ্ট OTC feed-এর ঐতিহাসিক ও লাইভ ডেটা দরকার। সাধারণ Forex ডেটা দিয়ে Quotex OTC যাচাই করা যায় না।"
    elif "backtest" in q or "ব্যাকটেস্ট" in q:
        answer = "ব্যাকটেস্টে signal candle-এর পরের candle দিয়ে ফল মাপা হয়। ভবিষ্যৎ ডেটা signal তৈরিতে ব্যবহার করা যাবে না।"
    else:
        answer = "এই সংস্করণের সহায়কটি সীমিত rule-based। পূর্ণ AI chat agent যুক্ত করতে আলাদা AI model/API সংযোগ ও নিরাপদ configuration প্রয়োজন।"
    st.write(answer)

st.caption("গুরুত্বপূর্ণ: এটি গবেষণা/শিক্ষামূলক প্রোটোটাইপ, বিনিয়োগ পরামর্শ নয়। Quotex-এর কোনো account credential এখানে দেবেন না।")
