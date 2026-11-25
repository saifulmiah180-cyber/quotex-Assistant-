
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Quotex AI Signal Pro",
    page_icon="📈",
    layout="wide"
)

st.title("QUOTEX AI SIGNAL PRO")
st.caption("Historical market analysis | Educational use only")

st.warning(
    "এই অ্যাপ CSV ডেটা বিশ্লেষণ করে। "
    "এটি Quotex-এর লাইভ বা OTC ডেটা নিজে সংগ্রহ করে না। "
    "কোনো সিগন্যাল লাভের নিশ্চয়তা দেয় না।"
)


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)

    return rsi


def analyze_data(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["open", "high", "low", "close"]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "CSV-তে প্রয়োজনীয় কলাম নেই: " + ", ".join(missing)
        )

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required)

    time_col = next(
        (c for c in ["timestamp", "time"] if c in df.columns),
        None
    )

    if time_col:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        if parsed.notna().any():
            df["_sort_time"] = parsed
            df = df.sort_values("_sort_time")

    invalid = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )

    removed = int(invalid.sum())
    df = df.loc[~invalid].reset_index(drop=True)

    if len(df) < 60:
        raise ValueError("কমপক্ষে ৬০টি সঠিক ক্যান্ডেল প্রয়োজন।")

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["rsi"] = calculate_rsi(df["close"])

    fast = df["close"].ewm(span=12, adjust=False).mean()
    slow = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = fast - slow
    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["bull_engulf"] = (
        (df["close"] > df["open"]) &
        (df["close"].shift(1) < df["open"].shift(1)) &
        (df["close"] >= df["open"].shift(1)) &
        (df["open"] <= df["close"].shift(1))
    )

    df["bear_engulf"] = (
        (df["close"] < df["open"]) &
        (df["close"].shift(1) > df["open"].shift(1)) &
        (df["close"] <= df["open"].shift(1)) &
        (df["open"] >= df["close"].shift(1))
    )

    previous_high = df["high"].shift(1).rolling(20).max()
    previous_low = df["low"].shift(1).rolling(20).min()

    df["support"] = previous_low
    df["resistance"] = previous_high

    df["bull_sweep"] = (
        (df["low"] < previous_low) &
        (df["close"] > previous_low)
    )

    df["bear_sweep"] = (
        (df["high"] > previous_high) &
        (df["close"] < previous_high)
    )

    return df, removed


def make_signals(df):
    d = df.copy()

    bull = np.zeros(len(d), dtype=int)
    bear = np.zeros(len(d), dtype=int)

    bull += (d["close"] > d["ema21"]).fillna(False).astype(int)
    bull += (d["ema9"] > d["ema21"]).fillna(False).astype(int)
    bull += (d["macd"] > d["macd_signal"]).fillna(False).astype(int)
    bull += (
        (d["rsi"] >= 50) & (d["rsi"] <= 70)
    ).fillna(False).astype(int)
    bull += d["bull_engulf"].fillna(False).astype(int)
    bull += d["bull_sweep"].fillna(False).astype(int)

    bear += (d["close"] < d["ema21"]).fillna(False).astype(int)
    bear += (d["ema9"] < d["ema21"]).fillna(False).astype(int)
    bear += (d["macd"] < d["macd_signal"]).fillna(False).astype(int)
    bear += (
        (d["rsi"] >= 30) & (d["rsi"] <= 50)
    ).fillna(False).astype(int)
    bear += d["bear_engulf"].fillna(False).astype(int)
    bear += d["bear_sweep"].fillna(False).astype(int)

    d["bull_score"] = bull
    d["bear_score"] = bear

    d["signal"] = np.where(
        (bull >= 4) & (bull > bear),
        "BUY",
        np.where(
            (bear >= 4) & (bear > bull),
            "SELL",
            "WAIT"
        )
    )

    d["confidence_score"] = np.where(
        d["signal"] == "BUY",
        bull,
        np.where(
            d["signal"] == "SELL",
            bear,
            np.maximum(bull, bear)
        )
    )

    return d


def run_backtest(df, limit):
    x = df.tail(limit).copy().reset_index(drop=True)

    x["next_open"] = x["open"].shift(-1)
    x["next_close"] = x["close"].shift(-1)

    x["outcome"] = np.where(
        x["next_close"] > x["next_open"],
        "GREEN",
        np.where(
            x["next_close"] < x["next_open"],
            "RED",
            "DOJI"
        )
    )

    trades = x[
        x["signal"].isin(["BUY", "SELL"]) &
        x["next_close"].notna()
    ].copy()

    trades["win"] = (
        ((trades["signal"] == "BUY") &
         (trades["outcome"] == "GREEN")) |
        ((trades["signal"] == "SELL") &
         (trades["outcome"] == "RED"))
    )

    decisive = trades[trades["outcome"].isin(["GREEN", "RED"])]

    return x, trades, decisive


st.subheader("CSV Data Upload")

uploaded = st.file_uploader(
    "ঐতিহাসিক ক্যান্ডেলের CSV আপলোড করুন",
    type=["csv"]
)

st.caption("প্রয়োজনীয় কলাম: open, high, low, close")

market = st.selectbox(
    "Market Type",
    [
        "Forex Historical Data",
        "Quotex OTC Historical Data",
        "Other OHLC Data"
    ]
)

timeframe = st.selectbox(
    "Timeframe",
    ["M1", "M2", "M5", "M15"]
)

limit = st.select_slider(
    "Backtest Candles",
    options=[100, 500, 2000, 3000, 10000, 20000, 30000, 50000],
    value=10000
)


if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        data, removed = analyze_data(raw)
        data = make_signals(data)

        st.success(f"লোড হয়েছে: {len(data):,}টি ক্যান্ডেল")

        if removed:
            st.warning(f"{removed}টি ভুল ক্যান্ডেল বাদ দেওয়া হয়েছে।")

        if market == "Quotex OTC Historical Data":
            st.info(
                "OTC ফলাফল যাচাই করতে আসল Quotex OTC ডেটা ব্যবহার করুন।"
            )

        latest = data.iloc[-1]

        st.divider()
        st.subheader("LATEST SIGNAL")

        if latest["signal"] == "BUY":
            st.success("🟢 BUY / CALL")
        elif latest["signal"] == "SELL":
            st.error("🔴 SELL / PUT")
        else:
            st.warning("🟡 WAIT")

        a, b, c, d = st.columns(4)

        a.metric("Price", f"{latest['close']:.6f}")
        b.metric(
            "RSI",
            f"{latest['rsi']:.2f}" if pd.notna(latest["rsi"]) else "N/A"
        )
        c.metric(
            "Bull Score",
            f"{int(latest['bull_score'])}/6"
        )
        d.metric(
            "Bear Score",
            f"{int(latest['bear_score'])}/6"
        )

        st.caption(
            "স্কোর হলো নিয়মভিত্তিক ইন্ডিকেটর গণনা, "
            "এটি জয়ের সম্ভাবনা নয়।"
        )

        st.subheader("INDICATOR ANALYSIS")

        x1, x2, x3 = st.columns(3)
        x1.metric("EMA 9", f"{latest['ema9']:.6f}")
        x2.metric("EMA 21", f"{latest['ema21']:.6f}")
        x3.metric("EMA 50", f"{latest['ema50']:.6f}")

        y1, y2, y3 = st.columns(3)
        y1.metric("MACD", f"{latest['macd']:.6f}")
        y2.metric("MACD Signal", f"{latest['macd_signal']:.6f}")
        y3.metric("RSI", f"{latest['rsi']:.2f}")

        st.subheader("SUPPORT & RESISTANCE")

        s1, s2 = st.columns(2)

        s1.metric(
            "Support",
            f"{latest['support']:.6f}"
            if pd.notna(latest["support"]) else "N/A"
        )

        s2.metric(
            "Resistance",
            f"{latest['resistance']:.6f}"
            if pd.notna(latest["resistance"]) else "N/A"
        )

        tested, trades, decisive = run_backtest(data, limit)

        st.divider()
        st.subheader("BACKTEST RESULTS")

        wins = int(decisive["win"].sum())
        losses = int((~decisive["win"]).sum())
        dojis = int((trades["outcome"] == "DOJI").sum())

        total = wins + losses

        rate = 100 * wins / total if total else 0

        a, b, c, d = st.columns(4)

        a.metric("Signals", f"{len(trades):,}")
        b.metric("Wins", f"{wins:,}")
        c.metric("Losses", f"{losses:,}")
        d.metric("Win Rate", f"{rate:.2f}%" if total else "N/A")

        st.metric("Doji", f"{dojis:,}")

        st.caption(
            "পরবর্তী ক্যান্ডেলের GREEN/RED দিক দিয়ে ফলাফল মাপা হয়েছে। "
            "Payout, spread, latency ও execution slippage অন্তর্ভুক্ত নয়।"
        )

        if len(decisive):
            st.line_chart(
                decisive["win"].astype(int)
                .rolling(50, min_periods=1)
                .mean() * 100
            )

        st.subheader("RECENT SIGNALS")

        view_columns = [
            c for c in [
                "timestamp", "time", "open", "high", "low",
                "close", "rsi", "bull_score", "bear_score", "signal"
            ] if c in tested.columns
        ]

        st.dataframe(
            tested[view_columns].tail(30),
            use_container_width=True
        )

        st.subheader("TRADE DETAILS")

        if len(trades):
            trade_columns = [
                c for c in [
                    "timestamp", "time", "signal",
                    "outcome", "win", "confidence_score"
                ] if c in trades.columns
            ]

            st.dataframe(
                trades[trade_columns].tail(100),
                use_container_width=True
            )

            st.download_button(
                "Download Signals CSV",
                data=tested.to_csv(index=False).encode("utf-8"),
                file_name="quotex_ai_signals.csv",
                mime="text/csv"
            )
        else:
            st.info("এই ডেটায় কোনো পরীক্ষাযোগ্য BUY বা SELL সিগন্যাল পাওয়া যায়নি।")

    except Exception as error:
        st.error(f"সমস্যা হয়েছে: {error}")

else:
    st.info("বিশ্লেষণ শুরু করতে প্রথমে CSV ফাইল আপলোড করুন।")


st.divider()
st.subheader("TRADING ASSISTANT")

question = st.text_input(
    "প্রশ্ন লিখুন",
    placeholder="যেমন: RSI 70 হলে কী বোঝায়?"
)

if question:
    q = question.lower()

    if "rsi" in q:
        answer = (
            "RSI একটি Momentum Indicator। সাধারণত 70-এর ওপরে "
            "Overbought এবং 30-এর নিচে Oversold বলা হয়।"
        )
    elif "ema" in q:
        answer = (
            "EMA Trend বুঝতে সাহায্য করে। EMA 9, EMA 21-এর "
            "ওপরে থাকলে স্বল্পমেয়াদি bullish trend-এর ইঙ্গিত হতে পারে।"
        )
    elif "macd" in q:
        answer = (
            "MACD Momentum ও Trend বুঝতে সাহায্য করে। "
            "MACD Signal Line-এর ওপরে থাকলে bullish momentum-এর ইঙ্গিত হতে পারে।"
        )
    elif "otc" in q:
        answer = (
            "Quotex OTC যাচাই করতে নির্দিষ্ট OTC historical বা live data প্রয়োজন।"
        )
    else:
        answer = (
            "এই Assistant বর্তমানে সীমিত Rule-Based। "
            "পূর্ণ AI সংযোগের জন্য আলাদা AI Model/API প্রয়োজন।"
        )

    st.write(answer)


st.caption(
    "শিক্ষামূলক প্রোটোটাইপ। বিনিয়োগ পরামর্শ নয়। "
    "Quotex-এর পাসওয়ার্ড বা লগইন তথ্য এখানে দেবেন না।"
)
