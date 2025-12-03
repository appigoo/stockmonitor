import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import plotly.graph_objects as go

st.set_page_config(page_title="多股票趨勢監控", layout="wide")

# ----------------------------
# session_state 初始化
# ----------------------------
if "sent_alerts" not in st.session_state:
    st.session_state["sent_alerts"] = set()

# ============================
# Telegram 推播（更穩定）
# ============================
def send_telegram(text: str):
    """
    傳送 Telegram 訊息，若已發送過就跳過（使用 st.session_state 持久化）。
    """
    if text in st.session_state["sent_alerts"]:
        return
    try:
        token = st.secrets.get("telegram_token")
        chat_id = st.secrets.get("telegram_chat_id")
        if not token or not chat_id:
            st.warning("Telegram token 或 chat_id 未設定於 st.secrets，跳過發送。")
            return
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        if resp.status_code == 200:
            st.session_state["sent_alerts"].add(text)
        else:
            st.error(f"Telegram 發送失敗：{resp.status_code} {resp.text}")
    except Exception as e:
        st.error(f"送出 Telegram 時發生例外：{e}")

# ============================
# 完全安全的 True Range（最終版）
# ============================
def calculate_tr(df: pd.DataFrame) -> pd.Series:
    """
    計算 TR（True Range），回傳一個與 df 相同 index 的 pd.Series
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    # 前一根的 close，第一筆用本筆代替避免 NaN
    prev_close = close.shift(1).fillna(close.iloc[0])

    tr0 = (high - low).abs()
    tr1 = (high - prev_close).abs()
    tr2 = (low - prev_close).abs()

    # 確保使用 numpy array 計算，回傳 pd.Series
    tr_vals = np.maximum(tr0.values, np.maximum(tr1.values, tr2.values))
    return pd.Series(tr_vals, index=df.index, name='TR')

# ============================
# SuperTrend（使用上面的安全 TR）
# ============================
def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    # 若沒有 High/Low/Close 則直接回傳
    for col in ['High', 'Low', 'Close']:
        if col not in df.columns:
            raise ValueError(f"DataFrame 必須包含欄位：{col}")

    hl2 = (df['High'] + df['Low']) / 2
    atr = calculate_tr(df).rolling(period, min_periods=1).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    df['UpperBand'] = upper
    df['LowerBand'] = lower
    df['SuperTrend'] = np.nan
    df['ST_Line'] = np.nan

    # 初始 trend：以第一可用點為基準（預設 1）
    trend = 1
    # 從 period 開始計算（也可從 1，但保留 period）
    for i in range(len(df)):
        if i == 0:
            df.iat[i, df.columns.get_loc('SuperTrend')] = np.nan
            df.iat[i, df.columns.get_loc('ST_Line')] = np.nan
            continue

        close = df['Close'].iat[i]
        prev_upper = df['UpperBand'].iat[i - 1]
        prev_lower = df['LowerBand'].iat[i - 1]
        cur_upper = df['UpperBand'].iat[i]
        cur_lower = df['LowerBand'].iat[i]

        if np.isnan(prev_upper) or np.isnan(prev_lower):
            # 尚無上下帶，維持 NaN
            df.iat[i, df.columns.get_loc('SuperTrend')] = np.nan
            df.iat[i, df.columns.get_loc('ST_Line')] = np.nan
            continue

        if close > prev_upper:
            trend = 1
        elif close < prev_lower:
            trend = -1
        else:
            # 未突破前日上下帶：延續或反轉的具體條件
            if trend == 1 and close < cur_lower:
                trend = -1
            elif trend == -1 and close > cur_upper:
                trend = 1
            # 否則維持 trend

        df.iat[i, df.columns.get_loc('SuperTrend')] = trend
        # ST_Line 為 trend 為 1 時的 lower，trend 為 -1 時的 upper
        df.iat[i, df.columns.get_loc('ST_Line')] = (cur_lower if trend == 1 else cur_upper)

    return df

# ============================
# 其他指標（全部最安全寫法）
# ============================
def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    以日為單位計算 VWAP（若 index 不是 DatetimeIndex 嘗試轉換）
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("VWAP 需要可以轉換為 DatetimeIndex 的 index")

    pv = df['Close'] * df['Volume']
    # 以 index.date 分組累計
    group_keys = df.index.date
    cum_pv = pv.groupby(group_keys).cumsum()
    cum_vol = df['Volume'].groupby(group_keys).cumsum()
    df['VWAP'] = cum_pv / cum_vol
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    更穩定的 RSI 計算，避免除以零造成 nan/inf
    """
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=period, min_periods=1).mean()

    # 防止除以 0
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # 若 loss 為 0，表示沒有下跌，RSI 應該為 100；若 gain 為 0 且 loss 為 0，視為 50
    rsi = rsi.fillna(100).where(~(gain == 0) | ~(loss == 0), 50)
    df['RSI'] = rsi
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['Close']
    df['EMA12'] = close.ewm(span=12, adjust=False).mean()
    df['EMA26'] = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    這裡我們使用 ATR 的簡易處理當作 ADX 的 proxy（若你需要完整 ADX 邏輯可以再擴展）。
    """
    df = df.copy()
    atr = calculate_tr(df).rolling(period, min_periods=1).mean()
    df['ADX'] = atr.rolling(period, min_periods=1).mean().fillna(0)
    return df

# ============================
# 美觀 10 根 K 線圖（繪圖前做欄位檢查）
# ============================
def plot_candlestick(df: pd.DataFrame, symbol: str):
    df_plot = df.tail(10).copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        name="K線", increasing_line_color='#00ff88', decreasing_line_color='#ff4444'
    ))

    # SuperTrend 線（如果存在且不是全 NaN）
    if 'ST_Line' in df_plot.columns and df_plot['ST_Line'].notna().any():
        last_st = df_plot['SuperTrend'].dropna()
        if last_st.empty:
            color = 'gray'
        else:
            color = 'green' if last_st.iloc[-1] == 1 else 'red'
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['ST_Line'],
            mode='lines', name='SuperTrend',
            line=dict(width=3, color=color)
        ))

    # VWAP / MA20 / MA50 若存在再畫
    if 'VWAP' in df_plot.columns and df_plot['VWAP'].notna().any():
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VWAP'], mode='lines', name='VWAP',
                                 line=dict(dash='dot')))
    if 'MA20' in df_plot.columns and df_plot['MA20'].notna().any():
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], mode='lines', name='MA20'))
    if 'MA50' in df_plot.columns and df_plot['MA50'].notna().any():
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], mode='lines', name='MA50'))

    fig.update_layout(
        title=f"{symbol} 最近 10 根 K 線",
        template="plotly_dark", height=600,
        hovermode="x unified", xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================
# 主程式
# ============================
st.title("多股票趨勢監控 + 專業K線圖")

symbols = st.text_input("輸入股票（逗號分隔）", "AAPL,TSLA,NVDA,BTC-USD,0050.TW")
# 允許使用者輸入混大小寫，但我們要保留輸入的字元（像 0050.TW 則維持）
symbols_list = [s.strip() for s in symbols.split(",") if s.strip()]

timeframe = st.selectbox("時間框", ["1m","5m","15m","30m","1h","1d"], index=5)
period = st.selectbox("資料期間", ["1d","5d","30d","60d","1y"], index=2)

refresh_map = {"不刷新":0, "30秒":30, "1分鐘":60, "3分鐘":180, "5分鐘":300}
refresh_choice = st.selectbox("自動刷新", list(refresh_map.keys()), index=0)
refresh_sec = refresh_map[refresh_choice]

# 自動刷新（簡單實作） - 注意 time.sleep 會阻塞執行緒
if refresh_sec > 0:
    st.info(f"每 {refresh_sec} 秒自動刷新（若你的 Streamlit 版本不支援自動重執行，請手動重新整理）")
    try:
        time.sleep(refresh_sec)
        # st.rerun() 是不存在的；使用 experimental_rerun（部分版本可能也沒有）
        try:
            st.experimental_rerun()
        except Exception:
            # 若無法呼叫 experimental_rerun，顯示訊息，讓使用者手動刷新
            st.warning("自動重載在此 Streamlit 版本不可用，請手動刷新頁面。")
    except Exception as e:
        st.error(f"自動刷新時發生錯誤：{e}")

# 逐個 Symbol 取得資料、計算指標並繪圖
for symbol in symbols_list:
    st.subheader(f"{symbol}")

    try:
        df = yf.download(symbol, period=period, interval=timeframe, progress=False)
        if df is None or df.empty:
            st.warning(f"{symbol} 資料抓取失敗或為空。")
            continue

        # 若資料列數太少，提醒
        if len(df) < 20:
            st.warning(f"{symbol} 資料筆數不足（{len(df)}筆），可能無法正確計算指標。")

        # 確保基本欄位存在
        expected_cols = ['Open','High','Low','Close','Volume']
        for col in expected_cols:
            if col not in df.columns:
                st.error(f"{symbol} 缺少必要欄位：{col}")
                raise RuntimeError(f"Missing column {col}")

        # 清理與指標計算
        df = df.dropna(how='all')
        df = add_macd(df)
        df = add_rsi(df)
        df = add_adx(df)
        df = add_vwap(df)
        df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
        df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        df = supertrend(df)

        # 趨勢判斷（有些欄位可能為 NaN，因此需防護）
        try:
            macd_val = df["MACD"].iloc[-1]
            signal_val = df["Signal"].iloc[-1]
            direction = "上升" if macd_val > signal_val else "下降"
        except Exception:
            direction = "未知"

        try:
            adx_val = df["ADX"].iloc[-1]
            strength = "強" if adx_val > df["ADX"].mean() else "弱"
        except Exception:
            strength = "未知"

        st.markdown(f"**趨勢**：{direction}　**強度**：{strength}")

        # 如果想要發 Telegram 通知（範例條件：MACD 黃金交叉且 ADX 強）
        try:
            if direction == "上升" and strength == "強":
                send_telegram(f"{symbol} 偵測到上升趨勢 (MACD > Signal 且 ADX 較強)。")
        except Exception as e:
            st.warning(f"檢查是否要發送 Telegram 時發生錯誤：{e}")

        plot_candlestick(df, symbol)

    except Exception as e:
        st.error(f"{symbol} 處理發生錯誤：{e}")

st.success("全部完成！")
