import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="Stock-Sentry AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
<style>
    .stApp { background-color: #121212; }
    
    /* Blue Metric Cards */
    .metric-container {
        background-color: #001F3F;
        border: 1px solid #003366;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label { font-size: 14px; font-weight: 500; color: #E0E0E0; margin-bottom: 5px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #FFFFFF; }

    /* Table & Signals */
    .btn-sell { background-color: #FF4B4B; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold; }
    .btn-buy { background-color: #00CC96; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold; }
    .btn-hold { background-color: #FF8C00; color: white; padding: 5px 15px; border-radius: 4px; font-weight: bold; }
    
    .text-red { color: #FF4B4B; font-weight: bold; }
    .text-green { color: #00CC96; font-weight: bold; }
    
    h1, h2, h3 { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Load Resources (Safe Loading)
# ==========================================
@st.cache_resource
def load_assets():
    if not os.path.exists('stock_sentry_best_model.h5'): return None, None, None
    try:
        # compile=False fixes the 'mse' error
        model = tf.keras.models.load_model('stock_sentry_best_model.h5', compile=False)
        scaler_X = joblib.load('scaler_features.gz')
        scaler_y = joblib.load('scaler_target.gz')
        return model, scaler_X, scaler_y
    except:
        return None, None, None

model, scaler_X, scaler_y = load_assets()

# ==========================================
# 3. Sidebar
# ==========================================
with st.sidebar:
    st.header("Model Controls")
    st.write("Select Prediction Engine")
    st.radio("Engine", ["Best Model (Recommended)", "SimpleRNN", "LSTM"], index=0, label_visibility="collapsed")
    st.success("✅ **Active Model: LSTM**")
    
    st.markdown("---")
    st.subheader("Forecast Settings")
    # This variable 'horizon' matches your snippet requirements
    horizon = st.selectbox("Select Horizon", ["1 Day", "5 Days", "10 Days"])
    
    run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

# ==========================================
# 4. Data Processing
# ==========================================
try:
    df = pd.read_csv('TSLA.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    last_close = df['Adj Close'].iloc[-1]
except:
    st.error("TSLA.csv not found.")
    st.stop()

def prepare_data(data):
    df_new = data.copy()
    delta = df_new['Adj Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    df_new['RSI'] = 100 - (100 / (1 + (gain/loss)))
    df_new['MACD'] = df_new['Adj Close'].ewm(span=12).mean() - df_new['Adj Close'].ewm(span=26).mean()
    return df_new.dropna()

df_feat = prepare_data(df)

# ==========================================
# 5. Dashboard Layout
# ==========================================

# -- Metric Cards --
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-container"><div class="metric-label">Latest Close</div><div class="metric-value">${last_close:,.2f}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-container"><div class="metric-label">Data Records</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-container"><div class="metric-label">Target Horizon</div><div class="metric-value">{horizon}</div></div>', unsafe_allow_html=True)

st.write("")

# -- Prediction Logic (Your Snippet Integration) --
if run_btn:
    if model is None:
        st.error("Model files missing. Train model first.")
    else:
        try:
            # --- START OF YOUR LOGIC ---
            if len(df_feat) < 60:
                st.error("Not enough data to generate forecast (need at least 60 days).")
            else:
                # Get last 60 days of features
                features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'RSI', 'MACD']
                last_60 = df_feat[features].iloc[-60:].values
                
                # Scale Input
                last_60_scaled = scaler_X.transform(last_60)
                
                # Reshape for LSTM (1 sample, 60 time steps, 7 features)
                input_tensor = np.array([last_60_scaled]) 
                
                # Predict
                pred_scaled = model.predict(input_tensor)
                
                # Inverse Transform to get Dollar Amount
                pred_dollar = scaler_y.inverse_transform(pred_scaled)[0] # Shape: [1D, 5D, 10D]
                
                # Map output index to selected horizon
                idx_map = {"1 Day": 0, "5 Days": 1, "10 Days": 2}
                final_price = pred_dollar[idx_map[horizon]]
                
                # Your requested Success Message
                st.success(f"🚀 Forecasted Price for **{horizon}**: **${final_price:.2f}**")
                # --- END OF YOUR LOGIC ---

                # --- VISUALIZATION & TABLE (Added for UI) ---
                st.markdown("---")
                
                # 1. Forecast Summary
                change_pct = ((final_price - last_close) / last_close) * 100
                color_cls = "text-green" if change_pct > 0 else "text-red"
                signal_badge = '<div class="btn-buy">BUY</div>' if change_pct > 0 else '<div class="btn-sell">SELL</div>'
                
                st.subheader("Forecast Summary")
                k1, k2, k3 = st.columns(3)
                k1.metric("Predicted Price", f"${final_price:.2f}")
                k2.markdown(f"**Change %**: <span class='{color_cls}'>{change_pct:.2f}%</span>", unsafe_allow_html=True)
                k3.markdown(f"**Signal**: {signal_badge}", unsafe_allow_html=True)
                
                # 2. Day-wise Outlook Table (Interpolated for UX)
                st.subheader("Day-wise Price Outlook")
                
                # Interpolate days based on horizon selection
                days_to_show = 1 if horizon == "1 Day" else 5 if horizon == "5 Days" else 10
                
                # Smart Interpolation: Day 1 is fixed. If Horizon > 1, target is Day 5 or 10.
                target_val = pred_dollar[1] if days_to_show == 5 else pred_dollar[2] if days_to_show == 10 else pred_dollar[0]
                
                if days_to_show == 1:
                    prices = [pred_dollar[0]]
                else:
                    # Linearly interpolate between Day 1 prediction and Target prediction
                    prices = np.linspace(pred_dollar[0], target_val, days_to_show)

                # Create DataFrame for Table
                table_data = []
                prev = last_close
                for i, p in enumerate(prices):
                    d_chg = ((p - prev)/prev)*100
                    sig = "📈 Buy" if d_chg > 0 else "📉 Sell"
                    table_data.append({
                        "Day": f"Day {i+1}",
                        "Close Forecast": f"${p:.2f}",
                        "Change %": f"{d_chg:.2f}%",
                        "Signal": sig
                    })
                    prev = p # Update prev for next day comparison if desired, or keep relative to start
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                # 3. Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index[-90:], y=df['Adj Close'].iloc[-90:], name="History", line=dict(color='gray')))
                
                # Add Forecast Point
                future_date = df.index[-1] + pd.Timedelta(days=days_to_show)
                fig.add_trace(go.Scatter(x=[df.index[-1], future_date], y=[last_close, final_price], 
                                         mode='lines+markers', name="Forecast", 
                                         line=dict(color='#00F2C4', width=3, dash='dash')))
                
                fig.update_layout(template="plotly_dark", height=400, title="Price Projection")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Application Error: {e}")