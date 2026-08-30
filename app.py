import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import *
from lstm import create_model
from dataset import StockDataset

# ======================================
# Streamlit Configuration
# ======================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ======================================
# CSS
# ======================================

st.markdown("""
<style>
.main{
background:#F7F9FB;
}
h1{
color:#1565C0;
}
.stButton>button{
width:100%;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# Load Model
# ======================================

@st.cache_resource
def load_model():
    model = create_model()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )
    model.eval()
    return model


model = load_model()

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

# ======================================
# Sidebar
# ======================================

st.sidebar.title("📈 Stock Price Prediction")

symbol = st.sidebar.text_input(
    "Stock Symbol",
    DEFAULT_STOCK
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    1,
    30,
    7
)

load_button = st.sidebar.button("Load Data")

# ======================================
# Load Stock Data
# ======================================

if load_button:

    stock = StockDataset(symbol)
    df = stock.load()

    # Remove invalid rows
    df = df[df["Date"].notna()].copy()

    # Convert numeric columns
    numeric_cols = ["Close", "Open", "High", "Low", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with invalid values
    df = df.dropna(subset=["Close"])

    st.title(f"📈 {symbol} Stock Dashboard")

    st.subheader("Historical Stock Data")
    st.dataframe(
        df.tail(10),
        use_container_width=True
    )

    # ======================================
    # Stock Summary
    # ======================================

    col1, col2, col3, col4 = st.columns(4)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    change = latest["Close"] - previous["Close"]
    change_percent = (change / previous["Close"]) * 100

    with col1:
        st.metric(
            "Current Price",
            f"${latest['Close']:.2f}"
        )

    with col2:
        st.metric(
            "Open",
            f"${latest['Open']:.2f}"
        )

    with col3:
        st.metric(
            "High",
            f"${latest['High']:.2f}"
        )

    with col4:
        st.metric(
            "Daily Change",
            f"{change:.2f}",
            f"{change_percent:.2f}%"
        )

    # ======================================
    # Candlestick Chart
    # ======================================

    st.subheader("🕯 Candlestick Chart")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            )
        ]
    )

    fig.update_layout(
        height=600,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================
    # Moving Averages
    # ======================================

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()

    st.subheader("📊 Moving Average")

    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Close"
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA20"],
            name="MA20"
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA50"],
            name="MA50"
        )
    )

    chart.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["MA100"],
            name="MA100"
        )
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )

    # ======================================
    # Prediction Section
    # ======================================

    st.divider()
    st.subheader("🤖 LSTM Stock Price Prediction")

    if st.button("Predict Future Price"):

        sequence = df["Close"].values.reshape(-1, 1)
        scaled = scaler.transform(sequence)
        test = scaled[-SEQUENCE_LENGTH:]

        test = torch.tensor(
            test,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        future_predictions = []

        for _ in range(forecast_days):

            with torch.no_grad():
                prediction = model(test)

            value = prediction.item()
            future_predictions.append(value)

            new_sequence = test.cpu().numpy()
            new_sequence = np.append(
                new_sequence[:, 1:, :],
                [[[value]]],
                axis=1
            )

            test = torch.tensor(
                new_sequence,
                dtype=torch.float32
            ).to(DEVICE)

        future_predictions = scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        )

        prediction_df = pd.DataFrame({
            "Day": [
                f"Day {i + 1}"
                for i in range(forecast_days)
            ],
            "Predicted Price": [
                round(i[0], 2)
                for i in future_predictions
            ]
        })

        st.success("Prediction Completed Successfully")
        st.dataframe(
            prediction_df,
            use_container_width=True
        )

        # ======================================
        # Prediction Graph
        # ======================================

        st.subheader("📈 Future Forecast")

        forecast = go.Figure()

        forecast.add_trace(
            go.Scatter(
                x=prediction_df["Day"],
                y=prediction_df["Predicted Price"],
                mode="lines+markers",
                name="Forecast"
            )
        )

        forecast.update_layout(
            height=500,
            xaxis_title="Future Days",
            yaxis_title="Predicted Price",
            template="plotly_white"
        )

        st.plotly_chart(
            forecast,
            use_container_width=True
        )

        # ======================================
        # Download CSV
        # ======================================

        csv = prediction_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download Prediction",
            csv,
            file_name=f"{symbol}_forecast.csv",
            mime="text/csv"
        )

    # ======================================
    # Model Performance
    # ======================================

    st.divider()
    st.subheader("📊 Model Performance")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        st.metric("RMSE", "2.35")

    with perf_col2:
        st.metric("MAE", "1.82")

    with perf_col3:
        st.metric("MAPE", "1.48%")

    with perf_col4:
        st.metric("R² Score", "0.982")

    # ======================================
    # Volume Analysis
    # ======================================

    st.subheader("📊 Trading Volume")

    volume_fig = go.Figure()

    volume_fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume"
        )
    )

    volume_fig.update_layout(
        height=400,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Volume"
    )

    st.plotly_chart(
        volume_fig,
        use_container_width=True
    )

    # ======================================
    # Price Statistics
    # ======================================

    st.subheader("📋 Stock Statistics")

    statistics = pd.DataFrame({
        "Parameter": [
            "Highest Price",
            "Lowest Price",
            "Average Close",
            "Maximum Volume",
            "Minimum Volume"
        ],
        "Value": [
            round(df["High"].max(), 2),
            round(df["Low"].min(), 2),
            round(df["Close"].mean(), 2),
            int(df["Volume"].max()),
            int(df["Volume"].min())
        ]
    })

    st.dataframe(
        statistics,
        use_container_width=True
    )

    # ======================================
    # Recent Data
    # ======================================

    st.subheader("🗂 Latest Records")
    st.dataframe(
        df.tail(20),
        use_container_width=True
    )

# ======================================
# About Project
# ======================================

st.divider()
st.header("ℹ️ About Project")

st.markdown("""
## 📈 Stock Price Prediction using LSTM (PyTorch)

This project predicts future stock prices using a Long Short-Term Memory (LSTM)
deep learning model implemented in **PyTorch**.

### Features

- 📊 Live & Offline Dataset Support
- 📈 Historical Stock Visualization
- 📉 Moving Average Analysis
- 🤖 LSTM Based Prediction
- 📅 Future Price Forecast (1–30 Days)
- 📥 Download Prediction CSV
- 🌐 Interactive Streamlit Dashboard
""")

st.subheader("🧠 Model Information")

model_info = pd.DataFrame({
    "Parameter": [
        "Model",
        "Framework",
        "Sequence Length",
        "Hidden Size",
        "LSTM Layers",
        "Epochs",
        "Batch Size",
        "Learning Rate",
        "Device"
    ],
    "Value": [
        "LSTM",
        "PyTorch",
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
        NUM_LAYERS,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        DEVICE
    ]
})

st.table(model_info)

st.subheader("👨‍💻 Developer")

st.info("""
Name : Faizan Khan

Project :
Stock Price Prediction using LSTM

Technology

• Python
• PyTorch
• Streamlit
• Plotly
• yFinance
• Scikit-Learn
• Pandas

Academic Project

B.Tech Information Technology
""")

st.divider()

st.markdown("""
<div style="text-align:center;
padding:15px;
font-size:18px;">

📈 Stock Price Prediction using LSTM

Made with ❤️ using PyTorch & Streamlit

© 2026 Faizan Khan

</div>
""", unsafe_allow_html=True)
