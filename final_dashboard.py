import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =====================================================
# CONFIG
# =====================================================

LOOKBACK = 20
HORIZON = 60

st.set_page_config(
    page_title="Traffic Density Forecast",
    layout="wide"
)

# =====================================================
# MODEL
# =====================================================

class CNNModel(nn.Module):

    def __init__(self, n_features=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.conv(x)

        x = torch.mean(x, dim=2)

        return self.fc(x)

# =====================================================
# LOAD MODEL
# =====================================================

model = CNNModel()

model.load_state_dict(
    torch.load(
        "cnn_traffic_model.pth",
        map_location="cpu"
    )
)

model.eval()

feature_mean = np.load("feature_mean.npy")
feature_std = np.load("feature_std.npy")

# =====================================================
# UI
# =====================================================

st.title("🚦 Traffic Density Forecast Dashboard")

st.write(
    f"Predict traffic density {HORIZON} seconds ahead "
    f"using last {LOOKBACK} seconds."
)

uploaded_file = st.file_uploader(
    "Upload Density CSV",
    type=["csv"]
)

# =====================================================
# PROCESS FILE
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.dataframe(df)

    # ---------------------------------------------
    # Expect a density column
    # ---------------------------------------------

    if "density" not in df.columns:

        st.error(
            "CSV must contain a column named 'density'"
        )

        st.stop()

    # ---------------------------------------------
    # Feature Engineering
    # ---------------------------------------------

    feature_df = pd.DataFrame()

    feature_df["density"] = df["density"]

    feature_df["mean_5"] = (
        feature_df["density"]
        .rolling(5)
        .mean()
        .fillna(0)
    )

    feature_df["std_5"] = (
        feature_df["density"]
        .rolling(5)
        .std()
        .fillna(0)
    )

    feature_df["diff1"] = (
        feature_df["density"]
        .diff()
        .fillna(0)
    )

    features = feature_df[
        ["density", "mean_5", "std_5", "diff1"]
    ].values

    # ---------------------------------------------
    # Normalize
    # ---------------------------------------------

    features = (
        features - feature_mean
    ) / (feature_std + 1e-8)

    if len(features) < LOOKBACK:

        st.error(
            f"Need at least {LOOKBACK} rows."
        )

        st.stop()

    # ---------------------------------------------
    # Last sequence
    # ---------------------------------------------

    seq = features[-LOOKBACK:]

    x = torch.tensor(
        seq.reshape(1, LOOKBACK, 4),
        dtype=torch.float32
    )

    # ---------------------------------------------
    # Predict
    # ---------------------------------------------

    with torch.no_grad():

        forecast_density = (
            model(x)
            .squeeze()
            .item()
        )

    current_density = (
        df["density"]
        .iloc[-1]
    )

    # ---------------------------------------------
    # Congestion Threshold
    # ---------------------------------------------

    threshold = 20

    forecast_status = (
        "🚨 CONGESTED"
        if forecast_density > threshold
        else "✅ NORMAL"
    )

    current_status = (
        "🚨 CONGESTED"
        if current_density > threshold
        else "✅ NORMAL"
    )

    # =================================================
    # RESULTS
    # =================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Current Density",
            f"{current_density:.2f}"
        )

        st.metric(
            "Current Status",
            current_status
        )

    with col2:

        st.metric(
            f"Forecast Density (+{HORIZON}s)",
            f"{forecast_density:.2f}"
        )

        st.metric(
            "Forecast Status",
            forecast_status
        )

    # =================================================
    # LAST 20 LOOKUPS
    # =================================================

    st.subheader(
        "📊 Last 20 Density Values"
    )

    last20 = df["density"].tail(20)

    chart_df = pd.DataFrame({
        "Density": last20.values
    })

    st.line_chart(chart_df)

    # =================================================
    # COMPARISON
    # =================================================

    st.subheader(
        "📈 Current vs Forecast"
    )

    compare = pd.DataFrame({
        "Value": [
            current_density,
            forecast_density
        ]
    },
    index=[
        "Current",
        "Forecast"
    ])

    st.bar_chart(compare)

    # =================================================
    # DETAILS
    # =================================================

    st.subheader(
        "Forecast Summary"
    )

    st.write(
        f"""
        Current Density : **{current_density:.2f}**

        Predicted Density after **{HORIZON} seconds** :
        **{forecast_density:.2f}**

        Status :
        **{forecast_status}**
        """
    )