"""CrowdCast AI — Smart Public Transport Crowd Forecasting System.

Interactive Streamlit Dashboard.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import (
    AVAILABLE_ROUTES,
    DAY_TYPES,
    WEATHER_CONDITIONS,
    generate_synthetic_data,
    load_dataset,
)
from src.model import (
    ModelEvaluationReport,
    Pipeline,
    predict_single_trip,
    train_and_evaluate_models,
)
from src.preprocessing import (
    CROWD_LEVEL_ORDER,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
)

# Page configuration
st.set_page_config(
    page_title="CrowdCast AI — Smart Public Transport Crowd Forecasting",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Navy / Teal Styling
CUSTOM_CSS = """
<style>
    /* Main container and typography */
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #0E7490 50%, #06B6D4 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #E2E8F0 !important;
        font-size: 1.05rem;
        margin: 0;
    }
    .synthetic-badge {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 600;
        font-size: 0.8rem;
        padding: 3px 10px;
        border-radius: 9999px;
        border: 1px solid #F59E0B;
        margin-top: 8px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-subtitle {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 0.2rem;
    }
    .status-badge-low {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #10B981;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        display: inline-block;
        text-align: center;
        width: 100%;
    }
    .status-badge-moderate {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #F59E0B;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        display: inline-block;
        text-align: center;
        width: 100%;
    }
    .status-badge-high {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #EF4444;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        display: inline-block;
        text-align: center;
        width: 100%;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #0E7490;
        padding: 0.9rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #334155;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_cached_dataset() -> pd.DataFrame:
    """Load or generate reproducible demonstration dataset."""
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_cached_models(df: pd.DataFrame) -> Tuple[Pipeline, Pipeline, ModelEvaluationReport, pd.DataFrame]:
    """Train and cache the ML pipelines and test evaluation report."""
    return train_and_evaluate_models(df, test_size=0.2, random_state=42)


# Load dataset and train models
raw_df = get_cached_dataset()
clf_pipeline, reg_pipeline, eval_report, X_test = get_cached_models(raw_df)

# Header Section
st.markdown(
    """
    <div class="main-header">
        <h1>CrowdCast AI — Smart Public Transport Crowd Forecasting</h1>
        <p>Real-time passenger load prediction and crowd intelligence powered by dual machine learning pipelines.</p>
        <div class="synthetic-badge">⚠️ SYNTHETIC DEMONSTRATION DATA &bull; Not for real-world life-critical scheduling</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: Trip Input Configuration
with st.sidebar:
    st.header("🚏 Trip Configuration")
    st.markdown("Configure trip parameters to generate live crowd and passenger estimates.")

    selected_route = st.selectbox(
        "Select Transit Route",
        options=AVAILABLE_ROUTES,
        index=0,
        help="Transit route corridor and operational characteristics.",
    )

    selected_day = st.selectbox(
        "Day Type",
        options=DAY_TYPES,
        index=0,
        help="Weekday commuter flow differs markedly from leisure weekend patterns.",
    )

    departure_hour = st.slider(
        "Departure Hour (24-hour clock)",
        min_value=5,
        max_value=23,
        value=8,
        step=1,
        format="%02d:00",
        help="Hour of trip departure (morning rush: 7-9 AM, evening rush: 4-7 PM).",
    )

    selected_weather = st.selectbox(
        "Weather Condition",
        options=WEATHER_CONDITIONS,
        index=0,
        help="Inclement weather increases passenger volume as commuters shift from active travel.",
    )

    st.subheader("Contextual Modifiers")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        has_nearby_event = st.checkbox(
            "Major Event",
            value=False,
            help="Stadium matches, concert, or festival near transit stops.",
        )
    with col_sb2:
        is_college_term = st.checkbox(
            "School in Term",
            value=True,
            help="Active university semester increases commuter volume.",
        )

    trip_duration = st.slider(
        "Scheduled Trip Duration (minutes)",
        min_value=12,
        max_value=85,
        value=38,
        step=1,
        help="Longer routes accumulate more passengers across multiple transit stops.",
    )

    predict_btn = st.button("🔮 Forecast Crowd & Demand", use_container_width=True, type="primary")

# Build prediction feature vector
current_input = {
    "route_id": selected_route,
    "day_type": selected_day,
    "departure_hour": departure_hour,
    "weather": selected_weather,
    "nearby_event": 1 if has_nearby_event else 0,
    "college_term": 1 if is_college_term else 0,
    "trip_duration": float(trip_duration),
}

prediction = predict_single_trip(clf_pipeline, reg_pipeline, current_input)
crowd_lvl = prediction["crowd_level"]
est_passengers = prediction["estimated_passengers"]
uncertainty = prediction["uncertainty_std"]
probs = prediction["crowd_probabilities"]

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "⚡ Live Crowd Forecast",
        "📊 Model Evaluation",
        "📈 Transit Demand Patterns",
        "📂 Dataset Explorer",
        "📖 Methodology & Limits",
    ]
)

# ----------------- TAB 1: Live Forecast -----------------
with tab1:
    st.subheader("Forecast Summary for Selected Departure")

    # Format departure hour nicely
    ampm = "AM" if departure_hour < 12 else "PM"
    hour_display = departure_hour if departure_hour <= 12 else departure_hour - 12
    if hour_display == 0:
        hour_display = 12
    time_str = f"{hour_display}:00 {ampm}"

    st.caption(
        f"**Route**: {selected_route} &nbsp;|&nbsp; **Schedule**: {selected_day} at {time_str} &nbsp;|&nbsp; "
        f"**Weather**: {selected_weather} &nbsp;|&nbsp; **Duration**: {trip_duration} mins"
    )

    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Predicted Crowd Level</div>', unsafe_allow_html=True)

        if crowd_lvl == "Low":
            badge_html = f'<div class="status-badge-low">🟢 LOW CROWD</div>'
            desc = f"Seats readily available (&lt; {LOW_THRESHOLD} passengers). Relaxed ride."
        elif crowd_lvl == "Moderate":
            badge_html = f'<div class="status-badge-moderate">🟡 MODERATE CROWD</div>'
            desc = f"Seating mostly taken ({LOW_THRESHOLD}–{HIGH_THRESHOLD} pass.). Light standing room."
        else:
            badge_html = f'<div class="status-badge-high">🔴 HIGH CROWD</div>'
            desc = f"Near or at capacity (&gt; {HIGH_THRESHOLD} pass.). Heavy standing room expected."

        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown(f'<div class="metric-subtitle" style="margin-top:0.6rem;">{desc}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Estimated Passenger Count</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{est_passengers} <span style="font-size:1.1rem; color:#64748B;">&plusmn; {uncertainty}</span></div>', unsafe_allow_html=True)

        # Vehicle capacity bar (standard city bus capacity ~85 seated + standing)
        bus_capacity = 85
        cap_pct = min(100, int((est_passengers / bus_capacity) * 100))
        st.markdown(
            f'<div class="metric-subtitle">Bus Capacity Utilization: <b>{cap_pct}%</b> (nominal 85-passenger vehicle)</div>',
            unsafe_allow_html=True,
        )
        st.progress(cap_pct / 100.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-title">Classifier Probability Distribution</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame(
            {
                "Level": CROWD_LEVEL_ORDER,
                "Probability": [probs.get(k, 0.0) for k in CROWD_LEVEL_ORDER],
                "Color": ["#10B981", "#F59E0B", "#EF4444"],
            }
        )

        fig_prob = px.bar(
            prob_df,
            x="Level",
            y="Probability",
            text="Probability",
            color="Level",
            color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"},
            range_y=[0, 1.05],
        )
        fig_prob.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside",
            marker_line_width=1.5,
            marker_line_color="#334155",
        )
        fig_prob.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=165,
            showlegend=False,
            xaxis_title="",
            yaxis_title="",
            yaxis=dict(tickformat=".0%"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Commuter & Dispatcher Advisory Card
    st.markdown('<div class="disclaimer-box">', unsafe_allow_html=True)
    if crowd_lvl == "High":
        st.markdown(
            "**Operational Advisory:** Severe passenger volume forecast. "
            "Dispatchers are advised to inject an auxiliary short-turn vehicle or advise passengers to depart 20 minutes earlier."
        )
    elif crowd_lvl == "Moderate":
        st.markdown(
            "**Operational Advisory:** Typical volume conditions. All scheduled vehicles are operating within standard headways."
        )
    else:
        st.markdown(
            "**Operational Advisory:** Low occupancy forecast. Ideal travel window for passengers carrying luggage or requiring accessibility seating."
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 2: Model Evaluation -----------------
with tab2:
    st.subheader("Model Performance on Held-Out Test Set (20% Split)")
    st.markdown(
        """
        <div class="disclaimer-box">
            <b>Evaluation Protocol:</b> Metrics are calculated against a strictly held-out 20% test partition (N = 400 trips)
            isolated before training. Cross-validation and fixed random seeds ensure zero leakage.
            <br><i>Notice: These statistics demonstrate modeling capability on the synthetic transit dataset and do not represent actual transit authority telemetry.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Row
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.metric(label="Classification Accuracy", value=f"{eval_report.accuracy * 100:.1f}%")
    with mcol2:
        macro_f1 = eval_report.classification_rep.get("macro avg", {}).get("f1-score", 0.0)
        st.metric(label="Macro F1 Score", value=f"{macro_f1:.3f}")
    with mcol3:
        st.metric(label="Regression MAE", value=f"{eval_report.mae:.2f} pass.")
    with mcol4:
        st.metric(label="Regression RMSE", value=f"{eval_report.rmse:.2f} pass.")
    with mcol5:
        st.metric(label="R² Score", value=f"{eval_report.r2:.3f}")

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.markdown("#### Classifier Confusion Matrix")
        cm = eval_report.confusion_mat
        classes = eval_report.classes

        # Heatmap annotation text
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=classes,
            y=classes,
            labels=dict(x="Predicted Class", y="Actual True Class", color="Trip Count"),
            color_continuous_scale="Teal",
        )
        fig_cm.update_layout(
            height=340,
            margin=dict(l=40, r=40, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class metrics table
        st.markdown("#### Per-Class Classification Report")
        report_data = []
        for cls in classes:
            c_dict = eval_report.classification_rep.get(cls, {})
            report_data.append(
                {
                    "Crowd Level": cls,
                    "Precision": f"{c_dict.get('precision', 0.0):.3f}",
                    "Recall": f"{c_dict.get('recall', 0.0):.3f}",
                    "F1-Score": f"{c_dict.get('f1-score', 0.0):.3f}",
                    "Support": int(c_dict.get("support", 0)),
                }
            )
        st.dataframe(pd.DataFrame(report_data), hide_index=True, use_container_width=True)

    with eval_col2:
        st.markdown("#### Actual vs. Predicted Passenger Counts (Test Split)")
        y_act = eval_report.y_test_reg_actual
        y_pred = eval_report.y_test_reg_pred

        scatter_df = pd.DataFrame(
            {
                "Actual Passengers": y_act,
                "Predicted Passengers": y_pred,
                "True Crowd Level": eval_report.y_test_clf_actual,
            }
        )

        fig_reg = px.scatter(
            scatter_df,
            x="Actual Passengers",
            y="Predicted Passengers",
            color="True Crowd Level",
            color_discrete_map={"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"},
            opacity=0.7,
            hover_data={"Actual Passengers": ":.0f", "Predicted Passengers": ":.1f"},
        )

        # 45-degree perfect prediction line
        min_val = min(y_act.min(), y_pred.min())
        max_val = max(y_act.max(), y_pred.max())
        fig_reg.add_shape(
            type="line",
            line=dict(dash="dash", color="#64748B", width=2),
            x0=min_val,
            y0=min_val,
            x1=max_val,
            y1=max_val,
        )

        fig_reg.update_layout(
            height=460,
            margin=dict(l=30, r=30, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_reg, use_container_width=True)


# ----------------- TAB 3: Demand Visualizations -----------------
with tab3:
    st.subheader("Public Transit Passenger Demand Analytics")
    st.markdown("Explore aggregate passenger loading trends across operating corridors, time of day, and weather.")

    vis_col1, vis_col2 = st.columns([1.6, 1.2])

    with vis_col1:
        st.markdown("#### Hourly Demand Profiles Across Transit Corridors")
        hourly_route = (
            raw_df.groupby(["departure_hour", "route_id"])["passenger_count"]
            .mean()
            .reset_index()
        )
        fig_hourly = px.line(
            hourly_route,
            x="departure_hour",
            y="passenger_count",
            color="route_id",
            markers=True,
            labels={"departure_hour": "Departure Hour (24h)", "passenger_count": "Avg Passenger Count", "route_id": "Route"},
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_hourly.add_vrect(
            x0=7,
            x1=9,
            fillcolor="#0E7490",
            opacity=0.1,
            annotation_text="Morning Rush",
            annotation_position="top left",
        )
        fig_hourly.add_vrect(
            x0=16,
            x1=19,
            fillcolor="#0E7490",
            opacity=0.1,
            annotation_text="Evening Rush",
            annotation_position="top left",
        )
        fig_hourly.update_layout(
            height=380,
            margin=dict(l=30, r=20, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with vis_col2:
        st.markdown("#### Demand Sensitivity: Weather & Day Type")
        weather_day = (
            raw_df.groupby(["weather", "day_type"])["passenger_count"]
            .mean()
            .reset_index()
        )
        fig_weather = px.bar(
            weather_day,
            x="weather",
            y="passenger_count",
            color="day_type",
            barmode="group",
            labels={"weather": "Weather", "passenger_count": "Avg Passengers", "day_type": "Day Type"},
            color_discrete_map={"Weekday": "#0E7490", "Weekend": "#06B6D4", "Holiday": "#94A3B8"},
        )
        fig_weather.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    # Route volume distribution
    st.markdown("#### Route Load Distribution & Capacity Thresholds")
    fig_box = px.box(
        raw_df,
        x="route_id",
        y="passenger_count",
        color="route_id",
        points="outliers",
        labels={"route_id": "Route ID", "passenger_count": "Passenger Count"},
    )
    # Add horizontal threshold guide lines
    fig_box.add_hline(
        y=LOW_THRESHOLD,
        line_dash="dot",
        line_color="#10B981",
        annotation_text="Low/Moderate Threshold (35 pass.)",
    )
    fig_box.add_hline(
        y=HIGH_THRESHOLD,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Moderate/High Threshold (70 pass.)",
    )
    fig_box.update_layout(
        height=320,
        margin=dict(l=30, r=20, t=20, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ----------------- TAB 4: Dataset Explorer -----------------
with tab4:
    st.subheader("Dataset Explorer & Export")
    st.markdown(
        "Inspect the training records, descriptive summary statistics, and export the dataset as CSV."
    )

    filter_c1, filter_c2, filter_c3 = st.columns(3)
    with filter_c1:
        route_filter = st.multiselect(
            "Filter by Route",
            options=AVAILABLE_ROUTES,
            default=AVAILABLE_ROUTES,
        )
    with filter_c2:
        level_filter = st.multiselect(
            "Filter by Crowd Level",
            options=CROWD_LEVEL_ORDER,
            default=CROWD_LEVEL_ORDER,
        )
    with filter_c3:
        day_filter = st.multiselect(
            "Filter by Day Type",
            options=DAY_TYPES,
            default=DAY_TYPES,
        )

    filtered_df = raw_df[
        raw_df["route_id"].isin(route_filter)
        & raw_df["crowd_level"].isin(level_filter)
        & raw_df["day_type"].isin(day_filter)
    ]

    st.write(f"Showing **{len(filtered_df):,}** of **{len(raw_df):,}** total records:")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="crowdcast_filtered_data.csv",
        mime="text/csv",
    )

    st.markdown("#### Descriptive Statistics (Numerical Features)")
    st.dataframe(
        raw_df[["departure_hour", "trip_duration", "passenger_count"]].describe().round(2),
        use_container_width=True,
    )


# ----------------- TAB 5: Methodology & Limitations -----------------
with tab5:
    st.subheader("System Architecture, Methodology & Project Limitations")

    st.markdown(
        """
        ### 1. Dual Machine Learning Pipeline Architecture
        CrowdCast AI adopts a dual-model formulation to provide both coarse operational classification and granular count estimation:
        - **RandomForestClassifier**: Trained directly on categorized crowd levels (*Low*, *Moderate*, *High*). Employs 100 decision trees to estimate calibrated class posterior probabilities $P(\\text{Crowd} = c \\mid X)$.
        - **RandomForestRegressor**: Trained to predict continuous passenger counts. Computes inter-tree variance $\\sigma_{\\text{trees}}$ across individual estimators to provide real-time prediction uncertainty intervals.

        ### 2. Feature Engineering & Preprocessing
        - **Numerical Features** (`departure_hour`, `nearby_event`, `college_term`, `trip_duration`): Imputed with median values and standardized via `StandardScaler`.
        - **Categorical Features** (`route_id`, `day_type`, `weather`): Imputed with mode and encoded using `OneHotEncoder(handle_unknown='ignore')`. Unseen categories during deployment gracefully fall back to zeroed one-hot vectors without crashing.
        - **Crowd Thresholds**:
            - **Low**: $< 35$ passengers (abundant seating)
            - **Moderate**: $35 - 70$ passengers (seats filled, light standing)
            - **High**: $> 70$ passengers (crush capacity, heavy standing)

        ### 3. Verification & Limitations
        - **Synthetic Data Disclaimer**: This system is trained and evaluated strictly on **SYNTHETIC DEMONSTRATION DATA** engineered to simulate commuter dynamics. Metrics must not be represented as actual city transit telemetry.
        - **External Factors**: The model does not currently ingest live GPS vehicle position, road construction bottlenecks, or automated fare gate tap-in feeds.
        - **Replacement with Real Data**: To retrain on real Automatic Passenger Counter (APC) logs, replace `data/crowdcast_sample_data.csv` with real agency records matching the standard schema and restart the app.
        """
    )
