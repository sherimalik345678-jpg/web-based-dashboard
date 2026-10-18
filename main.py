"""
================================================================================
STREAMLIT POWER SHOWCASE: Interactive Data Analytics & Visualization Hub
================================================================================
This application is designed to demonstrate the immense capabilities of Streamlit:
1. Native Reactive Charts (Line, Area, Bar, Scatter)
2. Interactive Plotly Visualizations (3D Surface, Candlestick, Bubble, Sunburst)
3. Statistical & Scientific Visualizations (Seaborn Correlation Heatmaps, Violin Plots)
4. Geospatial Intelligence (Interactive Maps with Dynamic Coordinates)
5. Live Real-time Streaming Simulation (In-place reactive component updating)
6. Two-Way Interactive Data Editing (Live updates from st.data_editor)
7. Advanced UI Layouts (Custom CSS, Metric Cards, Tabs, Filters, and Sidebars)

Run this file with:
    streamlit run main.py
================================================================================
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Streamlit Power Showcase",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished aesthetics and card components
st.markdown(
    """
    <style>
    /* Global font & background accents */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Title Styling */
    .hero-container {
        padding: 1.5rem 1.8rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Custom Card Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* Feature Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. SYNTHETIC DATA GENERATION (CACHED FOR OPTIMAL PERFORMANCE)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_timeseries_data(n_days: int = 180, volatility: float = 1.0, seed: int = 42):
    """Generates synthetic multi-department enterprise performance data."""
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="D")
    
    # Base walk with trend and volatility
    drift = np.linspace(100, 350, n_days)
    rev_saas = drift + np.cumsum(np.random.normal(0, 5 * volatility, n_days))
    rev_hardware = (drift * 0.7) + np.cumsum(np.random.normal(0, 4 * volatility, n_days))
    rev_services = (drift * 0.4) + np.cumsum(np.random.normal(0, 3 * volatility, n_days))
    
    df = pd.DataFrame({
        "Date": dates,
        "SaaS Cloud": np.maximum(rev_saas, 20),
        "Hardware Devices": np.maximum(rev_hardware, 15),
        "Professional Services": np.maximum(rev_services, 10),
    })
    df["Total Revenue"] = df["SaaS Cloud"] + df["Hardware Devices"] + df["Professional Services"]
    df["Operating Cost"] = df["Total Revenue"] * np.random.uniform(0.45, 0.62, n_days)
    df["Net Profit"] = df["Total Revenue"] - df["Operating Cost"]
    return df

@st.cache_data
def generate_scatter_data(n_records: int = 350, seed: int = 42):
    """Generates multi-attribute product metrics for high-dimensional plotting."""
    np.random.seed(seed)
    categories = ["AI Infrastructure", "DevTools", "Cybersecurity", "Fintech API", "Data Ops"]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    
    cat_choices = np.random.choice(categories, n_records)
    reg_choices = np.random.choice(regions, n_records)
    
    ad_spend = np.random.exponential(scale=18000, size=n_records) + 2000
    # Revenue correlates with spend + category multiplier
    multipliers = {
        "AI Infrastructure": 4.5,
        "DevTools": 3.2,
        "Cybersecurity": 3.8,
        "Fintech API": 4.1,
        "Data Ops": 3.0,
    }
    cat_mult = np.array([multipliers[c] for c in cat_choices])
    raw_revenue = ad_spend * cat_mult * np.random.normal(1.0, 0.15, n_records)
    revenue = np.maximum(raw_revenue, 5000)
    customer_acq = np.random.poisson(lam=120, size=n_records) + (ad_spend / 350).astype(int)
    satisfaction = np.random.beta(a=7, b=2, size=n_records) * 10  # 1-10 rating
    
    df = pd.DataFrame({
        "Product Category": cat_choices,
        "Region": reg_choices,
        "Ad Spend ($)": ad_spend.round(2),
        "Revenue ($)": revenue.round(2),
        "Customers Acquired": customer_acq,
        "Customer Satisfaction": satisfaction.round(1),
        "ROI Ratio": (revenue / ad_spend).round(2),
    })
    return df

@st.cache_data
def generate_candlestick_data(n_candles: int = 100, seed: int = 42):
    """Generates synthetic stock/crypto OHLCV candlestick data."""
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_candles, freq="B")
    
    price = 150.0
    opens, highs, lows, closes, volumes = [], [], [], [], []
    
    for _ in range(n_candles):
        change = np.random.normal(0.4, 2.8)
        current_open = price
        current_close = current_open + change
        current_high = max(current_open, current_close) + abs(np.random.normal(0.5, 1.2))
        current_low = min(current_open, current_close) - abs(np.random.normal(0.5, 1.2))
        volume = int(np.random.uniform(50000, 300000))
        
        opens.append(current_open)
        highs.append(current_high)
        lows.append(current_low)
        closes.append(current_close)
        volumes.append(volume)
        price = current_close
        
    df = pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    })
    # Moving Averages
    df["SMA 20"] = df["Close"].rolling(window=20).mean()
    df["SMA 50"] = df["Close"].rolling(window=50).mean()
    return df

@st.cache_data
def generate_geospatial_data(n_locations: int = 120, seed: int = 42):
    """Generates geospatial hub data across global tech regions."""
    np.random.seed(seed)
    # Centers: US West, US East, Central Europe, East Asia
    centers = [
        {"name": "Silicon Valley", "lat": 37.7749, "lon": -122.4194},
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
        {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    ]
    
    rows = []
    for _ in range(n_locations):
        c = centers[np.random.choice(len(centers))]
        lat = c["lat"] + np.random.normal(0, 2.5)
        lon = c["lon"] + np.random.normal(0, 3.5)
        activity = np.random.randint(100, 10000)
        servers = np.random.randint(5, 50)
        rows.append({"Hub": c["name"], "lat": lat, "lon": lon, "Activity": activity, "Active Nodes": servers})
        
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & DYNAMIC FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://streamlit.io/images/brand/streamlit-mark-color.svg",
        width=48,
    )
    st.title("Control Panel")
    st.caption("Interact with widgets to dynamically regenerate and filter dashboard state.")
    
    st.divider()
    
    st.subheader("⚙️ Dataset Parameters")
    sample_period = st.slider("Historical Timeframe (Days)", min_value=30, max_value=365, value=120, step=15)
    volatility_param = st.slider("Market Volatility Factor", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
    random_seed = st.number_input("Random Generator Seed", min_value=1, max_value=9999, value=42, step=1)
    
    st.divider()
    
    st.subheader("🎨 Theme & Aesthetics")
    palette_choice = st.selectbox(
        "Plotly Colorscale",
        options=["Viridis", "Plasma", "Turbo", "Cividis", "Spectral", "Magma"],
        index=0,
    )
    show_raw_data = st.toggle("Show Raw Data Inspector", value=False)
    
    st.divider()
    st.markdown(
        """
        **Why Streamlit?**
        - No HTML/JS boilerplate
        - Pure Python execution
        - Reactive computation
        - Instant interactive UI
        """
    )


# -----------------------------------------------------------------------------
# 4. LOAD STATE AND COMPUTE DATASETS
# -----------------------------------------------------------------------------
df_ts = generate_timeseries_data(n_days=sample_period, volatility=volatility_param, seed=random_seed)
df_scatter = generate_scatter_data(n_records=300, seed=random_seed)
df_candlestick = generate_candlestick_data(n_candles=min(sample_period, 120), seed=random_seed)
df_geo = generate_geospatial_data(n_locations=150, seed=random_seed)

# -----------------------------------------------------------------------------
# 5. HEADER & TOP-TIER METRIC CARDS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <span class="badge">STREAMLIT INTERACTIVE ANALYTICS</span>
        <h1 class="hero-title">Unleashing the Power of Streamlit</h1>
        <p class="hero-subtitle">
            Explore native charts, 3D interactive plots, statistical heatmaps, geospatial intelligence, 
            and real-time streaming simulations — all powered effortlessly with Python.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top metric summary cards with deltas
total_revenue = df_ts["Total Revenue"].iloc[-1]
prev_revenue = df_ts["Total Revenue"].iloc[-2]
rev_delta = ((total_revenue - prev_revenue) / prev_revenue) * 100

total_profit = df_ts["Net Profit"].sum()
profit_margin = (total_profit / df_ts["Total Revenue"].sum()) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Latest Daily Revenue",
        value=f"${total_revenue:,.2f}",
        delta=f"{rev_delta:+.2f}%",
    )
with col2:
    st.metric(
        label="Cumulative Net Profit",
        value=f"${total_profit:,.0f}",
        delta=f"{profit_margin:.1f}% Margin",
    )
with col3:
    st.metric(
        label="Avg Customer Satisfaction",
        value=f"{df_scatter['Customer Satisfaction'].mean():.1f} / 10",
        delta="+0.4 MoM",
    )
with col4:
    st.metric(
        label="Active Global Edge Nodes",
        value=f"{df_geo['Active Nodes'].sum():,} Nodes",
        delta="100% Operational",
    )

st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. TAB NAVIGATION ACROSS VISUALIZATION DOMAINS
# -----------------------------------------------------------------------------
tab_native, tab_plotly, tab_stats, tab_geo, tab_stream, tab_editor = st.tabs([
    "📊 Native Streamlit Charts",
    "🚀 Interactive Plotly Visualizations",
    "🔬 Statistical Analysis (Seaborn)",
    "🌍 Geospatial Intelligence",
    "⚡ Real-Time Streaming Simulation",
    "📝 Live Data Editor & Two-Way Reactivity",
])


# =============================================================================
# TAB 1: NATIVE STREAMLIT CHARTS
# =============================================================================
with tab_native:
    st.markdown("### ⚡ Instant, Zero-Boilerplate Native Charts")
    st.write(
        "Streamlit includes lightning-fast built-in chart elements (`st.line_chart`, `st.area_chart`, "
        "`st.bar_chart`, `st.scatter_chart`) built on Altair. They automatically handle responsive resizing, "
        "interactive zooming, panning, and tooltip inspection with a single line of Python code."
    )
    
    col_n1, col_n2 = st.columns(2)
    
    with col_n1:
        st.markdown("#### Multi-Department Revenue Trajectory")
        chart_data_lines = df_ts.set_index("Date")[["SaaS Cloud", "Hardware Devices", "Professional Services"]]
        st.line_chart(chart_data_lines, use_container_width=True)
        
    with col_n2:
        st.markdown("#### Cumulative Revenue vs Operating Cost")
        chart_data_area = df_ts.set_index("Date")[["Total Revenue", "Operating Cost"]]
        st.area_chart(chart_data_area, use_container_width=True)
        
    col_n3, col_n4 = st.columns(2)
    
    with col_n3:
        st.markdown("#### Revenue by Product Category (Aggregated Bar)")
        cat_summary = df_scatter.groupby("Product Category")["Revenue ($)"].sum().reset_index()
        st.bar_chart(cat_summary, x="Product Category", y="Revenue ($)", use_container_width=True)
        
    with col_n4:
        st.markdown("#### Ad Spend vs Revenue Distribution (Native Scatter)")
        st.scatter_chart(
            df_scatter,
            x="Ad Spend ($)",
            y="Revenue ($)",
            color="Product Category",
            size="Customers Acquired",
            use_container_width=True,
        )


# =============================================================================
# TAB 2: INTERACTIVE PLOTLY VISUALIZATIONS
# =============================================================================
with tab_plotly:
    st.markdown("### 🚀 High-Fidelity Plotly Charts with 360° Interactivity")
    st.write(
        "Plotly charts rendered through `st.plotly_chart` give users cinematic 3D rotation, "
        "financial range sliders, hierarchical sunburst drill-downs, and multi-dimensional bubble charts."
    )
    
    # Row 1: 3D Surface & Financial Candlestick
    pcol1, pcol2 = st.columns(2)
    
    with pcol1:
        st.markdown("#### 3D Mathematical & Volatility Terrain Surface")
        # Generate 3D grid data
        x_grid = np.linspace(-3, 3, 40)
        y_grid = np.linspace(-3, 3, 40)
        X, Y = np.meshgrid(x_grid, y_grid)
        # Sinc function with ripple
        R = np.sqrt(X**2 + Y**2) + 0.1
        Z = np.sin(R * volatility_param) / R
        
        fig_surface = go.Figure(
            data=[
                go.Surface(
                    z=Z,
                    x=X,
                    y=Y,
                    colorscale=palette_choice,
                    opacity=0.9,
                    contours={
                        "z": {"show": True, "usecolormap": True, "project_z": True}
                    },
                )
            ]
        )
        fig_surface.update_layout(
            title="Interactive Volatility Topology (Rotate, Pan & Zoom)",
            autosize=True,
            margin=dict(l=10, r=10, b=10, t=40),
            scene=dict(
                xaxis_title="Risk Alpha",
                yaxis_title="Beta Exposure",
                zaxis_title="Yield Ratio",
            ),
            template="plotly_dark",
        )
        st.plotly_chart(fig_surface, use_container_width=True)
        
    with pcol2:
        st.markdown("#### Financial Candlestick (OHLCV) with Moving Averages")
        fig_candle = go.Figure()
        
        # Candlestick trace
        fig_candle.add_trace(
            go.Candlestick(
                x=df_candlestick["Date"],
                open=df_candlestick["Open"],
                high=df_candlestick["High"],
                low=df_candlestick["Low"],
                close=df_candlestick["Close"],
                name="Asset Price",
            )
        )
        # 20 SMA & 50 SMA
        fig_candle.add_trace(
            go.Scatter(
                x=df_candlestick["Date"],
                y=df_candlestick["SMA 20"],
                mode="lines",
                name="20-Day SMA",
                line=dict(color="#38bdf8", width=1.5),
            )
        )
        fig_candle.add_trace(
            go.Scatter(
                x=df_candlestick["Date"],
                y=df_candlestick["SMA 50"],
                mode="lines",
                name="50-Day SMA",
                line=dict(color="#f43f5e", width=1.5, dash="dot"),
            )
        )
        fig_candle.update_layout(
            title="Interactive Price Range Slider & Technical Indicators",
            xaxis_rangeslider_visible=True,
            template="plotly_dark",
            margin=dict(l=10, r=10, b=10, t=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    # Row 2: Multi-dimensional Bubble Chart & Hierarchical Sunburst
    pcol3, pcol4 = st.columns(2)
    
    with pcol3:
        st.markdown("#### Multi-Dimensional Bubble Scatter: ROI vs Satisfaction")
        fig_bubble = px.scatter(
            df_scatter,
            x="Ad Spend ($)",
            y="Revenue ($)",
            size="Customers Acquired",
            color="Product Category",
            hover_name="Product Category",
            hover_data={"Region": True, "ROI Ratio": True, "Customer Satisfaction": True},
            color_continuous_scale=palette_choice,
            template="plotly_dark",
            title="Size = Customer Base | Hover = Deep Details",
        )
        fig_bubble.update_layout(margin=dict(l=10, r=10, b=10, t=40))
        st.plotly_chart(fig_bubble, use_container_width=True)

    with pcol4:
        st.markdown("#### Hierarchical Drilldown: Sunburst Tree")
        fig_sunburst = px.sunburst(
            df_scatter,
            path=["Region", "Product Category"],
            values="Revenue ($)",
            color="Customer Satisfaction",
            color_continuous_scale=palette_choice,
            template="plotly_dark",
            title="Click Segments to Drill In / Out",
        )
        fig_sunburst.update_layout(margin=dict(l=10, r=10, b=10, t=40))
        st.plotly_chart(fig_sunburst, use_container_width=True)


# =============================================================================
# TAB 3: STATISTICAL & SCIENTIFIC ANALYSIS (SEABORN & MATPLOTLIB)
# =============================================================================
with tab_stats:
    st.markdown("### 🔬 Scientific & Statistical Analysis with Seaborn & Matplotlib")
    st.write(
        "Data science workflows built with `matplotlib` and `seaborn` render seamlessly "
        "via `st.pyplot(fig)`. You can incorporate publication-grade statistical figures, "
        "correlation matrices, and distribution densities directly into your web applications."
    )
    
    scol1, scol2 = st.columns(2)
    
    with scol1:
        st.markdown("#### Correlation Matrix with Annotated Heatmap")
        # Numerical features for correlation
        corr_cols = ["Ad Spend ($)", "Revenue ($)", "Customers Acquired", "Customer Satisfaction", "ROI Ratio"]
        corr_matrix = df_scatter[corr_cols].corr()
        
        # Mask upper triangle for aesthetic visualization
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig_heat, ax_heat = plt.subplots(figsize=(6.5, 5))
        fig_heat.patch.set_facecolor("#0e1117")
        ax_heat.set_facecolor("#0e1117")
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="mako",
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": 0.8},
            ax=ax_heat,
            linewidths=1.2,
            linecolor="#1e293b",
        )
        ax_heat.tick_params(colors="#cbd5e1", labelsize=9)
        plt.setp(ax_heat.get_xticklabels(), rotation=30, ha="right")
        plt.title("Pearson Feature Correlation Matrix", color="#f8fafc", fontsize=12, pad=12)
        fig_heat.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)
        
    with scol2:
        st.markdown("#### Kernel Density & Violin Distribution across Regions")
        fig_vio, ax_vio = plt.subplots(figsize=(6.5, 5))
        fig_vio.patch.set_facecolor("#0e1117")
        ax_vio.set_facecolor("#0e1117")
        
        sns.violinplot(
            data=df_scatter,
            x="Region",
            y="ROI Ratio",
            hue="Region",
            palette="crest",
            legend=False,
            inner="quartile",
            ax=ax_vio,
        )
        ax_vio.tick_params(colors="#cbd5e1", labelsize=9)
        ax_vio.spines["bottom"].set_color("#475569")
        ax_vio.spines["left"].set_color("#475569")
        ax_vio.spines["top"].set_visible(False)
        ax_vio.spines["right"].set_visible(False)
        ax_vio.xaxis.label.set_color("#f8fafc")
        ax_vio.yaxis.label.set_color("#f8fafc")
        plt.setp(ax_vio.get_xticklabels(), rotation=20, ha="right")
        plt.title("ROI Ratio Density & Quartile Spread by Territory", color="#f8fafc", fontsize=12, pad=12)
        fig_vio.tight_layout()
        st.pyplot(fig_vio)
        plt.close(fig_vio)


# =============================================================================
# TAB 4: GEOSPATIAL INTELLIGENCE
# =============================================================================
with tab_geo:
    st.markdown("### 🌍 Global Geospatial Intelligence")
    st.write(
        "Streamlit makes map visualization intuitive. Pass any DataFrame with `lat`/`lon` coordinates "
        "to `st.map` to instantly generate an interactive, web-accelerated global map."
    )
    
    # Filter by hub
    selected_hubs = st.multiselect(
        "Filter by Operational Hub Region:",
        options=list(df_geo["Hub"].unique()),
        default=list(df_geo["Hub"].unique()),
    )
    
    filtered_geo = df_geo[df_geo["Hub"].isin(selected_hubs)]
    
    gcol1, gcol2 = st.columns([2.5, 1.5])
    
    with gcol1:
        st.map(
            filtered_geo,
            latitude="lat",
            longitude="lon",
            size="Activity",
            color="#38bdf8",
            zoom=1,
            use_container_width=True,
        )
        
    with gcol2:
        st.markdown("#### Regional Hub Telemetry")
        summary_geo = filtered_geo.groupby("Hub").agg(
            Nodes=("Active Nodes", "sum"),
            AvgActivity=("Activity", "mean"),
        ).round(1).reset_index()
        
        st.dataframe(
            summary_geo,
            column_config={
                "Nodes": st.column_config.ProgressColumn(
                    "Active Nodes",
                    help="Active cloud nodes",
                    format="%d",
                    min_value=0,
                    max_value=int(summary_geo["Nodes"].max() * 1.2),
                ),
                "AvgActivity": st.column_config.NumberColumn(
                    "Traffic Score",
                    format="%.0f ops/s",
                ),
            },
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# TAB 5: REAL-TIME STREAMING SIMULATION
# =============================================================================
with tab_stream:
    st.markdown("### ⚡ Live In-Place Real-Time Reactive Streaming")
    st.write(
        "One of Streamlit's most beloved superpowers is `st.empty()` and element mutation. "
        "You can dynamically animate metrics, charts, and progress bars without refreshing the browser page."
    )
    
    col_ctrl, col_stats = st.columns([1, 3])
    with col_ctrl:
        stream_cycles = st.slider("Simulation Ticks", min_value=10, max_value=60, value=25, step=5)
        stream_delay = st.slider("Latency Interval (s)", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
        start_button = st.button("▶️ Launch Real-Time Stream", type="primary")

    with col_stats:
        status_box = st.empty()
        kpi_row = st.empty()
        chart_slot = st.empty()
        prog_bar = st.empty()
        
        if start_button:
            status_box.info("📡 Streaming live telemetry data packet feed...")
            stream_df = pd.DataFrame(columns=["Second", "CPU Load (%)", "Memory (MB)", "Requests/sec"])
            
            curr_cpu = 45.0
            for i in range(1, stream_cycles + 1):
                # Random walk
                curr_cpu = float(np.clip(curr_cpu + np.random.normal(0, 3.5), 10, 98))
                curr_mem = 2048 + (curr_cpu * 12) + np.random.randint(-50, 50)
                curr_reqs = int(350 + (curr_cpu * 8) + np.random.randint(-30, 30))
                
                new_row = pd.DataFrame({
                    "Second": [i],
                    "CPU Load (%)": [round(curr_cpu, 1)],
                    "Memory (MB)": [round(curr_mem, 1)],
                    "Requests/sec": [curr_reqs],
                })
                stream_df = pd.concat([stream_df, new_row], ignore_index=True)
                
                # Dynamic KPI updates
                with kpi_row.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Live CPU Utilization", f"{curr_cpu:.1f}%", f"{np.random.normal(0, 2):+.1f}%")
                    m2.metric("Allocated Memory", f"{curr_mem:.0f} MB", "+12 MB")
                    m3.metric("Throughput", f"{curr_reqs} req/s", "+5%")
                    
                # In-place chart update
                chart_slot.line_chart(stream_df.set_index("Second")[["CPU Load (%)", "Requests/sec"]])
                prog_bar.progress(i / stream_cycles, text=f"Stream Progress: {i}/{stream_cycles} ticks")
                
                time.sleep(stream_delay)
                
            status_box.success("✅ Real-time simulation cycle completed successfully!")
        else:
            status_box.info("Press 'Launch Real-Time Stream' to witness dynamic chart animation.")


# =============================================================================
# TAB 6: LIVE DATA EDITOR & TWO-WAY REACTIVITY
# =============================================================================
with tab_editor:
    st.markdown("### 📝 Two-Way Reactive Data Editor (`st.data_editor`)")
    st.write(
        "Edit cell values, toggle checkboxes, or modify budgets directly inside the browser table below. "
        "Watch how the live summary graph reacts **instantaneously** to your inputs!"
    )
    
    # Initial budget dataset
    default_budget_data = pd.DataFrame({
        "Department": ["Research & AI", "Engineering", "Design & UX", "Product Marketing", "Sales & Ops"],
        "Allocated Budget ($k)": [450, 680, 180, 320, 290],
        "Approved": [True, True, True, True, False],
        "Priority Tier": ["Tier 1", "Tier 1", "Tier 2", "Tier 2", "Tier 3"],
    })
    
    ed_col1, ed_col2 = st.columns([1.8, 1.2])
    
    with ed_col1:
        st.markdown("#### Interactive Editable Budget Table")
        edited_df = st.data_editor(
            default_budget_data,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Allocated Budget ($k)": st.column_config.NumberColumn(
                    "Budget ($k)",
                    help="Allocated departmental budget in thousands USD",
                    min_value=0,
                    max_value=2000,
                    step=10,
                ),
                "Approved": st.column_config.CheckboxColumn(
                    "Approved?",
                    help="Is this expenditure sanctioned?",
                    default=True,
                ),
                "Priority Tier": st.column_config.SelectboxColumn(
                    "Tier",
                    options=["Tier 1", "Tier 2", "Tier 3"],
                ),
            },
            key="budget_editor",
        )
        
    with ed_col2:
        st.markdown("#### Reactive Budget Allocation Chart")
        # Filter for approved budget and plot reactively
        approved_items = edited_df[edited_df["Approved"] == True]
        
        if not approved_items.empty:
            fig_react = px.pie(
                approved_items,
                names="Department",
                values="Allocated Budget ($k)",
                color_discrete_sequence=px.colors.sequential.Teal,
                hole=0.45,
                template="plotly_dark",
                title=f"Total Approved: ${approved_items['Allocated Budget ($k)'].sum():,.0f}k",
            )
            fig_react.update_layout(margin=dict(l=10, r=10, b=10, t=40))
            st.plotly_chart(fig_react, use_container_width=True)
        else:
            st.warning("⚠️ No departments currently marked as approved.")


# -----------------------------------------------------------------------------
# 7. RAW DATA INSPECTION & CSV EXPORT
# -----------------------------------------------------------------------------
if show_raw_data:
    st.divider()
    st.subheader("🔍 Deep Data Inspector & Export Facility")
    
    inspector_tab1, inspector_tab2 = st.tabs(["Time Series Performance", "High-Dimensional Scatter Data"])
    
    with inspector_tab1:
        st.dataframe(df_ts, use_container_width=True)
        csv_ts = df_ts.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Time Series Data as CSV",
            data=csv_ts,
            file_name="streamlit_timeseries_data.csv",
            mime="text/csv",
        )
        
    with inspector_tab2:
        st.dataframe(df_scatter, use_container_width=True)
        csv_scatter = df_scatter.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Scatter Analytics as CSV",
            data=csv_scatter,
            file_name="streamlit_scatter_analytics.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# 8. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
        Built with ❤️ using <strong>Streamlit</strong>, <strong>Plotly</strong>, <strong>Seaborn</strong>, 
        and <strong>Pandas</strong>. To start this application, run: <code>streamlit run main.py</code>
    </div>
    """,
    unsafe_allow_html=True,
)

