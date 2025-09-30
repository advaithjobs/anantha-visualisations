import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_plotly_events import plotly_events
import datetime
from scipy.interpolate import griddata
from plotly.subplots import make_subplots
import scipy.stats as stats
from prophet import Prophet
from prophet.plot import plot_plotly
# --- Page Configuration ---
st.set_page_config(
    page_title="ARGO Float Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Clean Dark Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    .stApp {
        background-color: #0E1117;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #FAFAFA;
        font-weight: 600;
    }
    
    .st-bb {
        background-color: transparent;
    }
    
    .st-at {
        background-color: #262730;
    }
    
    .css-18e3th9 {
        padding: 2rem 5rem;
    }
    
    .css-1d391kg {
        background-color: #262730;
    }
    
    .stSelectbox, .stSlider, .stTextInput {
        background-color: #262730;
        border-radius: 5px;
        padding: 10px;
        color: white;
        border: 1px solid #393A46;
    }
    
    .stMetric {
        background-color: #262730;
        border: 1px solid #393A46;
        border-radius: 5px;
        padding: 15px;
    }
    
    div[data-testid="stMetricValue"] {
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #DADADA;
        font-family: 'Inter', sans-serif;
    }
    
    .data-container {
        background: #262730;
        border: 1px solid #393A46;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .sidebar .sidebar-content {
        background-color: #262730;
        border-right: 1px solid #393A46;
    }
    
    .header {
        border-bottom: 1px solid #393A46;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    .parameter-card {
        background: #262730;
        border: 1px solid #393A46;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .tab-container {
        background: #262730;
        border: 1px solid #393A46;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        border: 1px solid #393A46;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0E1117;
        border-bottom: 1px solid #0E1117;
    }
</style>
""", unsafe_allow_html=True)


# --- Anantha Banner Header ---


# --- Data Loading (Cached) ---
# --- Data Loading (Cached) ---
@st.cache_data
def load_argo_data(folder_path=None):
    if folder_path:
        search_pattern = os.path.join(folder_path, '**', '*.csv')
        all_files = glob.glob(search_pattern, recursive=True)
        if all_files:
            list_of_dfs = [pd.read_csv(file) for file in all_files if os.path.getsize(file) > 0]
            if list_of_dfs:
                combined_df = pd.concat(list_of_dfs, ignore_index=True)
                combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='mixed', errors='coerce')
                combined_df.dropna(subset=['Date'], inplace=True)
                return combined_df
    # If folder path invalid or empty, fallback to demo CSV
    demo_csv_path = "./2900537_profiles.csv"  # your demo CSV in repo
    if os.path.exists(demo_csv_path):
        df_demo = pd.read_csv(demo_csv_path)
        df_demo['Date'] = pd.to_datetime(df_demo['Date'], errors='coerce')
        df_demo.dropna(subset=['Date'], inplace=True)
        return df_demo
    return pd.DataFrame()  # empty if nothing found

# --- Function to create 3D surface plots ---
def create_3d_surface(df, z_value, title, color_scale='Viridis'):
    # Create grid for surface plot
    float_ids = df['Float_ID'].unique()
    
    # For simplicity, let's focus on one float for the 3D surface
    if len(float_ids) > 0:
        sample_float = float_ids[0]
        float_data = df[df['Float_ID'] == sample_float]
        
        # Create a grid for the surface
        lat_unique = np.linspace(float_data['Latitude'].min(), float_data['Latitude'].max(), 50)
        lon_unique = np.linspace(float_data['Longitude'].min(), float_data['Longitude'].max(), 50)
        lon_grid, lat_grid = np.meshgrid(lon_unique, lat_unique)
        
        # Interpolate values onto grid
        points = float_data[['Longitude', 'Latitude']].values
        values = float_data[z_value].values
        z_grid = griddata(points, values, (lon_grid, lat_grid), method='linear')
        
        # Create the surface plot
        fig = go.Figure(data=[go.Surface(
            z=z_grid,
            x=lon_grid,
            y=lat_grid,
            colorscale=color_scale,
            opacity=0.9,
            hoverinfo='none',
            contours = {
                "z": {"show": True, "start": z_grid.min(), "end": z_grid.max(), "size": (z_grid.max()-z_grid.min())/10, "width": 1}
            }
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title=z_value,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=500,
        )
        
        return fig
    return go.Figure()

# --- Function to create time series decomposition ---
def create_decomposition_plot(df, parameter, float_id):
    float_data = df[df['Float_ID'] == float_id].sort_values('Date')
    
    if len(float_data) < 10:  # Need enough data points
        return None
        
    # Resample to regular intervals
    float_data = float_data.set_index('Date')
    float_data = float_data.resample('D').mean().ffill()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original Series', 'Trend', 'Seasonality', 'Residuals'),
        vertical_spacing=0.08
    )
    
    # Original series
    fig.add_trace(
        go.Scatter(x=float_data.index, y=float_data[parameter], name='Original'),
        row=1, col=1
    )
    
    # Calculate rolling mean for trend
    trend = float_data[parameter].rolling(window=7, center=True).mean()
    fig.add_trace(
        go.Scatter(x=float_data.index, y=trend, name='Trend'),
        row=2, col=1
    )
    
    # Calculate seasonality (difference between original and trend)
    detrended = float_data[parameter] - trend
    seasonal = detrended.rolling(window=7, center=True).mean()
    fig.add_trace(
        go.Scatter(x=float_data.index, y=seasonal, name='Seasonality'),
        row=3, col=1
    )
    
    # Calculate residuals
    residuals = detrended - seasonal
    fig.add_trace(
        go.Scatter(x=float_data.index, y=residuals, name='Residuals'),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

# --- Function to create correlation matrix ---
def create_correlation_matrix(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Parameter Correlation Matrix'
    )
    fig.update_layout(height=600)
    return fig

# --- Main App ---

from PIL import Image
banner = Image.open("banner.png")
st.image("banner.png",width="stretch")


st.markdown("<div class='header'>", unsafe_allow_html=True)
st.title("Anantha AI Visualisations & Research Tools ")
st.markdown("Comprehensive analysis and visualization of oceanic data from ARGO floats")
st.markdown("</div>", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("### Data Configuration")
    
    data_folder = st.text_input(
        "Data Directory Path (optional, leave empty for demo):",
        ""
    )
    

    st.markdown("### Visualization Settings")
    
    
    st.markdown("---")
    st.markdown("### Visualization Settings")
    
    color_theme = st.selectbox(
        "Color Theme:",
        ["Viridis", "Plasma", "Inferno", "Magma"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This application visualizes and analyzes data collected by ARGO floats deployed across the world's oceans.")

# Load data
df = load_argo_data(data_folder)

if df.empty:
    st.error("No data found. Please check the folder path.")
else:
    st.success(f"Data loaded successfully. Contains {df['Float_ID'].nunique()} floats and {len(df):,} measurements.")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 , tab7, tab8, tab9 = st.tabs([
        "Global Overview", 
        "Float Analysis", 
        "Time Series", 
        "Correlations", 
        "Distributions",
        "Anomaly Detection",
        "Drift Analysis",
        "Time Series Forecasting",
        "Data Interploation"
    ])
    
    with tab1:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Global Data Overview")
        st.info("Explore the overall distribution of floats across oceans, view key metrics like temperature, salinity, and pressure, and visualize global trends on an interactive map.")


        # Date selection
        available_dates = sorted(df['Date'].dt.date.unique())
        selected_date = st.select_slider(
            "Select Observation Date:",
            options=available_dates,
            value=available_dates[len(available_dates)//2],
            format_func=lambda date: date.strftime('%Y-%m-%d')
        )

        # Filter data for selected date
        df_on_date = df[df['Date'].dt.date == selected_date].copy()
        df_on_date.dropna(subset=['Latitude', 'Longitude'], inplace=True)

        if df_on_date.empty:
            st.info("No float data with valid coordinates available for the selected date.")
        else:
            map_display_df = df_on_date.drop_duplicates(subset=['Float_ID'])
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Floats", len(map_display_df))
            with col2:
                st.metric("Avg Temperature", f"{map_display_df['Temp_adj(C)'].mean():.2f} °C")
            with col3:
                st.metric("Avg Salinity", f"{map_display_df['Psal_adj(psu)'].mean():.2f} psu")
            with col4:
                st.metric("Avg Pressure", f"{map_display_df['Pres_adj(dbar)'].mean():.2f} dbar")
            
            st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
            st.subheader("Global Float Distribution")

            map_display_df_clean = map_display_df.dropna(subset=['Pres_adj(dbar)'])
            map_display_df_clean['Marker_Size'] = map_display_df_clean['Pres_adj(dbar)'].clip(lower=1)

            # Parameter selector for glowing map
            map_type = st.radio("Select parameter to visualize:", 
                                ["Temperature", "Salinity", "Pressure"], horizontal=True)
            if map_type == "Temperature":
                color_param = "Temp_adj(C)"
                title = "Temperature Distribution (°C)"
                colorscale = "Viridis"
            elif map_type == "Salinity":
                color_param = "Psal_adj(psu)"
                title = "Salinity Distribution (psu)"
                colorscale = "Cividis"
            else:
                color_param = "Pres_adj(dbar)"
                title = "Pressure Distribution (dbar)"
                colorscale = "Plasma"

            # Scatter mapbox with dark theme and glow
            fig_map = px.scatter_map(
                map_display_df_clean,
                lat="Latitude",
                lon="Longitude",
                color=color_param,
                size="Marker_Size",
                hover_name="Float_ID",
                hover_data={"Latitude": True, "Longitude": True, color_param: True},
                color_continuous_scale=colorscale,
                size_max=15,
                zoom=1,
            )

            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0)
            )

            # Glow effect by adding semi-transparent markers
            fig_map.add_trace(go.Scattermap(
                lat=map_display_df_clean["Latitude"],
                lon=map_display_df_clean["Longitude"],
                mode='markers',
                marker=go.scattermap.Marker(
                    size=map_display_df_clean['Marker_Size'] * 1.5,
                    color=map_display_df_clean[color_param],
                    opacity=0.3,
                    colorscale=colorscale,
                    showscale=False
                ),
                hoverinfo='skip'
            ))

            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    
    with tab2:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Individual Float Analysis")
        st.info("Dive into individual float data, check depth profiles, latest measurements, and examine the float’s behavior at different depths.")

        
        if not df_on_date.empty:
            available_floats = map_display_df['Float_ID'].unique()
            selected_float = st.selectbox("Select Float ID:", available_floats)
            
            # Get data for selected float
            float_data = df[df['Float_ID'] == selected_float]
            current_data = df_on_date[df_on_date['Float_ID'] == selected_float]
            
            if not current_data.empty:
                latest_data = current_data.iloc[0]
                
                # Display float info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("First Detection", float_data['Date'].min().strftime('%Y-%m-%d'))
                with col2:
                    st.metric("Measurements Count", len(float_data))
                with col3:
                    st.metric("Depth Range", f"{float_data['Pres_adj(dbar)'].min():.0f} - {float_data['Pres_adj(dbar)'].max():.0f} dbar")
                
                # Depth profile
                st.subheader("Depth Profile")
                
                depth_profile = current_data.sort_values('Pres_adj(dbar)')
                
                fig_profile = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Temperature Profile', 'Salinity Profile')
                )
                
                fig_profile.add_trace(
                    go.Scatter(
                        x=depth_profile['Temp_adj(C)'],
                        y=depth_profile['Pres_adj(dbar)'],
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='#FF6B6B')
                    ),
                    row=1, col=1
                )
                
                fig_profile.add_trace(
                    go.Scatter(
                        x=depth_profile['Psal_adj(psu)'],
                        y=depth_profile['Pres_adj(dbar)'],
                        mode='lines+markers',
                        name='Salinity',
                        line=dict(color='#4ECDC4')
                    ),
                    row=1, col=2
                )
                
                fig_profile.update_yaxes(title_text="Pressure (dbar)", row=1, col=1, autorange="reversed")
                fig_profile.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
                fig_profile.update_xaxes(title_text="Salinity (psu)", row=1, col=2)
                fig_profile.update_layout(height=400)
                
                st.plotly_chart(fig_profile, use_container_width=True)
                
                # Parameter cards
                st.subheader("Current Measurements")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='parameter-card'>", unsafe_allow_html=True)
                    st.metric("Temperature", f"{latest_data['Temp_adj(C)']:.2f} °C")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='parameter-card'>", unsafe_allow_html=True)
                    st.metric("Salinity", f"{latest_data['Psal_adj(psu)']:.2f} psu")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='parameter-card'>", unsafe_allow_html=True)
                    st.metric("Pressure", f"{latest_data['Pres_adj(dbar)']:.0f} dbar")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            else:
                st.info("No data available for the selected float on this date.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Time Series Analysis")

        st.info("Analyze temporal trends of temperature, salinity, or pressure for selected floats, view trendlines, and decompose the series into trend, seasonality, and residuals.")

        
        if not df.empty:
            available_floats = df['Float_ID'].unique()
            selected_float_ts = st.selectbox("Select Float ID for time series:", available_floats)
            
            float_data_ts = df[df['Float_ID'] == selected_float_ts].sort_values('Date')
            
            if len(float_data_ts) > 1:
                # Resample for cleaner visualization
                ts_data = float_data_ts.set_index('Date')
                ts_data = ts_data.resample('W').mean().reset_index()
                
                parameter = st.selectbox("Select parameter:", 
                                       ["Temperature", "Salinity", "Pressure"])
                
                if parameter == "Temperature":
                    y_value = "Temp_adj(C)"
                    title = "Temperature Over Time"
                    color = "#FF6B6B"
                elif parameter == "Salinity":
                    y_value = "Psal_adj(psu)"
                    title = "Salinity Over Time"
                    color = "#4ECDC4"
                else:
                    y_value = "Pres_adj(dbar)"
                    title = "Pressure Over Time"
                    color = "#45B7D1"
                
                fig_ts = px.line(
                    ts_data, 
                    x='Date', 
                    y=y_value,
                    title=title,
                    color_discrete_sequence=[color]
                )
                
                # Add trendline
                z = np.polyfit(range(len(ts_data)), ts_data[y_value], 1)
                p = np.poly1d(z)
                fig_ts.add_trace(go.Scatter(
                    x=ts_data['Date'],
                    y=p(range(len(ts_data))),
                    name='Trend',
                    line=dict(color='white', dash='dash')
                ))
                
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Decomposition plot
                st.subheader("Time Series Decomposition")
                decomp_fig = create_decomposition_plot(df, y_value, selected_float_ts)
                
                if decomp_fig:
                    st.plotly_chart(decomp_fig, use_container_width=True)
                else:
                    st.info("Not enough data points for decomposition analysis.")
            else:
                st.info("Not enough data points for time series analysis.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Parameter Correlations")
        
        st.info("Study correlations between parameters using scatter plots, scatter matrix, and interactive 3D visualizations to understand relationships among variables.")

        if not df.empty:
            # Scatter matrix
            max_rows = 2000  # limit rows to avoid MessageSizeError
            scatter_df = df.sample(n=min(len(df), max_rows), random_state=42)

            fig_matrix = px.scatter_matrix(
                scatter_df,
                dimensions=['Temp_adj(C)', 'Psal_adj(psu)', 'Pres_adj(dbar)'],
                color='Temp_adj(C)',
                title="Scatter Matrix of Selected Variables (sampled)"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

            # Correlation matrix
            st.subheader("Correlation Matrix")
            st.info("If there are any missing values , they are given a void space in the correlation matrix")
            corr_fig = create_correlation_matrix(df)
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Interactive 3D scatter

        st.markdown("</div>", unsafe_allow_html=True)

        with tab5:
            st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
            st.header("Parameter Distributions")
            st.info("Inspect the distributions of temperature, salinity, and pressure using histograms, box plots, and violin plots for different depth ranges.")



            if not df.empty:
                # Sample dataset to reduce size for plotting
                sample_size = min(5000, len(df))  # limit to 5000 rows
                df_sample = df.sample(sample_size, random_state=42)
                df_depth_bins = df_sample.copy()
                df_depth_bins['Depth Range'] = pd.cut(
                    df_depth_bins['Pres_adj(dbar)'], 
                    bins=5, 
                    labels=['Shallow', 'Mid', 'Deep', 'Very Deep', 'Abyssal']
                )

                col1, col2 = st.columns(2)

                with col1:
                    fig_temp_dist = px.histogram(
                        df_sample, 
                        x='Temp_adj(C)',
                        title='Temperature Distribution',
                        nbins=30,
                        color_discrete_sequence=['#FF6B6B']
                    )
                    st.plotly_chart(fig_temp_dist, use_container_width=True)

                    fig_box = px.box(
                        df_depth_bins,
                        x='Depth Range',
                        y='Temp_adj(C)',
                        title='Temperature Distribution by Depth Range',
                        color='Depth Range',
                        color_discrete_sequence=px.colors.sequential.Plasma_r
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

                with col2:
                    fig_sal_dist = px.histogram(
                        df_sample, 
                        x='Psal_adj(psu)',
                        title='Salinity Distribution',
                        nbins=30,
                        color_discrete_sequence=['#4ECDC4']
                    )
                    st.plotly_chart(fig_sal_dist, use_container_width=True)

                    fig_violin = px.violin(
                        df_depth_bins,
                        x='Depth Range',
                        y='Psal_adj(psu)',
                        title='Salinity Distribution by Depth Range',
                        color='Depth Range',
                        color_discrete_sequence=px.colors.sequential.Plasma_r
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            
    with tab6:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Advanced Analysis")
        

        st.info("Perform statistical analysis, detect anomalies based on z-scores, and export either the current view or the full dataset for further analysis.")

        if not df.empty:
            # Statistical summary
            st.subheader("Statistical Summary")
            
            numeric_df = df.select_dtypes(include=[np.number])
            st.dataframe(numeric_df.describe())
            
            # Anomaly detection
            st.subheader("Anomaly Detection")
            
            parameter_anomaly = st.selectbox("Select parameter for anomaly detection:", 
                                           ["Temp_adj(C)", "Psal_adj(psu)", "Pres_adj(dbar)"])
            
            # Calculate z-scores
            # Calculate z-scores while keeping alignment
            # Calculate z-scores only for the selected parameter
            param_series = df[parameter_anomaly].dropna()
            z_scores = np.abs(stats.zscore(param_series))

            threshold = st.slider("Anomaly threshold (z-score):", 1.0, 5.0, 3.0, 0.5)

            # Align indexes correctly
            anomaly_idx = param_series.index[z_scores > threshold]
            anomalies = df.loc[anomaly_idx, ['Date', parameter_anomaly]]

            st.metric("Anomalies detected", f"{len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")

            if not anomalies.empty:
                # Sample both normal and anomaly points for plotting
                sample_size = min(5000, len(df))  # limit total size
                plot_df = df.sample(sample_size, random_state=42).copy()
                plot_df["is_anomaly"] = False
                plot_df.loc[anomalies.index.intersection(plot_df.index), "is_anomaly"] = True

                fig_anomalies = px.scatter(
                    plot_df,
                    x="Date",
                    y=parameter_anomaly,
                    color="is_anomaly",
                    title=f"Anomalies in {parameter_anomaly} (Z-score > {threshold})",
                    color_discrete_map={False: "#6A0572", True: "red"}
                )
                st.plotly_chart(fig_anomalies, use_container_width=True)

            
            # Data export
            st.subheader("Data Export")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Current View Data"):
                    csv = df_on_date.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"argo_data_{selected_date}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("Export Full Dataset"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="argo_full_dataset.csv",
                        mime="text/csv"
                    )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab7:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Float Drift Analysis")
        st.info("Visualize the movement trajectory of individual floats over time on a geospatial map.")
        
        if not df.empty:
            available_floats = df['Float_ID'].unique()
            selected_float_drift = st.selectbox("Select Float ID for drift analysis:", available_floats)
            
            float_data_drift = df[df['Float_ID'] == selected_float_drift].sort_values('Date')
            float_data_drift.dropna(subset=['Latitude', 'Longitude'], inplace=True)
            
            if not float_data_drift.empty:
                # Map showing float trajectory
                fig_drift = px.line_mapbox(
                    float_data_drift,
                    lat="Latitude",
                    lon="Longitude",
                    hover_name="Date",
                    hover_data={"Latitude": True, "Longitude": True, "Pres_adj(dbar)": True},
                    color_discrete_sequence=["#FF6B6B"],
                    zoom=2,
                    height=500
                )
                # Add start and end markers
                # Add start marker
                fig_drift.add_trace(go.Scattermapbox(
                    lat=[float_data_drift['Latitude'].iloc[0]],
                    lon=[float_data_drift['Longitude'].iloc[0]],
                    mode='markers',
                    marker=dict(size=12, color='green'),
                    text=["Start"],
                    hoverinfo="text"
                ))

                # Add end marker
                fig_drift.add_trace(go.Scattermapbox(
                    lat=[float_data_drift['Latitude'].iloc[-1]],
                    lon=[float_data_drift['Longitude'].iloc[-1]],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    text=["End"],
                    hoverinfo="text"
                ))

                
                fig_drift.update_layout(
                    mapbox_style="carto-darkmatter",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_drift, use_container_width=True)
                
                # Show summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Drift Points", len(float_data_drift))
                with col2:
                    lat_range = float_data_drift['Latitude'].max() - float_data_drift['Latitude'].min()
                    st.metric("Latitude Range", f"{lat_range:.2f}°")
                with col3:
                    lon_range = float_data_drift['Longitude'].max() - float_data_drift['Longitude'].min()
                    st.metric("Longitude Range", f"{lon_range:.2f}°")
            else:
                st.info("No latitude and longitude data available for this float.")
        else:
            st.info("No data loaded.")
        
        st.markdown("</div>", unsafe_allow_html=True)


    with tab8:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Forecast Future Values")
        st.info("Forecast future temperature, salinity, or pressure of selected floats using time series models.")

        if not df.empty:
            available_floats = df['Float_ID'].unique()
            selected_float_forecast = st.selectbox("Select Float ID for forecasting:", available_floats)

            float_data_forecast = df[df['Float_ID'] == selected_float_forecast].sort_values('Date')
            float_data_forecast.dropna(subset=['Date'], inplace=True)

            if len(float_data_forecast) > 10:
                parameter = st.selectbox("Select parameter to forecast:", 
                                        ["Temperature", "Salinity", "Pressure"])

                if parameter == "Temperature":
                    y_col = "Temp_adj(C)"
                    color = "#FF6B6B"
                elif parameter == "Salinity":
                    y_col = "Psal_adj(psu)"
                    color = "#4ECDC4"
                else:
                    y_col = "Pres_adj(dbar)"
                    color = "#45B7D1"

                forecast_period = st.slider("Select forecast horizon (days):", 7, 365, 30, 7)

                # Prepare data for Prophet
                ts_df = float_data_forecast[['Date', y_col]].rename(columns={'Date': 'ds', y_col: 'y'})
                
                model = Prophet(daily_seasonality=True, yearly_seasonality=True)
                model.fit(ts_df)

                future = model.make_future_dataframe(periods=forecast_period)
                forecast = model.predict(future)

                # --- Graph 1: Prophet default interactive forecast ---
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(
                    title=f"{parameter} Forecast for Float {selected_float_forecast}",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=600
                )
                st.plotly_chart(fig_forecast, use_container_width=True)

                # --- Graph 2: Overlay historical + forecast values ---
                import plotly.graph_objects as go

                fig2 = go.Figure()

                # Historical data
                fig2.add_trace(go.Scatter(
                    x=ts_df['ds'],
                    y=ts_df['y'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color=color),
                ))

                # Forecasted data
                fig2.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='yellow', dash='dash'),
                ))

                # Confidence interval
                fig2.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(255, 255, 0, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='Confidence Interval'
                ))

                fig2.update_layout(
                    title=f"{parameter} Historical + Forecast for Float {selected_float_forecast}",
                    xaxis_title="Date",
                    yaxis_title=parameter,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500
                )

                st.subheader("Historical vs Forecasted Values")
                st.plotly_chart(fig2, use_container_width=True)

                # Show forecast table
                st.subheader("Forecasted Values (next days)")
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
                forecast_display.rename(columns={
                    'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'
                }, inplace=True)
                st.dataframe(forecast_display.style.format({
                    'Forecast': '{:.2f}', 'Lower Bound': '{:.2f}', 'Upper Bound': '{:.2f}'
                }))
            else:
                st.info("Not enough data points for forecasting (minimum 10 required).")
        else:
            st.info("No data loaded.")

            st.markdown("</div>", unsafe_allow_html=True)

    with tab9:
        st.markdown("<div class='tab-container'>", unsafe_allow_html=True)
        st.header("Gap Filling / Missing Data Interpolation")
        st.info("Visualize original time series with missing values and see how interpolation fills them.")

        if not df.empty:
            available_floats = df['Float_ID'].unique()
            selected_float_gap = st.selectbox("Select Float ID for gap filling:", available_floats)

            float_data_gap = df[df['Float_ID'] == selected_float_gap].sort_values('Date')
            float_data_gap.dropna(subset=['Date'], inplace=True)

            if len(float_data_gap) > 5:
                parameter = st.selectbox("Select parameter to interpolate:", 
                                        ["Temperature", "Salinity", "Pressure"])

                if parameter == "Temperature":
                    y_col = "Temp_adj(C)"
                    color_original = "#FF6B6B"
                    color_filled = "#FFD93D"
                elif parameter == "Salinity":
                    y_col = "Psal_adj(psu)"
                    color_original = "#4ECDC4"
                    color_filled = "#FFD93D"
                else:
                    y_col = "Pres_adj(dbar)"
                    color_original = "#45B7D1"
                    color_filled = "#FFD93D"

                # Prepare time series
                ts_gap = float_data_gap[['Date', y_col]].set_index('Date').sort_index()
                
                # Count missing before
                missing_before = ts_gap[y_col].isna().sum()

                # Interpolation
                ts_filled = ts_gap.interpolate(method='linear', limit_direction='both')

                missing_after = ts_filled[y_col].isna().sum()

                # Find interpolated points
                interpolated_mask = ts_gap[y_col].isna()

                # Plot
                import plotly.graph_objects as go
                fig_gap = go.Figure()

                # Original data (gaps visible)
                fig_gap.add_trace(go.Scatter(
                    x=ts_gap.index,
                    y=ts_gap[y_col],
                    mode='markers+lines',
                    name='Original',
                    line=dict(color=color_original),
                    marker=dict(size=6, symbol='circle')
                ))

                # Filled data
                fig_gap.add_trace(go.Scatter(
                    x=ts_filled.index,
                    y=ts_filled[y_col],
                    mode='lines',
                    name='Filled',
                    line=dict(color=color_filled, width=2, dash='dash')
                ))

                # Highlight interpolated points
                fig_gap.add_trace(go.Scatter(
                    x=ts_filled.index[interpolated_mask],
                    y=ts_filled[y_col][interpolated_mask],
                    mode='markers',
                    name='Interpolated Points',
                    marker=dict(size=8, color='yellow', symbol='x')
                ))

                fig_gap.update_layout(
                    title=f"{parameter} Gap Filling for Float {selected_float_gap}",
                    xaxis_title="Date",
                    yaxis_title=parameter,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=500
                )

                st.plotly_chart(fig_gap, use_container_width=True)
                st.markdown(f"**Missing values before interpolation:** {missing_before}")
                st.markdown(f"**Missing values after interpolation:** {missing_after}")

                # Export filled data
                if st.button("Export Filled Data for This Float"):
                    export_df = float_data_gap.copy()
                    export_df[y_col] = ts_filled[y_col].values
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"float_{selected_float_gap}_filled.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Not enough data points for gap filling (minimum 5 required).")
        else:
            st.info("No data loaded.")

        st.markdown("</div>", unsafe_allow_html=True)



# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #DADADA;'>ARGO Float Data Explorer | Advanced Oceanographic Analysis</p>", 
    unsafe_allow_html=True
)
