import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Ensure custom modules from local path can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.api_utils import fetch_coordinates, fetch_historical_data, fetch_forecast_data
from utils.model_utils import train_polynomial_regression, predict_future_temperatures, calculate_metrics
from utils.plot_utils import plot_historical_trend, plot_comparison, plot_error_bar

# 1. Page Configuration
st.set_page_config(
    page_title="Weather Forecast & ML Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Theme Injections
st.markdown("""
    <style>
        /* Base page styling */
        .stApp {
            background-color: #050A18;
            color: #FFFFFF;
        }
        
        /* Custom card style */
        .metric-card {
            background-color: #0B1F3A;
            border: 1px solid #1E90FF;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(30, 144, 255, 0.15);
            text-align: center;
        }
        .metric-title {
            color: #B0BEC5;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .metric-value {
            color: #00BFFF;
            font-size: 32px;
            font-weight: bold;
        }
        .metric-subtitle {
            color: #FFFFFF;
            font-size: 16px;
            margin-top: 5px;
        }
        
        /* Explanation Section Container */
        .info-card {
            background-color: #0B1F3A;
            border-left: 5px solid #00BFFF;
            border-radius: 6px;
            padding: 15px;
            margin-top: 20px;
        }
        
        /* Sidebar customization */
        section[data-testid="stSidebar"] {
            background-color: #0B1F3A;
            border-right: 1px solid #1E90FF;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
            color: #FFFFFF !important;
        }
        
        /* Tab formatting */
        button[data-baseweb="tab"] {
            color: #B0BEC5 !important;
            font-size: 16px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00BFFF !important;
            border-bottom-color: #00BFFF !important;
        }
        
        /* Table styles */
        .dataframe {
            background-color: #0B1F3A !important;
            color: #FFFFFF !important;
            border: 1px solid #1E90FF !important;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Setup
st.sidebar.title("🛠️ Model Controller")
st.sidebar.markdown("Configure the parameters to fetch historical weather data, train the Polynomial Regression model, and compare it with the official forecast.")

# Predefined Locations list as required
locations = [
    "Kolkata",
    "Delhi",
    "Mumbai",
    "Chennai",
    "Bengaluru",
    "Hyderabad",
    "Pune",
    "Jaipur",
    "Ahmedabad",
    "Bhubaneswar"
]

selected_city = st.sidebar.selectbox("📍 Select Location", locations)

# Parameters input
historical_days = st.sidebar.slider(
    "⏳ Historical Training Days",
    min_value=100,
    max_value=730,
    value=400,
    step=50,
    help="Number of days in the past used to train the Polynomial Regression model."
)

forecast_days = st.sidebar.slider(
    "🔮 Forecast Days",
    min_value=1,
    max_value=14,
    value=7,
    step=1,
    help="Number of upcoming days to forecast and compare."
)

poly_degree = st.sidebar.slider(
    "📈 Polynomial Degree",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="Higher degree captures more complex curvatures but might overfit."
)

predict_btn = st.sidebar.button("Train Model & Predict 🚀", use_container_width=True)

# 4. Main App Layout & Header
st.title("⚡ Weather Forecasting & ML Prediction Dashboard")
st.markdown("Compare a **Polynomial Regression Model** forecast with the official professional forecast from the **Open-Meteo API**.")
st.markdown("---")

# Cache coordinate fetching
@st.cache_data
def get_coords(city):
    return fetch_coordinates(city)

# Cache weather data fetching to avoid rate limits
@st.cache_data
def get_weather_data(lat, lon, hist_days, fore_days):
    # Historical archive starts from historical_days ago to yesterday
    end_date = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=hist_days + 2)).strftime('%Y-%m-%d')
    
    df_hist = fetch_historical_data(lat, lon, start_date, end_date)
    df_forecast = fetch_forecast_data(lat, lon, fore_days)
    
    return df_hist, df_forecast

# Active workflow execution trigger
if predict_btn or 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    
    with st.spinner("Fetching geolocation coordinates..."):
        loc_data = get_coords(selected_city)
        
    if loc_data is None:
        st.error(f"❌ Error: Unable to fetch coordinates for {selected_city}. Please try again later.")
    else:
        lat, lon = loc_data["latitude"], loc_data["longitude"]
        country, timezone = loc_data["country"], loc_data["timezone"]
        
        with st.spinner("Retrieving historical records & forecasting data..."):
            df_hist, df_forecast = get_weather_data(lat, lon, historical_days, forecast_days)
            
        if df_hist is None or len(df_hist) == 0:
            st.error("❌ Error: Unable to load historical weather data. The Archive API may be temporarily unavailable.")
        elif df_forecast is None or len(df_forecast) == 0:
            st.error("❌ Error: Unable to fetch current weather forecast data.")
        else:
            # Model training & fitting
            with st.spinner("Training Polynomial Regression model..."):
                model, X_train, y_train, df_hist_fitted = train_polynomial_regression(df_hist, degree=poly_degree)
                
                # Predict upcoming forecast days
                start_idx = len(df_hist) + 1
                pred_results = predict_future_temperatures(model, start_idx, len(df_forecast))
                
                # Link predictions back to forecast data
                df_comparison = df_forecast.copy()
                df_comparison["predicted_temp"] = pred_results["predictions"]
                df_comparison["difference"] = (df_comparison["forecast_mean"] - df_comparison["predicted_temp"]).abs()
                df_comparison["error_pct"] = (df_comparison["difference"] / df_comparison["forecast_mean"] * 100)
                
                # Metric evaluations
                metrics = calculate_metrics(df_comparison)

            # Metadata Display Container
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Location Info</div>
                        <div class="metric-subtitle"><b>{selected_city}, {country}</b></div>
                        <div class="metric-subtitle" style="font-size: 13px; color: #B0BEC5;">Lat: {lat}° | Lon: {lon}°</div>
                    </div>
                """, unsafe_allow_html=True)
            with meta_col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Average Forecast Error</div>
                        <div class="metric-value">{metrics['avg_error']}%</div>
                        <div class="metric-subtitle" style="font-size: 13px; color: #B0BEC5;">Target: &le; 5.0% for Good</div>
                    </div>
                """, unsafe_allow_html=True)
            with meta_col3:
                # Colorize prediction quality
                q_color = "#2ECC71" if metrics['quality'] == "Good" else ("#FFA500" if metrics['quality'] == "Moderate" else "#FF6B6B")
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Prediction Quality</div>
                        <div class="metric-value" style="color: {q_color};">{metrics['quality']}</div>
                        <div class="metric-subtitle" style="font-size: 13px; color: #B0BEC5;">Based on average error</div>
                    </div>
                """, unsafe_allow_html=True)
            with meta_col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Error Range</div>
                        <div class="metric-value" style="font-size: 26px;">{metrics['min_error']}% - {metrics['max_error']}%</div>
                        <div class="metric-subtitle" style="font-size: 13px; color: #B0BEC5;">Min Error to Max Error</div>
                    </div>
                """, unsafe_allow_html=True)

            # Visualization Tabs
            tab1, tab2, tab3 = st.tabs([
                "📊 Historical Model Fit", 
                "🔮 Forecast Comparison", 
                "📈 Daily Error Breakdown"
            ])
            
            with tab1:
                st.plotly_chart(plot_historical_trend(df_hist_fitted), use_container_width=True)
                
            with tab2:
                st.plotly_chart(plot_comparison(df_comparison), use_container_width=True)
                
            with tab3:
                st.plotly_chart(plot_error_bar(df_comparison), use_container_width=True)

            # Data Table & Explanation Section
            col_table, col_explain = st.columns([3, 2])
            
            with col_table:
                st.subheader("📋 Detailed Comparison Table")
                # Format dataframe for presentation
                df_presentation = df_comparison.copy()
                df_presentation["date"] = df_presentation["date"].dt.strftime('%Y-%m-%d')
                df_presentation = df_presentation.rename(columns={
                    "date": "Date",
                    "forecast_max": "Official Max (°C)",
                    "forecast_min": "Official Min (°C)",
                    "forecast_mean": "Official Mean (°C)",
                    "predicted_temp": "Model Predicted (°C)",
                    "difference": "Difference (°C)",
                    "error_pct": "Error (%)"
                })
                
                # Apply rounding for cleaner display
                for col in df_presentation.columns:
                    if col != "Date":
                        df_presentation[col] = df_presentation[col].round(2)
                
                # Render formatted table
                st.dataframe(df_presentation, use_container_width=True, hide_index=True)
                
            with col_explain:
                st.subheader("💡 Model Interpretation")
                st.markdown(f"""
                    <div class="info-card">
                        <h4>Polynomial Regression (Degree {poly_degree})</h4>
                        <p>
                            Weather patterns are highly curved and non-linear due to seasonal variations. 
                            A Polynomial Regression model maps the training timeline into high-dimensional space 
                            to draw curved trends rather than a straight line.
                        </p>
                        <p>
                            <b>Limitations of this model:</b>
                            <ul>
                                <li>The regression is <i>univariate</i>: it only uses the day sequence and historical temperature to forecast.</li>
                                <li>It lacks understanding of atmosphere metrics like wind speed, humidity, cloud coverage, and pressure.</li>
                                <li>Polynomial bounds can lead to runaway estimations (unbounded growth or decline) for long-term forecasts.</li>
                            </ul>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
st.markdown("---")
st.markdown("<p style='text-align: center; color: #B0BEC5;'>Weather Forecasting App &copy; 2026. Made using Streamlit & Scikit-learn.</p>", unsafe_allow_html=True)
