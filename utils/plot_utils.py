import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Design Theme Constants
BG_COLOR = "#050A18"
CARD_BG_COLOR = "#0B1F3A"
PRIMARY_BLUE = "#1E90FF"
SECONDARY_BLUE = "#00BFFF"
TEXT_COLOR = "#FFFFFF"
MUTED_TEXT = "#B0BEC5"
ERROR_HIGHLIGHT = "#FF6B6B"
SUCCESS_HIGHLIGHT = "#2ECC71"

def apply_chart_theme(fig):
    """
    Applies custom blue and black styling parameters to a Plotly figure.
    """
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, family="Inter, Outfit, sans-serif"),
        margin=dict(t=40, b=40, l=40, r=40),
        xaxis=dict(
            gridcolor="#112244",
            linecolor="#1A3A66",
            zeroline=False,
            tickfont=dict(color=MUTED_TEXT)
        ),
        yaxis=dict(
            gridcolor="#112244",
            linecolor="#1A3A66",
            zeroline=False,
            tickfont=dict(color=MUTED_TEXT)
        ),
        legend=dict(
            bgcolor="rgba(11, 31, 58, 0.7)",
            bordercolor="#1E90FF",
            borderwidth=1,
            font=dict(color=TEXT_COLOR)
        )
    )
    return fig

def plot_historical_trend(df_history):
    """
    Plots historical temperatures (Max, Min, Mean) and overlays the polynomial regression fitted curve.
    """
    fig = go.Figure()

    # Historical Mean
    fig.add_trace(go.Scatter(
        x=df_history["date"],
        y=df_history["temp_mean"],
        mode="lines",
        name="Historical Mean Temp",
        line=dict(color=SECONDARY_BLUE, width=2)
    ))

    # Fitted Curve
    if "temp_fitted" in df_history.columns:
        fig.add_trace(go.Scatter(
            x=df_history["date"],
            y=df_history["temp_fitted"],
            mode="lines",
            name="Model Fitted Trend (Poly)",
            line=dict(color=PRIMARY_BLUE, width=3, dash="dash")
        ))
        
    # Range Area (Max/Min)
    fig.add_trace(go.Scatter(
        x=df_history["date"],
        y=df_history["temp_max"],
        mode="lines",
        name="Max Temp",
        line=dict(color="#FF8C00", width=1, dash="dot"),
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatter(
        x=df_history["date"],
        y=df_history["temp_min"],
        mode="lines",
        name="Min Temp",
        line=dict(color="#32CD32", width=1, dash="dot"),
        opacity=0.6
    ))

    fig.update_layout(
        title="Historical Temperature Trend & Polynomial Fit",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)"
    )
    
    return apply_chart_theme(fig)

def plot_comparison(df_comparison):
    """
    Plots a direct comparison between the Official Weather Forecast and the Polynomial Regression predictions.
    """
    fig = go.Figure()

    # Official Forecast Mean Temp
    fig.add_trace(go.Scatter(
        x=df_comparison["date"],
        y=df_comparison["forecast_mean"],
        mode="lines+markers",
        name="Official Forecast (Open-Meteo)",
        line=dict(color=SUCCESS_HIGHLIGHT, width=3),
        marker=dict(size=8)
    ))

    # Model Predicted Temp
    fig.add_trace(go.Scatter(
        x=df_comparison["date"],
        y=df_comparison["predicted_temp"],
        mode="lines+markers",
        name="Model Prediction (Poly Regression)",
        line=dict(color=PRIMARY_BLUE, width=3),
        marker=dict(size=8, symbol="diamond")
    ))

    fig.update_layout(
        title="Official Weather Forecast vs. ML Model Prediction",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)"
    )
    
    return apply_chart_theme(fig)

def plot_error_bar(df_comparison):
    """
    Plots a bar chart showing the error percentage for each predicted date.
    """
    # Calculate error values for plotting
    errors = (abs(df_comparison["forecast_mean"] - df_comparison["predicted_temp"]) / df_comparison["forecast_mean"] * 100).round(2)
    dates = df_comparison["date"].dt.strftime('%b %d')

    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dates,
        y=errors,
        text=errors.astype(str) + "%",
        textposition="auto",
        marker=dict(
            color=errors,
            colorscale=[[0, "#2ECC71"], [0.5, "#E67E22"], [1, "#FF6B6B"]],
            showscale=False
        ),
        hovertemplate="Date: %{x}<br>Error: %{y}%<extra></extra>"
    ))

    fig.update_layout(
        title="Daily Prediction Error Percentage",
        xaxis_title="Forecast Date",
        yaxis_title="Error Percentage (%)",
        yaxis=dict(ticksuffix="%")
    )
    
    return apply_chart_theme(fig)
