import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def train_polynomial_regression(df_history, degree=3):
    """
    Trains a Polynomial Regression model on historical temperature data.
    df_history: DataFrame with ['date', 'temp_mean']
    degree: int, polynomial degree
    Returns (model, X_train, y_train, df_history_with_index)
    """
    df_history = df_history.copy()
    # Create day number index
    df_history["day_index"] = np.arange(1, len(df_history) + 1)
    
    X = df_history[["day_index"]].values
    y = df_history["temp_mean"].values
    
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    model.fit(X, y)
    
    # Calculate fitted values for visualization
    df_history["temp_fitted"] = model.predict(X)
    
    return model, X, y, df_history

def predict_future_temperatures(model, start_index, num_days):
    """
    Predicts temperatures for upcoming days.
    start_index: int (typically len(df_history) + 1)
    num_days: int (forecast period)
    Returns a dict with day_indices and predicted_temperatures.
    """
    future_indices = np.arange(start_index, start_index + num_days).reshape(-1, 1)
    predictions = model.predict(future_indices)
    return {
        "day_indices": future_indices.flatten(),
        "predictions": predictions
    }

def calculate_metrics(df_comparison):
    """
    Calculates errors between official forecast and model predictions.
    df_comparison: DataFrame containing ['forecast_mean', 'predicted_temp']
    Returns a dict with error_percentages, avg_error, min_error, max_error, and quality label.
    """
    if len(df_comparison) == 0:
        return {
            "errors": [],
            "avg_error": 0.0,
            "min_error": 0.0,
            "max_error": 0.0,
            "quality": "Unknown"
        }
        
    actual = df_comparison["forecast_mean"]
    pred = df_comparison["predicted_temp"]
    
    errors = np.abs(actual - pred) / actual * 100
    
    avg_error = float(np.mean(errors))
    min_error = float(np.min(errors))
    max_error = float(np.max(errors))
    
    if avg_error <= 5.0:
        quality = "Good"
    elif avg_error <= 10.0:
        quality = "Moderate"
    else:
        quality = "Needs Improvement"
        
    return {
        "errors": errors.tolist(),
        "avg_error": round(avg_error, 2),
        "min_error": round(min_error, 2),
        "max_error": round(max_error, 2),
        "quality": quality
    }
