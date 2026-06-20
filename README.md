# Weather Forecasting & ML Prediction Web App

A clean, interactive, premium blue-and-black themed weather forecasting dashboard built in Python using **Streamlit**, **Scikit-learn (Polynomial Regression)**, and **Plotly**. It downloads historical weather records, trains a polynomial model to capture temperature curves, predicts upcoming weather, and evaluates its accuracy directly against official weather forecasts from the **Open-Meteo API**.

## Features

1. **Location Selection:** Supports 10 major Indian cities (Kolkata, Delhi, Mumbai, Chennai, Bengaluru, Hyderabad, Pune, Jaipur, Ahmedabad, Bhubaneswar).
2. **Dynamic Geocoding:** Automatically fetches real-time coordinates, country, and timezone using the Open-Meteo Geocoding API.
3. **Historical Training Range:** Dynamic training dataset (100 to 730 days of archives) to feed the Polynomial Regression model.
4. **Interactive Dashboard Controls:** Custom controls for Polynomial degree (1-5), Forecast days (1-14), and Historical period.
5. **Accuracy Metric Evaluation:** Computes daily error percentages, average error, min error, max error, and rates the model's prediction quality.
6. **Premium Styling:** Sleek blue-and-black custom UI layout with interactive Plotly visualizations.

## Folder Structure

```text
weather_forecasting_app/
│
├── app.py
├── requirements.txt
├── README.md
│
└── utils/
    ├── api_utils.py
    ├── model_utils.py
    └── plot_utils.py
```

## Running the App Locally

### 1. Set Up Environment
Navigate to the `weather_forecasting_app` folder and initialize a Python virtual environment:

```bash
# Create virtual environment
py -m venv .venv

# Activate virtual environment (Windows Powershell)
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Launch the Application
Run the Streamlit server:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to interact with the dashboard.
