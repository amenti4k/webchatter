!pip install streamlit
!pip install jupyter_http_over_ws
!pip install pyngrok
!pip install streamlit
!pip install pyngrok
!pip install pyod
!pip install adtk


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from adtk.detector import LevelShiftAD, QuantileAD, InterQuartileRangeAD
from adtk.data import validate_series
from scipy.signal import find_peaks

import pandas as pd
def load_data(url):
    df = pd.read_csv(url)
    return df


rounds = load_data('https://raw.githubusercontent.com/amenti4k/webchatter/main/funding_rounds.csv')
rounds.head()
# just get stuff I want
rounds1 = rounds[(rounds['funding_round_type'] == 'series-a') & (rounds['raised_amount_usd'].notna())]
rounds1 = rounds[['funded_at', 'raised_amount_usd']]
rounds1.head()
acquisitions = load_data('https://raw.githubusercontent.com/amenti4k/webchatter/main/acquisitions.csv')
acquisitions.head()
acquisitions2 = acquisitions[['acquired_at','price_amount']]
acquisitions2.head()
temp_fail = load_data('https://raw.githubusercontent.com/amenti4k/webchatter/main/ambient_temperature_system_failure.csv')
temp_fail.head()
cpu = load_data('https://raw.githubusercontent.com/amenti4k/webchatter/main/cpu_utilization_asg_misconfiguration.csv')
cpu.head()
taxi = load_data('https://raw.githubusercontent.com/amenti4k/webchatter/main/nyc_taxi.csv')
taxi.head()
dau = load_data('https://raw.githubusercontent.com/amenti4k/webchatter/main/trialthriveportcodau.csv')
dau.head()
# Data Preprocessing
def preprocess_data(df, date_col, value_col):
    df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna(subset=['ds', 'y'])
    df = df.sort_values(by='ds')
    return df
rounds = preprocess_data(rounds, 'funded_at', 'raised_amount_usd')
acquisitions = preprocess_data(acquisitions, 'acquired_at', 'price_amount')
temp_fail = preprocess_data(temp_fail, 'timestamp', 'value')
cpu = preprocess_data(cpu, 'timestamp', 'value')
taxi = preprocess_data(taxi, 'timestamp', 'value')
dau = preprocess_data(dau, 'Date', 'DAU')

# Forecasting
def prophet_forecast(df, periods):
    model = Prophet(daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def arima_forecast(df, periods):
    df_temp = df.reset_index(drop=True)
    model = ARIMA(df_temp['y'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    forecast_df = pd.DataFrame({'ds': pd.date_range(start=df['ds'].max(), periods=periods+1, freq='D')[1:], 'y': forecast})
    return forecast_df

# Anomaly Detection
def isolation_forest(df):
    model = IsolationForest(contamination=0.05)
    anomalies = model.fit_predict(df[['y']])
    df['anomaly'] = anomalies
    return df

def knn(df):
    model = KNN(contamination=0.05)
    model.fit(df[['y']])
    anomalies = model.labels_
    df['anomaly'] = anomalies
    return df

def lof(df):
    model = LOF(contamination=0.05)
    model.fit(df[['y']])
    anomalies = model.labels_
    df['anomaly'] = anomalies
    return df

def pca(df):
    model = PCA(contamination=0.05)
    model.fit(df[['y']])
    anomalies = model.labels_
    df['anomaly'] = anomalies
    return df
    
def adtk_detector(df):
    df_original = df.copy()  # Create a copy of the original DataFrame
    df = df.set_index('ds')
    s = validate_series(df)
    level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
    anomalies_level_shift = level_shift_ad.fit_detect(s)
    quantile_ad = QuantileAD(high=0.99, low=0.01)
    anomalies_quantile = quantile_ad.fit_detect(s)
    iqr_ad = InterQuartileRangeAD(c=1.5)
    anomalies_iqr = iqr_ad.fit_detect(s)
    if isinstance(anomalies_level_shift, pd.DataFrame):
        anomalies_level_shift = anomalies_level_shift.iloc[:, 0]
    if isinstance(anomalies_quantile, pd.DataFrame):
        anomalies_quantile = anomalies_quantile.iloc[:, 0]
    if isinstance(anomalies_iqr, pd.DataFrame):
        anomalies_iqr = anomalies_iqr.iloc[:, 0]
    df_original['level_shift'] = anomalies_level_shift.reindex(df_original.index, fill_value=False)
    df_original['quantile'] = anomalies_quantile.reindex(df_original.index, fill_value=False)
    df_original['iqr'] = anomalies_iqr.reindex(df_original.index, fill_value=False)
    return df_original
  # Inflection Point Analysis
def find_inflection_points(df, threshold=2):
    rolling_mean = df['y'].rolling(window=30).mean()
    rolling_std = df['y'].rolling(window=30).std()
    inflection_points = df[(df['y'] > rolling_mean + threshold * rolling_std) | (df['y'] < rolling_mean - threshold * rolling_std)]
    return inflection_points
  # Trend Analysis
def calculate_trends(df):
    df['mom'] = df['y'].pct_change()
    df['qoq'] = df['y'].pct_change(periods=3)
    df['yoy'] = df['y'].pct_change(periods=12)
    return df
# Apply forecasting, anomaly detection, and trend analysis to all datasets
datasets = {'rounds': rounds, 'acquisitions': acquisitions, 'temp_fail': temp_fail, 'cpu': cpu, 'taxi': taxi, 'dau': dau}
results = {}

for name, df in datasets.items():
    forecast_prophet = prophet_forecast(df, periods=12)
    forecast_arima = arima_forecast(df, periods=12)
    anomalies_if = isolation_forest(df)
    anomalies_knn = knn(df)
    anomalies_lof = lof(df)
    anomalies_pca = pca(df)
    anomalies_adtk = adtk_detector(df)
    inflection_points = find_inflection_points(df)
    trends = calculate_trends(df)
    
    results[name] = {
        'forecast_prophet': forecast_prophet,
        'forecast_arima': forecast_arima,
        'anomalies_if': anomalies_if,
        'anomalies_knn': anomalies_knn,
        'anomalies_lof': anomalies_lof,
        'anomalies_pca': anomalies_pca,
        'anomalies_adtk': anomalies_adtk,
        'inflection_points': inflection_points,
        'trends': trends
    }

# Visualization
def plot_forecast(df, forecast):
    fig = px.line(df, x='ds', y='y', title='Forecast')
    fig.add_trace(px.line(forecast, x='ds', y='yhat').data[0])
    fig.add_trace(px.scatter(forecast, x='ds', y='yhat_upper').data[0])
    fig.add_trace(px.scatter(forecast, x='ds', y='yhat_lower').data[0])
    st.plotly_chart(fig)

def plot_anomalies(df, anomaly_col):
    fig = px.scatter(df, x='ds', y='y', color=anomaly_col, title='Anomalies')
    st.plotly_chart(fig)

def plot_inflection_points(df, inflection_points):
    fig = px.line(df, x='ds', y='y', title='Inflection Points')
    fig.add_trace(px.scatter(inflection_points, x='ds', y='y').data[0])
    st.plotly_chart(fig)

def plot_trends(df):
    fig = px.line(df, x='ds', y=['mom', 'qoq', 'yoy'], title='Trends')
    st.plotly_chart(fig)

# Streamlit Dashboard
def main():
    st.title('Prospecting and Monitoring Dashboard')
    
    dataset = st.sidebar.selectbox('Select Dataset', list(datasets.keys()))
    analysis = st.sidebar.selectbox('Select Analysis', ['Forecast', 'Anomalies', 'Inflection Points', 'Trends'])
    
    if analysis == 'Forecast':
        forecast_method = st.sidebar.selectbox('Select Forecast Method', ['Prophet', 'ARIMA'])
        if forecast_method == 'Prophet':
            plot_forecast(datasets[dataset], results[dataset]['forecast_prophet'])
        else:
            plot_forecast(datasets[dataset], results[dataset]['forecast_arima'])
    
    elif analysis == 'Anomalies':
        anomaly_method = st.sidebar.selectbox('Select Anomaly Detection Method', ['Isolation Forest', 'KNN', 'LOF', 'PCA', 'ADTK'])
        if anomaly_method == 'Isolation Forest':
            plot_anomalies(results[dataset]['anomalies_if'], 'anomaly')
        elif anomaly_method == 'KNN':
            plot_anomalies(results[dataset]['anomalies_knn'], 'anomaly')
        elif anomaly_method == 'LOF':
            plot_anomalies(results[dataset]['anomalies_lof'], 'anomaly')
        elif anomaly_method == 'PCA':
            plot_anomalies(results[dataset]['anomalies_pca'], 'anomaly')
        else:
            plot_anomalies(results[dataset]['anomalies_adtk'], 'level_shift')
            plot_anomalies(results[dataset]['anomalies_adtk'], 'quantile')
            plot_anomalies(results[dataset]['anomalies_adtk'], 'iqr')
    
    elif analysis == 'Inflection Points':
        plot_inflection_points(datasets[dataset], results[dataset]['inflection_points'])
    
    else:
        plot_trends(results[dataset]['trends'])
