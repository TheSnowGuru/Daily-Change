import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsai.all import *
from sklearn.model_selection import train_test_split
import streamlit as st

# Load Data
df = pd.read_csv('data\ixic.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Preprocessing
# Weekly Open Price and Distance from EMA21
df['WeeklyOpenPrice'] = df['Open'].resample('W-MON').transform('first')
df['WeeklyOpenPrice'] = df['WeeklyOpenPrice'].ffill()
df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
df['DistFromEMA21'] = df['Close'] - df['EMA21']

# Percent Changes
df['DailyPercentChange'] = df['Close'].pct_change()
df['WeeklyPercentChange'] = df['Close'].pct_change(periods=5)

# Normalize Features
scaler = MinMaxScaler()
df[['Open', 'WeeklyOpenPrice', 'DailyPercentChange', 'WeeklyPercentChange', 'DistFromEMA21']] = scaler.fit_transform(
    df[['Open', 'WeeklyOpenPrice', 'DailyPercentChange', 'WeeklyPercentChange', 'DistFromEMA21']]
)

# Prepare data for each day of the week
days_data = {i: df[df.index.dayofweek == i] for i in range(5)}  # 0 is Monday, 4 is Friday

# Placeholder for predictions
predictions = pd.DataFrame()

# Training separate models for each day
for day, day_df in days_data.items():
    X = day_df[['Open', 'WeeklyOpenPrice', 'DailyPercentChange', 'WeeklyPercentChange', 'DistFromEMA21']]
    y = day_df['DailyPercentChange'].shift(-1)
    
    # Dropping NaN values created by shift operation
    X = X.iloc[:-1, :]
    y = y.dropna()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model - example using LSTM
    model = LSTM(X_train.shape[1], y_train.shape[0])
    learn = Learner(model, X_train, y_train, loss_func=MSELossFlat(), metrics=[mae, rmse])
    
    # Training
    learn.fit_one_cycle(5, 1e-3)

    # Make predictions on test set
    preds, targs = learn.get_preds(ds_idx=1)  # ds_idx=1 for validation set
    day_name = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}[day]
    predictions[day_name] = preds.numpy().flatten()

# Streamlit dashboard
def display_dashboard(preds):
    st.title('NASDAQ Daily Percent Change Prediction')
    st.write('Predictions by Model:')
    st.table(preds)

if __name__ == "__main__":
    display_dashboard(predictions)
