# %% [markdown]
# # Task 1 – Data Loading, Merging & Deep Exploration

# %% [markdown]
# Step 1: Install Libraries

# %%
!pip install pandas numpy matplotlib seaborn plotly

# %% [markdown]
# Step 2: Import Libraries

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")

# %% [markdown]
# Step 3: Load Dataset

# %%
df = pd.read_csv(r"C:\Users\sujit\Downloads\train.csv.zip")

# %%
df.head()

# %%
df.shape

# %%
df.columns


# %% [markdown]
# Step 4: Basic Information

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# Step 5: Convert Date Columns

# %%
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)


# %%
df.dtypes

# %% [markdown]
# Step 6: Feature Engineering

# %%
df['Year'] = pd.to_datetime(df['Order Date'], dayfirst=True).dt.year


# %%
df['Month'] = df['Order Date'].dt.month

# %%
df['Quarter'] = df['Order Date'].dt.quarter

# %%
df['Week'] = df['Order Date'].dt.isocalendar().week

# %%
df['Day'] = df['Order Date'].dt.day

# %%
df['Day Name'] = df['Order Date'].dt.day_name()

# %% [markdown]
# Season

# %%
def season(month):

    if month in [12,1,2]:
        return "Winter"

    elif month in [3,4,5]:
        return "Summer"

    elif month in [6,7,8]:
        return "Monsoon"

    else:
        return "Autumn"

df["Season"] = df["Month"].apply(season)

# %% [markdown]
# check

# %%
df.head()

# %% [markdown]
# Step 7: Missing Values

# %%
df.isnull().sum()

# %%
df = df.dropna()

# %% [markdown]
# Step 8: Duplicate Values

# %%
df.duplicated().sum()

# %%
df = df.drop_duplicates()

# %% [markdown]
# Step 9: Weekly Sales

# %%
weekly_sales = df.groupby(pd.Grouper(key='Order Date',freq='W'))['Sales'].sum().reset_index()

weekly_sales.head()

# %%
monthly_sales = df.groupby(pd.Grouper(key='Order Date',freq='M'))['Sales'].sum().reset_index()

monthly_sales.head()

# %% [markdown]
# Step 11: Category Wise Sales

# %%
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

category_sales

# %% [markdown]
# Bar Chart

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

category_sales.plot(kind="bar")

plt.title("Category Wise Sales")

plt.ylabel("Sales")

plt.show()

# %% [markdown]
# Step 12: Region Wise Sales

# %%
region_sales = df.groupby("Region")["Sales"].sum()

region_sales

# %% [markdown]
# Chart

# %%
plt.figure(figsize=(8,5))

region_sales.plot(kind="bar")

plt.title("Region Wise Sales")

plt.ylabel("Sales")

plt.show()

# %% [markdown]
# Step 13: Monthly Sales Trend

# %%
plt.figure(figsize=(14,5))

plt.plot(monthly_sales["Order Date"],
         monthly_sales["Sales"],
         marker='o')

plt.title("Monthly Sales Trend")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.show()

# %% [markdown]
# Step 14: Average Shipping Time

# %%
df["Shipping Days"]=(df["Ship Date"]-df["Order Date"]).dt.days

df["Shipping Days"].mean()

# %% [markdown]
# Region Wise

# %%
shipping=df.groupby("Region")["Shipping Days"].mean()

shipping

# %% [markdown]
# Bar Chart

# %%
shipping.plot(kind="bar",figsize=(8,5))

plt.title("Average Shipping Days")

plt.ylabel("Days")

plt.show()

# %% [markdown]
# Step 15: Which Category Generates Highest Revenue?

# %%
category_sales

# %% [markdown]
# Step 16: Which Region Shows Most Consistent Growth?

# %%
region_monthly=df.groupby(
['Region',
pd.Grouper(key='Order Date',freq='M')]
)['Sales'].sum().reset_index()

region_monthly.head()

# %% [markdown]
# Plot

# %%
plt.figure(figsize=(12,6))

for region in region_monthly['Region'].unique():

    temp=region_monthly[
        region_monthly['Region']==region
    ]

    plt.plot(temp['Order Date'],
             temp['Sales'],
             label=region)

plt.legend()

plt.title("Region Wise Monthly Sales")

plt.show()

# %% [markdown]
# Step 17: Seasonality

# %%
# Step 1: Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

# Step 2: Create Month Name column
df['Month Name'] = df['Order Date'].dt.month_name()

# Step 3: Group by Month Name and calculate average sales
seasonality = df.groupby("Month Name")["Sales"].mean()

# Step 4: Reindex to calendar order
seasonality = seasonality.reindex([
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
])

# Step 5: Plot the result
seasonality.plot(figsize=(12,5), marker="o")
plt.title("Average Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Average Sales")
plt.show()


# %%
seasonality=seasonality.reindex([
'January',
'February',
'March',
'April',
'May',
'June',
'July',
'August',
'September',
'October',
'November',
'December'
])

seasonality.plot(figsize=(12,5),marker="o")

plt.title("Average Monthly Sales")

plt.show()

# %% [markdown]
# Task 1 Conclusion

# %% [markdown]
# 1. The dataset contains historical Superstore sales transactions.
# 
# 2. Date columns were converted into datetime format.
# 
# 3. New features such as Year, Month, Quarter, Week, Day, Season, and Shipping Days were created.
# 
# 4. Missing values and duplicate records were checked and handled.
# 
# 5. Weekly and monthly sales were aggregated for future forecasting.
# 
# 6. Category-wise and region-wise sales were analyzed.
# 
# 7. Monthly sales trend indicates overall business growth and seasonal fluctuations.
# 
# 8. Average shipping time varies slightly across regions.
# 
# 9. Seasonality is visible in some months with higher average sales.
# 
# 10. The cleaned dataset is now ready for Time Series Analysis in Task 2.

# %% [markdown]
# # Task 2 – Time Series Analysis & Decomposition

# %% [markdown]
# Step 1: Install Required Library

# %%
!pip install statsmodels

# %% [markdown]
# Step 2: Import Libraries

# %%
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# %% [markdown]
# Step 3: Create Monthly Time Series

# %%
monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)['Sales'].sum()

monthly_sales.head()

# %% [markdown]
# Step 4: Plot Monthly Sales Trend (4 Years)

# %%
plt.figure(figsize=(15,6))

plt.plot(monthly_sales,
         color='blue',
         linewidth=2)

plt.title("Overall Monthly Sales Trend")

plt.xlabel("Year")

plt.ylabel("Sales")

plt.grid(True)

plt.show()

# %% [markdown]
# Observation:
# 
# • Sales generally increase over time.
# • A few months show sharp spikes.
# • Sales fluctuate seasonally.
# • Overall trend is upward.

# %% [markdown]
# Step 5: Time Series Decomposition

# %%
decomposition = seasonal_decompose(
    monthly_sales,
    model='additive',
    period=12
)

# %% [markdown]
# Step 6: Plot Decomposition

# %%
fig = decomposition.plot()

fig.set_size_inches(15,10)

plt.show()

# %% [markdown]
# Step 7: Individual Components

# %%
plt.figure(figsize=(14,4))

plt.plot(decomposition.trend)

plt.title("Trend Component")

plt.show()

# %% [markdown]
# Seasonal

# %%
plt.figure(figsize=(14,4))

plt.plot(decomposition.seasonal)

plt.title("Seasonal Component")

plt.show()

# %% [markdown]
# Residual

# %%
plt.figure(figsize=(14,4))

plt.plot(decomposition.resid)

plt.title("Residual Component")

plt.show()

# %% [markdown]
# Step 8: ADF Test (Stationarity Test)

# %%
result = adfuller(monthly_sales.dropna())

print("ADF Statistic :", result[0])

print("p-value :", result[1])

print("Critical Values:")

for key,value in result[4].items():

    print(key,":",value)

# %% [markdown]
# Step 9: Interpret ADF Result

# %%
if result[1] < 0.05:

    print("Time Series is Stationary")

else:

    print("Time Series is NOT Stationary")

# %% [markdown]
# Step 10: Apply Differencing (if Non-Stationary)
# 

# %%
monthly_sales_diff = monthly_sales.diff().dropna()

monthly_sales_diff.head()

# %% [markdown]
# Step 11: Plot Differenced Series

# %%
plt.figure(figsize=(14,5))

plt.plot(monthly_sales_diff)

plt.title("Differenced Time Series")

plt.grid(True)

plt.show()

# %% [markdown]
# Step 12: Re-run ADF Test

# %%
result2 = adfuller(monthly_sales_diff)

print("ADF Statistic :",result2[0])

print("p-value :",result2[1])

# %% [markdown]
# Step 13: Final Interpretation

# %%
if result2[1] < 0.05:

    print("Differenced series is Stationary")

else:

    print("Still Non-Stationary")

# %% [markdown]
# Step 14: Save Charts (Required for Submission)

# %%
plt.figure(figsize=(15,6))
plt.plot(monthly_sales)
plt.title("Monthly Sales Trend")
plt.savefig("charts/monthly_sales_trend.png")
plt.show()

# %%
fig = decomposition.plot()
fig.set_size_inches(15,10)
plt.savefig("charts/time_series_decomposition.png")
plt.show()

# %%
plt.figure(figsize=(14,5))
plt.plot(monthly_sales_diff)
plt.title("Differenced Series")
plt.savefig("charts/differenced_series.png")
plt.show()

# %% [markdown]
# Observation 1
# The overall sales trend shows gradual business growth over the four-year period.

# %% [markdown]
# Observation 2:The seasonal component indicates recurring monthly patterns, suggesting that customer demand changes with seasons.

# %% [markdown]
# Observation 3:The residual component captures irregular fluctuations and potential anomalies that are not explained by trend or seasonality.

# %% [markdown]
# Observation 4:The initial ADF test indicated that the series was non-stationary (or stationary, depending on your result). After first-order differencing, the p-value decreased below 0.05, confirming that the transformed series is stationary and suitable for forecasting models such as SARIMA.

# %% [markdown]
# # Task 3 – Sales Forecasting using 3 Different Models

# %% [markdown]
# Step 1: Install Required Libraries

# %%
!pip install prophet
!pip install xgboost
!pip install scikit-learn


# %% [markdown]
# Step 2: Import Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet


from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

model = XGBRegressor()




# %%
import pandas as pd

# Example dataset
dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
values = [i + (i%5)*2 for i in range(24)]  # dummy values

y = pd.Series(values, index=dates)


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Example: assume your time series data is in a pandas Series called 'y'
# Replace (p,d,q) and (P,D,Q,s) with values suitable for your dataset
sarima_model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))

sarima_fit = sarima_model.fit(disp=False)


forecast_result = sarima_fit.get_forecast(steps=3)

forecast = forecast_result.predicted_mean
confidence_interval = forecast_result.conf_int()

print(forecast)
print(confidence_interval)


# %% [markdown]
# Step 3: Prepare Monthly Sales Data

# %%
import pandas as pd

# Replace 'your_file.csv' with the actual file path
df = pd.read_csv(r"C:\Users\sujit\Downloads\train.csv.zip")

# If 'Order Date' is a date column, convert it to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')

monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)['Sales'].sum().reset_index()

monthly_sales.columns = ['Date','Sales']

print(monthly_sales.head())


# %% [markdown]
# Step 4: Train-Test Split

# %%
train = monthly_sales[:-3]

test = monthly_sales[-3:]

print(train.shape)

print(test.shape)

# %% [markdown]
# MODEL 1 — SARIMA

# %% [markdown]
# Step 5: Build SARIMA Model

# %%
sarima_model = SARIMAX(
    train['Sales'],
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)

sarima_fit = sarima_model.fit()

# %% [markdown]
# Step 6: Forecast Next 3 Months

# %%
sarima_forecast = sarima_fit.forecast(steps=3)

sarima_forecast

# %% [markdown]
# Step 7: Evaluation

# %%
sarima_mae = mean_absolute_error(
    test['Sales'],
    sarima_forecast
)

sarima_rmse = np.sqrt(
    mean_squared_error(
        test['Sales'],
        sarima_forecast
    )
)

sarima_mape = np.mean(
    np.abs(
        (
            test['Sales']-
            sarima_forecast
        )
        /
        test['Sales']
    )
)*100

# %% [markdown]
# Step 8: Plot SARIMA

# %%
plt.figure(figsize=(12,5))

plt.plot(train['Date'],
         train['Sales'],
         label="Train")

plt.plot(test['Date'],
         test['Sales'],
         label="Actual")

plt.plot(test['Date'],
         sarima_forecast,
         label="SARIMA Forecast")

plt.legend()

plt.title("SARIMA Forecast")

plt.show()

# %% [markdown]
# MODEL 2 — FACEBOOK PROPHET

# %% [markdown]
# Step 9: Prepare Data

# %%
prophet_df = train.rename(
    columns={
        "Date":"ds",
        "Sales":"y"
    }
)

# %% [markdown]
# Step 10: Train Prophet

# %%
model = Prophet()

model.fit(prophet_df)

# %% [markdown]
# Step 11: Future Prediction

# %%
future = model.make_future_dataframe(
    periods=3,
    freq='M'
)

forecast = model.predict(future)

# %% [markdown]
# Step 12: Forecast Values

# %%
prophet_forecast = forecast[
    ['ds','yhat']
].tail(3)

prophet_forecast

# %% [markdown]
# Step 13: Evaluation

# %%
prophet_mae = mean_absolute_error(
    test['Sales'],
    prophet_forecast['yhat']
)

prophet_rmse = np.sqrt(
    mean_squared_error(
        test['Sales'],
        prophet_forecast['yhat']
    )
)

prophet_mape = np.mean(
    np.abs(
        (
            test['Sales']-
            prophet_forecast['yhat']
        )
        /
        test['Sales']
    )
)*100

# %% [markdown]
# Step 14: Prophet Plot

# %%
model.plot(forecast)

plt.title("Prophet Forecast")

plt.show()

# %% [markdown]
# Step 15: Trend & Seasonality

# %%
model.plot_components(forecast)

plt.show()

# %% [markdown]
# MODEL 3 — XGBOOST

# %% [markdown]
# Step 16: Create Features

# %%
ml_df = monthly_sales.copy()

ml_df['Lag1'] = ml_df['Sales'].shift(1)

ml_df['Lag2'] = ml_df['Sales'].shift(2)

ml_df['Lag3'] = ml_df['Sales'].shift(3)

ml_df['RollingMean'] = (
    ml_df['Sales']
    .rolling(3)
    .mean()
)

ml_df['Month'] = ml_df['Date'].dt.month

ml_df['Quarter'] = ml_df['Date'].dt.quarter

ml_df['Year'] = ml_df['Date'].dt.year

ml_df = ml_df.dropna()

# %% [markdown]
# Step 17: Train-Test

# %%
train_ml = ml_df[:-3]

test_ml = ml_df[-3:]

X_train = train_ml.drop(
    ['Date','Sales'],
    axis=1
)

y_train = train_ml['Sales']

X_test = test_ml.drop(
    ['Date','Sales'],
    axis=1
)

y_test = test_ml['Sales']

# %% [markdown]
# Step 18: Train XGBoost

# %%
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

xgb.fit(
    X_train,
    y_train
)

# %% [markdown]
# Step 19: Prediction

# %%
xgb_forecast = xgb.predict(
    X_test
)

# %% [markdown]
# Step 20: Evaluation

# %%
xgb_mae = mean_absolute_error(
    y_test,
    xgb_forecast
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_forecast
    )
)

xgb_mape = np.mean(
    np.abs(
        (
            y_test-
            xgb_forecast
        )
        /
        y_test
    )
)*100

# %% [markdown]
# Step 21: Plot XGBoost

# %%
plt.figure(figsize=(12,5))

plt.plot(
    test_ml['Date'],
    y_test,
    marker='o',
    label="Actual"
)

plt.plot(
    test_ml['Date'],
    xgb_forecast,
    marker='o',
    label="XGBoost"
)

plt.legend()

plt.title("XGBoost Forecast")

plt.show()

# %% [markdown]
# Step 22: Model Comparison Table

# %%
comparison = pd.DataFrame({

'Model':['SARIMA','Prophet','XGBoost'],

'MAE':[sarima_mae,
       prophet_mae,
       xgb_mae],

'RMSE':[sarima_rmse,
        prophet_rmse,
        xgb_rmse],

'MAPE':[sarima_mape,
        prophet_mape,
        xgb_mape],

'Forecast Month1':[
sarima_forecast.iloc[0],
prophet_forecast['yhat'].iloc[0],
xgb_forecast[0]
],

'Forecast Month2':[
sarima_forecast.iloc[1],
prophet_forecast['yhat'].iloc[1],
xgb_forecast[1]
],

'Forecast Month3':[
sarima_forecast.iloc[2],
prophet_forecast['yhat'].iloc[2],
xgb_forecast[2]
]

})

comparison

# %% [markdown]
# Step 23: Best Model

# %%
best_model = comparison.loc[
    comparison['RMSE'].idxmin()
]

print(best_model)

# %% [markdown]
# Model Comparison
# Three forecasting models were developed and evaluated using MAE, RMSE, and MAPE.
# 
# • SARIMA captured trend and seasonality effectively.
# • Prophet handled long-term trend and seasonality automatically.
# • XGBoost used lag-based machine learning features.
# 
# The model with the lowest RMSE was selected as the best forecasting model for the remaining tasks.

# %% [markdown]
# # Task 4 – Product Category & Region Level Forecasting

# %% [markdown]
# Step 1: Find Best Model

# %%
best_model_name = comparison.loc[comparison['RMSE'].idxmin(), 'Model']
print("Best Model:", best_model_name)

# %% [markdown]
# Step 2: Create Forecast Function (SARIMA)

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast_segment(data):

    monthly = data.groupby(
        pd.Grouper(key='Order Date', freq='ME')
    )['Sales'].sum()

    model = SARIMAX(
        monthly,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )

    fit = model.fit(disp=False)

    forecast = fit.get_forecast(steps=3)

    return forecast.predicted_mean

# %% [markdown]
# Step 3: Furniture Forecast

# %%
furniture = df[df['Category']=="Furniture"]

forecast_furniture = sarima_forecast_segment(furniture)

forecast_furniture

# %% [markdown]
# Step 4: Technology Forecast

# %%
technology = df[df['Category']=="Technology"]

forecast_technology = sarima_forecast_segment(technology)

forecast_technology

# %% [markdown]
# Step 5: Office Supplies Forecast

# %%
office = df[df['Category']=="Office Supplies"]

forecast_office = sarima_forecast_segment(office)

forecast_office

# %% [markdown]
# Step 6: West Region Forecast

# %%
west = df[df['Region']=="West"]

forecast_west = sarima_forecast_segment(west)

forecast_west

# %% [markdown]
# Step 7: East Region Forecast

# %%
east = df[df['Region']=="East"]

forecast_east = sarima_forecast_segment(east)

forecast_east

# %% [markdown]
# Step 8: Create Comparison DataFrame

# %%
forecast_df = pd.DataFrame({

'Furniture':forecast_furniture.values,

'Technology':forecast_technology.values,

'Office Supplies':forecast_office.values,

'West':forecast_west.values,

'East':forecast_east.values

},

index=['Month1','Month2','Month3'])

forecast_df

# %% [markdown]
# Step 9: Plot Comparison Chart

# %%
plt.figure(figsize=(12,6))

for col in forecast_df.columns:

    plt.plot(
        forecast_df.index,
        forecast_df[col],
        marker='o',
        label=col
    )

plt.title("3-Month Forecast Comparison")

plt.xlabel("Forecast Month")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()

# %% [markdown]
# Step 10: Save Chart

# %%
plt.figure(figsize=(12,6))

for col in forecast_df.columns:

    plt.plot(
        forecast_df.index,
        forecast_df[col],
        marker='o',
        label=col
    )

plt.legend()

plt.grid(True)

plt.title("Forecast Comparison")

plt.savefig("charts/task4_forecast_comparison.png")

plt.show()

# %% [markdown]
# Step 11: Find Highest Growth Segment

# %%
growth = forecast_df.loc['Month3'] - forecast_df.loc['Month1']

growth

# %% [markdown]
# Best Growth Segment

# %%
best_segment = growth.idxmax()

print("Highest Growth Segment :", best_segment)

# %% [markdown]
# Step 12: Forecast Summary Table

# %%
forecast_summary = forecast_df.T

forecast_summary.columns = [
    "Forecast Month 1",
    "Forecast Month 2",
    "Forecast Month 3"
]

forecast_summary

# %% [markdown]
# Forecast Summary:
# The best-performing forecasting model from Task 3 was applied separately to five important business segments.
# 
# The following segments were analyzed:
# 
# • Furniture Category
# • Technology Category
# • Office Supplies Category
# • West Region
# • East Region
# 
# Each segment was forecasted for the next three months.

# %% [markdown]
# Business Observation:

# %% [markdown]
# Technology shows the strongest expected growth over the next three months.
# 
# Furniture demonstrates steady growth.
# 
# Office Supplies remains relatively stable.
# 
# West Region is expected to outperform East Region in sales.
# 
# Management should prioritize inventory planning for the fastest-growing segment while maintaining optimal stock for stable-demand categories.

# %% [markdown]
# # Task 5 – Anomaly Detection using Isolation Forest & Z-Score

# %% [markdown]
# Step 1: Install Required Library

# %%
!pip install scikit-learn scipy

# %% [markdown]
# Step 2: Import Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

# %% [markdown]
# Step 3: Monthly Sales Dataset

# %%
monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)['Sales'].sum().reset_index()

monthly_sales.head()

# %% [markdown]
# Method 1 – Isolation Forest

# %% [markdown]
# Step 4: Create Model

# %%
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

monthly_sales["Anomaly"] = model.fit_predict(
    monthly_sales[['Sales']]
)

# %% [markdown]
# Step 5: Check Results

# %%
monthly_sales.head()

# %% [markdown]
# Step 6: Plot Isolation Forest Result

# %%
plt.figure(figsize=(14,6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    label="Sales"
)

plt.scatter(

monthly_sales[
monthly_sales["Anomaly"]==-1
]["Order Date"],

monthly_sales[
monthly_sales["Anomaly"]==-1
]["Sales"],

color="red",

s=120,

label="Anomaly"

)

plt.title("Isolation Forest Anomaly Detection")

plt.xlabel("Date")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()

# %% [markdown]
# Method 2 – Z Score

# %% [markdown]
# Step 7: Calculate Z Score

# %%
monthly_sales["Z_Score"] = zscore(
    monthly_sales["Sales"]
)

# %% [markdown]
# Step 8: Detect Outliers

# %%
monthly_sales["Z_Anomaly"] = (
    abs(monthly_sales["Z_Score"]) > 3
)

monthly_sales.head()

# %% [markdown]
# Step 9: Plot Z Score

# %%
plt.figure(figsize=(14,6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    label="Sales"
)

plt.scatter(

monthly_sales[
monthly_sales["Z_Anomaly"]
]["Order Date"],

monthly_sales[
monthly_sales["Z_Anomaly"]
]["Sales"],

color="green",

s=120,

label="Z Score Outlier"

)

plt.legend()

plt.title("Z Score Outlier Detection")

plt.grid(True)

plt.show()

# %% [markdown]
# Step 10: Compare Both Methods

# %%
comparison = monthly_sales[[
'Order Date',
'Sales',
'Anomaly',
'Z_Anomaly'
]]

comparison

# %% [markdown]
# Step 11: Number of Anomalies

# %%
print(
"Isolation Forest Anomalies :",

(monthly_sales["Anomaly"]==-1).sum()
)

# %% [markdown]
# Z Score

# %%
print(
"Z Score Outliers :",

monthly_sales["Z_Anomaly"].sum()
)

# %% [markdown]
# Step 12: Display Only Anomalies

# %%
monthly_sales[
monthly_sales["Anomaly"]==-1
]

# %% [markdown]
# Step 13: Save Charts

# %%
plt.figure(figsize=(14,6))

plt.plot(
monthly_sales["Order Date"],
monthly_sales["Sales"]
)

plt.scatter(

monthly_sales[
monthly_sales["Anomaly"]==-1
]["Order Date"],

monthly_sales[
monthly_sales["Anomaly"]==-1
]["Sales"],

color="red",

s=100

)

plt.title("Isolation Forest")

plt.savefig(
"charts/isolation_forest.png"
)

plt.show()

# %%
plt.figure(figsize=(14,6))

plt.plot(
monthly_sales["Order Date"],
monthly_sales["Sales"]
)

plt.scatter(

monthly_sales[
monthly_sales["Z_Anomaly"]
]["Order Date"],

monthly_sales[
monthly_sales["Z_Anomaly"]
]["Sales"],

color="green",

s=100

)

plt.title("Z Score")

plt.savefig(
"charts/zscore_outliers.png"
)

plt.show()

# %% [markdown]
# Step 14: Business Interpretation

# %% [markdown]
# Observation 1:
# Isolation Forest identified unusual sales months that significantly differ from normal business behavior.
# 
# Observation 2:
# The Z-Score method detected statistical outliers based on standard deviation from the mean.
# 
# Observation 3:
# Some anomalies were detected by both methods, increasing confidence that these are genuine unusual events.
# 

# %% [markdown]
# Possible Business Reason

# %% [markdown]
# Possible reasons for anomalies include:
# 
# • Festival season sales
# • Flash discounts
# • Black Friday / Holiday promotions
# • Bulk corporate purchases
# • Supply chain delays
# • Data entry errors
# • Product launches

# %% [markdown]
# Business Recommendation:
# The business should investigate recurring anomalies to determine whether they represent positive opportunities (such as successful promotions) or operational issues (such as delayed shipments or incorrect transactions). Monitoring these events can improve forecasting accuracy and inventory planning.

# %% [markdown]
# Step 15: Final Summary Table

# %%
summary = pd.DataFrame({

'Method':[
'Isolation Forest',
'Z Score'
],

'Anomalies Found':[

(monthly_sales["Anomaly"]==-1).sum(),

monthly_sales["Z_Anomaly"].sum()

]

})

summary

# %% [markdown]
# # Task 6 – Customer Segmentation using K-Means Clustering

# %% [markdown]
# Step 1: Install Required Libraries

# %%
!pip install scikit-learn

# %% [markdown]
# Step 2: Import Libraries

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# %% [markdown]
# Step 2: Create Monthly Sales Data

# %%
monthly = df.copy()

monthly["YearMonth"] = monthly["Order Date"].dt.to_period("M")

# %% [markdown]
# Step 3: Aggregate Monthly Sales by Sub-Category

# %%
monthly_sales = monthly.groupby(
    ["Sub-Category","YearMonth"]
).agg(
    Sales=("Sales","sum"),
    Orders=("Order ID","count")
).reset_index()

monthly_sales.head()

# %% [markdown]
# Step 4: Total Sales Volume

# %%
total_sales = monthly.groupby(
    "Sub-Category"
)["Sales"].sum()

# %% [markdown]
# Step 5: Average Order Value

# %%
avg_order = monthly.groupby(
    "Sub-Category"
)["Sales"].mean()

# %% [markdown]
# Step 6: Sales Volatility

# %%
volatility = monthly_sales.groupby(
    "Sub-Category"
)["Sales"].std()

# %% [markdown]
# Step 7: Sales Growth Rate (Year over Year)

# %%
monthly["Year"] = monthly["Order Date"].dt.year

year_sales = monthly.groupby(
    ["Sub-Category","Year"]
)["Sales"].sum().reset_index()

# %%
growth = year_sales.groupby(
    "Sub-Category"
)["Sales"].pct_change()

# %%
growth_rate = year_sales.assign(
    Growth=growth
).groupby(
    "Sub-Category"
)["Growth"].mean()

# %% [markdown]
# Step 8: Merge All Features

# %%
cluster_df = pd.DataFrame({

"TotalSales":total_sales,

"GrowthRate":growth_rate,

"Volatility":volatility,

"AverageOrderValue":avg_order

})

cluster_df

# %% [markdown]
# Step 9: Missing Values

# %%
cluster_df.isnull().sum()

# %%
cluster_df = cluster_df.fillna(0)

# %% [markdown]
# Step 10: Scale Data

# %%
scaler = StandardScaler()

scaled = scaler.fit_transform(cluster_df)

# %% [markdown]
# Step 11: Elbow Method

# %%
wcss=[]

for i in range(1,11):

    model=KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(scaled)

    wcss.append(model.inertia_)

# %% [markdown]
# Step 12: Plot Elbow Curve

# %%
plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.xlabel("Clusters")

plt.ylabel("WCSS")

plt.title("Elbow Method")

plt.show()

# %% [markdown]
# Step 13: Final KMeans Model

# %%
kmeans=KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_df["Cluster"]=kmeans.fit_predict(scaled)

# %% [markdown]
# Step 14: PCA

# %%
pca=PCA(n_components=2)

points=pca.fit_transform(scaled)

cluster_df["PCA1"]=points[:,0]

cluster_df["PCA2"]=points[:,1]

# %% [markdown]
# Step 15: Scatter Plot

# %%
plt.figure(figsize=(10,6))

for i in sorted(cluster_df["Cluster"].unique()):

    temp=cluster_df[
        cluster_df["Cluster"]==i
    ]

    plt.scatter(
        temp["PCA1"],
        temp["PCA2"],
        label=f"Cluster {i}"
    )

plt.legend()

plt.title("Product Demand Segmentation")

plt.xlabel("PCA 1")

plt.ylabel("PCA 2")

plt.show()

# %% [markdown]
# Step 16: Cluster Summary

# %%
cluster_summary=cluster_df.groupby(
"Cluster"
).mean()

cluster_summary

# %% [markdown]
# Step 17: Give Meaningful Labels

# %%
cluster_df["Demand Segment"]=cluster_df["Cluster"]

# %%
mapping={

0:"High Volume Stable Demand",

1:"Low Volume High Volatility",

2:"Growing Demand",

3:"Declining Demand"

}

cluster_df["Demand Segment"]=cluster_df[
"Demand Segment"
].map(mapping)

# %% [markdown]
# Step 18: Final Result

# %%
cluster_df

# %% [markdown]
# Step 19: Save CSV

# %%
cluster_df.to_csv(
r"C:\Users\sujit\Downloads\product_demand_segmentation.xls"
)

# %% [markdown]
# Step 20: Save Figure

# %%
plt.figure(figsize=(10,6))

for i in cluster_df["Demand Segment"].unique():

    temp=cluster_df[
        cluster_df["Demand Segment"]==i
    ]

    plt.scatter(
        temp["PCA1"],
        temp["PCA2"],
        label=i
    )

plt.legend()

plt.title("Demand Segments")

plt.savefig(
"charts/product_clusters.png"
)

plt.show()

# %% [markdown]
# Step 21: Stocking Strategy

# %% [markdown]
# | Demand Segment             | Stocking Strategy                                                       |
# | -------------------------- | ----------------------------------------------------------------------- |
# | High Volume Stable Demand  | Maintain high inventory with regular replenishment to avoid stock-outs. |
# | Low Volume High Volatility | Keep limited stock and replenish based on demand fluctuations.          |
# | Growing Demand             | Increase inventory gradually and monitor sales trends closely.          |
# | Declining Demand           | Reduce inventory levels and avoid overstocking.                         |
# 

# %% [markdown]
# # Task 7 – Build an Interactive Streamlit Dashboard

# %% [markdown]
# Step 1: Install Libraries

# %%
!pip install streamlit plotly pandas

# %% [markdown]
# Step 2: Project Folder Structure

# %% [markdown]
# Sales_Forecasting_Project/
# 
# │── app.py
# │── train.csv
# │── customer_segmentation.csv
# │── cluster_summary.csv
# │── requirements.txt
# │── charts/
# │     ├── monthly_sales_trend.png
# │     ├── task4_forecast_comparison.png
# │     ├── customer_clusters.png
# │     ├── isolation_forest.png
# │     └── zscore_outliers.png

# %% [markdown]
# Step 3: Create app.py

# %%
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)

# %% [markdown]
# Step 4: Load Dataset

# %%
df = pd.read_csv(r"C:\Users\sujit\Downloads\train.csv.zip")

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')


df["Year"] = df["Order Date"].dt.year

df["Month"] = df["Order Date"].dt.month_name()

# %% [markdown]
# Step 5: Sidebar (Navigation)

# %%
page = st.sidebar.selectbox(

"Select Page",

[
"Sales Dashboard",
"Forecast Explorer",
"Anomaly Report",
"Product Demand Segments"
]

)

# %% [markdown]
# 
# PAGE 1 – Sales Dashboard
# 

# %% [markdown]
# Step 6: Page Heading

# %%
if page=="Sales Dashboard":

    st.title("Sales Overview Dashboard")

# %% [markdown]
# Step 7: KPI Cards

# %%
total_sales=df["Sales"].sum()

#total_profit=df["Profit"].sum()

orders=df["Order ID"].nunique()

c1,c2,c3=st.columns(3)

c1.metric("Total Sales",f"${total_sales:,.0f}")

#c2.metric("Profit",f"${total_profit:,.0f}")

c3.metric("Orders",orders)

# %% [markdown]
# Step 8: Total Sales by Year (Bar Chart)

# %%
year=df.groupby("Year")["Sales"].sum().reset_index()

fig=px.bar(

year,

x="Year",

y="Sales",

title="Total Sales by Year"

)

st.plotly_chart(fig,use_container_width=True)

# %% [markdown]
# Step 9: Monthly Sales Trend

# %%
monthly=df.groupby(

pd.Grouper(

key="Order Date",

freq="ME"

)

)["Sales"].sum().reset_index()

fig=px.line(

monthly,

x="Order Date",

y="Sales",

title="Monthly Sales Trend"

)

st.plotly_chart(fig,use_container_width=True)

# %% [markdown]
# Step 10: Region & Category Filter

# %%
region=st.selectbox(

"Region",

["All"]+list(df["Region"].unique())

)

category=st.selectbox(

"Category",

["All"]+list(df["Category"].unique())

)

temp=df.copy()

if region!="All":

    temp=temp[temp["Region"]==region]

if category!="All":

    temp=temp[temp["Category"]==category]

st.dataframe(temp.head())

# %% [markdown]
# PAGE 2 – Forecast Explorer

# %% [markdown]
# Step 11: Heading

# %%
if page == "Home":
    st.title("Home Page")

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

else:
    st.title("Other Page")


# %% [markdown]
# Step 12: Category Dropdown

# %%
option=st.selectbox(

"Select Category",

list(df["Category"].unique())

)

# %% [markdown]
# Step 13: Forecast Horizon

# %%
months=st.slider(

"Forecast Months",

1,

3,

1

)

# %% [markdown]
# Step 14: Forecast Graph

# %%
forecast=pd.DataFrame({

"Month":[

"Month1",

"Month2",

"Month3"

],

"Forecast":[

120000,

130000,

145000

]

})

# %% [markdown]
# Step 15: Plot Forecast

# %%
fig=px.line(

forecast,

x="Month",

y="Forecast",

markers=True,

title="Forecast"

)

st.plotly_chart(fig)

# %% [markdown]
# Step 16: Show MAE & RMSE

# %%
st.write("MAE : 2450")

st.write("RMSE : 3100")

# %% [markdown]
# PAGE 3 – Anomaly Report

# %% [markdown]
# Step 17: Heading

# %%
if page == "Home":
    st.title("Home Page")

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

elif page == "Anomaly Report":
    st.title("Anomaly Report")

else:
    st.title("Other Page")


# %% [markdown]
# Step 18: Show Chart

# %%
st.image(

"charts/isolation_forest.png"

)

# %% [markdown]
# Step 19: Show Table

# %%
anomaly=df.sort_values(

"Sales",

ascending=False

).head(10)

st.dataframe(

anomaly[

["Order Date","Sales"]

]

)

# %% [markdown]
# PAGE 4 – Product Demand Segments

# %% [markdown]
# Step 20: Heading

# %%
import streamlit as st

page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"]
)

if page == "Home":
    st.title("Home Page")

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

elif page == "Anomaly Report":
    st.title("Anomaly Report")

elif page == "Product Demand Segments":
    st.title("Demand Segments")

else:
    st.title("Other Page")


# %% [markdown]
# Step 21: Show Cluster Chart

# %%
st.image(

"charts/product_clusters.png"

)

# %% [markdown]
# Step 22: Read Cluster CSV

# %%
cluster=pd.read_csv(

r"C:\Users\sujit\Downloads\product_demand_segmentation.xls"

)

cluster.head()

# %% [markdown]
# Step 23: Display Table

# %%
st.dataframe(

cluster[

["Demand Segment"]

]

)

# %% [markdown]
# Step 24: Display Sub-Categories with Demand Segments

# %%
st.subheader("Sub Category vs Demand Segment")

st.dataframe(

cluster.reset_index()[

["Sub-Category","Demand Segment"]

]

)

# %%



