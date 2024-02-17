# Final Project
# Spring 2023 Time-Series
# HaeLee Kim

import csv
# with open(r'C:\Users\haele\Desktop\23 Spring Time-Series\Final Project\household_power_consumption.txt') as infile:
#     with open('TS_Final_Dataset.csv', 'w', newline='') as outfile:
#         writer = csv.writer(outfile, delimiter=',')
#         writer.writerow(['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])
#         for line in infile:
#             line = line.strip().replace(';', ',')
#             writer.writerow(line.split(','))
#
# import pandas as pd
# df = pd.read_csv(r'C:\Users\haele\Desktop\23 Spring Time-Series\Final Project\TS_Final_Dataset.csv')
# df = df.drop(0)
# df = df.reset_index(drop=True)
# print(df.info())
# print(df.head())
# df = df.replace('?', pd.NA)
# if df.isna().any().any():
#     print("Yes, there are missing values in the DataFrame.")
#     # print(df.isna().sum())
# else:
#     print("No, there are no missing values in the DataFrame.")
#
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
# df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
# df = df.set_index("DateTime")
# df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
# df['Global_reactive_power'] = pd.to_numeric(df['Global_reactive_power'], errors='coerce')
# df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
# df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
# df['Sub_metering_1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
# df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')
# df['Sub_metering_3'] = pd.to_numeric(df['Sub_metering_3'], errors='coerce')
#
# # Resample the data to hourly frequency and aggregate the values using mean
# hourly_means = df.resample("H").mean(numeric_only=True)
# df = df.fillna(hourly_means.ffill())
# df_hourly = df.resample("H").mean(numeric_only=True)
# # print(df_hourly.head())
# # print(df_hourly.shape)
# # print(df_hourly.isna().sum())
# # df_hourly.to_csv('TS_Final_Dataset_Hourly.csv', index=True)
#
# # Resample the data by 6-hourly interval and calculate the mean for each group
# df_six_hourly = df_hourly.resample("6H").mean(numeric_only=True)
# print(df_six_hourly.head())
# print(df_six_hourly.shape)
# print(df_six_hourly.isna().sum())
# df_six_hourly.to_csv('TS_Final_Dataset_Six_Hourly.csv', index=True)

# 6. Description of the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math
import statsmodels.api as sm
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv(r'C:\Users\haele\Desktop\23 Spring Time-Series\Final Project\TS_Final_Dataset_Six_Hourly.csv', parse_dates=['DateTime'], index_col='DateTime')
y = df.index # Extract the datetime index as the y variable
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.shape)

# 6-b. Plot of the dependent variable versus time
global_active = df['Global_active_power']
global_reactive = df['Global_reactive_power']
plt.figure()
plt.plot(y, global_active, label='Global_active_power')
plt.grid()
plt.title('6-b. Global Active Power between 2007 and 2010')
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Power (Voltage)')
plt.tight_layout()
plt.show()

# 6-c. ACF/PACF of the dependent variable
def ACF_PACF_Plot(y,lags):
 acf = sm.tsa.stattools.acf(y, nlags=lags)
 pacf = sm.tsa.stattools.pacf(y, nlags=lags)
 fig = plt.figure()
 plt.subplot(211)
 plt.title(f'6-c. ACF/PACF of the dataset')
 plot_acf(y, ax=plt.gca(), lags=lags)
 plt.subplot(212)
 plot_pacf(y, ax=plt.gca(), lags=lags)
 fig.tight_layout(pad=3)
 plt.show()
ACF_PACF_Plot(df['Global_active_power'],50)

# 6-d. ACF/PACF of the dependent variable
corr_matrix = df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('6-d. Correlation Matrix')
plt.show()

# 6-e. Split the dataset into train set (80%) and test set (20%)
X = df.drop('Global_active_power', axis=1)
y = df['Global_active_power']
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 7. Stationarity
# 7-a. ADF-test
from statsmodels.tsa.stattools import adfuller
def ADF_Cal(x):
 result = adfuller(x)
 print("ADF Statistic: %f" %result[0])
 print('p-value: %f' % result[1])
 print('Critical Values:')
 for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
ADF_Cal(df['Global_active_power'])
print('7-a. The ADF result shows that p-value is 0.00 below a threshold (5%) and it suggests we reject the null hypothesis. It means Global Active Power is stationary.')

# 7-b. KPSS-test
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
 print ('Results of KPSS Test:')
 kpsstest = kpss(timeseries, regression='c', nlags="auto")
 kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
 for key,value in kpsstest[3].items():
    kpss_output['Critical Value (%s)'%key] = value
 print (kpss_output)
kpss_test(df['Global_active_power'])
print('7-b. The result shows that p-value is 0.058 above a threshold (5%) and it suggests we cannot reject the null hypothesis. It means Global Active Power is stationary.')

# 7-c. rolling mean and variance
def Cal_rolling_mean_var(y, N):
    rolling_mean = np.zeros(N)
    rolling_var = np.zeros(N)
    for i in range(N):
        rolling_mean[i] = np.mean(y[:i+1])
        rolling_var[i] = np.var(y[:i+1], ddof=0)
    # print("Rolling Mean:\n", rolling_mean[-5:])
    # print("Rolling Variance:\n", rolling_var[-5:])

    fig, ax = plt.subplots(2,1)
    for j in range(2):
        if j==0:
            ax[j].plot(rolling_mean)
            ax[j].set_title('7-c. Rolling Mean')
        else:
            ax[j].plot(rolling_var, label='Varying variance')
            ax[j].set_title('7-c. Rolling Variance')
            ax[j].legend(loc='lower right')
        ax[j].set_xlabel('Samples')
        ax[j].set_ylabel('Magnitude')
    fig.tight_layout()
    plt.show()
Cal_rolling_mean_var(df['Global_active_power'], len(df))

# 8. Time series Decomposition
from statsmodels.tsa.seasonal import STL
Temp = pd.Series(df['Global_active_power'].values, index=df.index, name = 'Global Active Power')
STL = STL(Temp, period=365*4)
res = STL.fit()
fig = res.plot()
plt.show()
T = res.trend
S = res.seasonal
R = res.resid
plt.figure()
plt.plot(T.index, T.values, label = 'trend')
plt.plot(S.index, R.values, label = 'residuals')
plt.plot(R.index, S.values, label = 'Seasonal')
plt.title('8. STL decomposition')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
# Calculate the strength of trend
FT = max(0, 1 - np.var(R) / (np.var(T + R)))
print(f'8. The strength of trend for this data set is {FT}')
# Calculate the strength of seasonality
FS = max(0, 1 - np.var(R) / np.var(S + R))
print(f'8. The strength of Seasonality for this data set is {FS}')

# 9. Holt-Winters method
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(y_train, seasonal_periods=365*4, trend='add', seasonal='add')
model_fit = model.fit()
y_pred = model_fit.forecast(len(X_test))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print('9. Mean Squared Error (MSE) of Holt-Winters method:', mse)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='y test')
plt.plot(y_pred, label='Holt-Winters method prediction')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('9. y test and Holt-Winters method prediction')
plt.legend()
plt.show()

# 10. Feature selection/elimination
# 10-a. SVD analysis
H = X.T @ X
s, d, v = np.linalg.svd(H)
print("10-a. Step 1. SingularValues = ", d)

# 10-b  condition number
con_num = np.linalg.cond(X)
print("10-b. Step 1. ConditionNumber =", con_num)

# 10-c. VIF test and Check/Eliminate multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_tuples = [(vif.iloc[i, 0], vif.iloc[i, 1]) for i in range(vif.shape[0])]
    return vif_tuples
X1 = sm.add_constant(X) # add intercept term
model1 = sm.OLS(y, X1).fit()
print(model1.summary())
vif_scores1 = calculate_vif(X)
print("step 1. VIF:", vif_scores1)
# step2
Xa = X.drop('Global_intensity', axis=1)
H = X.T @ Xa
s, d, v = np.linalg.svd(H)
print("10-a. step 2. SingularValues = ", d)
con_num = np.linalg.cond(Xa)
print("10-b. step 2. ConditionNumber =", con_num)
X2 = sm.add_constant(Xa)
model2 = sm.OLS(y, X2).fit()
print(model2.summary())
vif_scores2 = calculate_vif(Xa)
print("step 2. VIF:", vif_scores2)
# step3
Xb = Xa.drop('Global_reactive_power', axis=1)
H = X.T @ Xb
s, d, v = np.linalg.svd(H)
print("10-a. step 3. SingularValues = ", d)
con_num = np.linalg.cond(Xb)
print("10-b. step 3. ConditionNumber =", con_num)
X3 = sm.add_constant(Xb) # add intercept term
model3 = sm.OLS(y, X3).fit()
print(model3.summary())
vif_scores3 = calculate_vif(Xb)
print("step 3. VIF:", vif_scores3)


# 10-d. backward stepwise regression
# standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = pd.DataFrame(scaler.fit_transform(Xb), columns=Xb.columns)
y_std = pd.DataFrame(scaler.fit_transform(y.values.reshape(-1, 1)))
# unknown coefficients/ statsmodels package and OLS function
train_size = int(len(df) * 0.8)
X_train_std, X_test_std= X_std[:train_size], X_std[train_size:]
y_train_std, y_test_std = y_std[:train_size], y_std[train_size:]
X_train1_std= sm.add_constant(X_train_std)
Y_train_std = np.array(y_train_std).reshape((-1, 1))
# backward stepwise regression
keep = ['Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
eliminate = []
OLS = sm.OLS(Y_train_std, sm.add_constant(X_train1_std)).fit()
pvalues = OLS.pvalues
max_pvalue = pvalues.drop('const').max()
while max_pvalue >= 0.05 and len(keep) > 1:
    k = pvalues.drop('const').idxmax()
    eliminate.append(k)
    keep.remove(k)
    X_train_sub = X_train_std[keep]
    OLS = sm.OLS(Y_train_std, sm.add_constant(X_train_sub)).fit()
    pvalues = OLS.pvalues
    max_pvalue = pvalues.drop('const').max()
print(OLS.summary())
print(f"10-d. Features to keep: {keep}")
print(f"10-d. Features to eliminate: {eliminate}")

# prediction for the test and plot
X_train_selected = sm.add_constant(X_train_std[keep])
new_OLS = sm.OLS(y_train_std, X_train_selected).fit()
y_train_pred = new_OLS.predict(X_train_selected)
X_test_selected = sm.add_constant(X_test_std[keep])
y_test_pred = new_OLS.predict(X_test_selected)
plt.plot(y_train_std, label='Train')
plt.plot(y_test_std, label='Test')
plt.plot(y_test_pred, label='Predicted Test')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.title('10-d. Train, Test, and Predicted Values')
plt.legend()
plt.show()

# 11.Base-models: average, naïve, drift, simple and exponential smoothing.
# 11-a. Average Forecast Method
# 1-step ahead prediction (training set)
y = y_train_std.values
y_hat = np.zeros(len(y))
e = np.zeros(len(y))
for i in range(1, len(y)):
    y_hat[i] = np.mean(y[:i])
    e[i] = y[i] - y_hat[i]
e2 = np.square(e)
# h-step ahead forecast (testing set)
yh = y_test_std.values
yh_hat = np.mean(y)*np.ones(len(yh))
eh = np.zeros(len(yh))
for i in range(len(yh)):
    eh[i] = yh[i] - yh_hat[i]
eh2 = np.square(eh)
# Plot the test set, training set and the h-step forecast in one graph.
plt.plot(y, color='red', marker='o', label='Training set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh, color='blue', marker='x', label='Testing set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh_hat, color='green', marker='d', label='H-step forecast')
plt.title('11-a. Training set, Testing set, and H-step Forecast with Average Forecast Method')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.legend()
plt.show()
# Average: MSE of prediction errors and the forecast errors
mse_prediction = np.mean(e2[2:])
mse_forecast = np.mean(eh2)
table = [["MSE of prediction errors (Average)","MSE of forecast errors (Average)"], [mse_prediction, mse_forecast]]
print(table)
rmse_prediction = np.sqrt(mse_prediction)
rmse_forecast = np.sqrt(mse_forecast)
table = [["RMSE of prediction errors (Average)","RMSE of forecast errors (Average)"], [rmse_prediction, rmse_forecast]]
print(table)

# 11-b. Naive Method
# 1-step ahead prediction (training set)
y = y_train_std.values
y_hat = np.zeros(len(y))
e = np.zeros(len(y))
for i in range(1, len(y)):
    y_hat[i] = y[i-1]
    e[i] = y[i] - y_hat[i]
e2 = np.square(e)
# h-step ahead forecast (testing set)
yh = y_test_std.values
yh_hat = y[-1]*np.ones(len(yh))
eh = np.zeros(len(yh))
for i in range(len(yh)):
    eh[i] = yh[i] - y[-1]
eh2 = np.square(eh)
# Plot the test set, training set and the h-step forecast in one graph.
plt.plot(y, color='red', marker='o', label='Training set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh, color='blue', marker='x', label='Testing set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh_hat, color='green', marker='d', label='H-step forecast')
plt.title('11-b. Training set, Testing set, and H-step Forecast with Naive Method')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.legend()
plt.show()
# Naive MSE
mse_prediction = np.mean(e2[2:])
mse_forecast = np.mean(eh2)
table = [["MSE of prediction errors (Naive)","MSE of forecast errors (Naive)"], [mse_prediction, mse_forecast]]
print(table)
rmse_prediction = np.sqrt(mse_prediction)
rmse_forecast = np.sqrt(mse_forecast)
table = [["RMSE of prediction errors (Naive)","RMSE of forecast errors (Naive)"], [rmse_prediction, rmse_forecast]]
print(table)

# 11-c. Drift Method
# 1-step ahead prediction (training set)
y = y_train_std.values
y_hat = np.zeros(len(y))
e = np.zeros(len(y))
for i in range(2, len(y)):
    y_hat[i] = y[i-1]+1*(y[i-1]-y[0])/((i+1-1)-1)
    e[i] = y[i] - y_hat[i]
e2 = np.square(e)
# h-step ahead forecast (testing set)
yh = y_test_std.values
yh_hat = np.zeros(len(yh))
eh = np.zeros(len(yh))
for i in range(len(yh)):
    yh_hat[i] = y[-1] + (i+1)*(y[-1]-y[0])/(len(y)-1)
    eh[i] = yh[i] - yh_hat[i]
eh2 = np.square(eh)
# Plot the test set, training set and the h-step forecast in one graph.
plt.plot(y, color='red', marker='o', label='Training set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh, color='blue', marker='x', label='Testing set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh_hat, color='green', marker='d', label='H-step forecast')
plt.title('11-c. Training set, Testing set, and H-step Forecast with Drift Method')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.legend()
plt.show()
# Drift MSE
mse_prediction = np.mean(e2[2:])
mse_forecast = np.mean(eh2)
table = [["MSE of prediction errors (Drift)","MSE of forecast errors (Drift)"], [mse_prediction, mse_forecast]]
print(table)
rmse_prediction = np.sqrt(mse_prediction)
rmse_forecast = np.sqrt(mse_forecast)
table = [["RMSE of prediction errors (Drift)","RMSE of forecast errors (Drift)"], [rmse_prediction, rmse_forecast]]
print(table)

# 11-d Simple Exponential Method
# 1-step ahead prediction (training set)
y = y_train_std.values
y_hat = np.zeros(len(y))
e = np.zeros(len(y))
prehat= y[0]
alpha = 0.5
for i in range(1, len(y)):
    y_hat[i] = alpha*y[i-1] + (1-alpha)*prehat
    prehat = y_hat[i]
    e[i] = y[i] - y_hat[i]
e2 = np.square(e)
# h-step ahead forecast (testing set)
yh = y_test_std.values
yh_hat = np.zeros(len(yh))
eh = np.zeros(len(yh))
for i in range(len(yh)):
    yh_hat[i] = alpha*y[-1] + (1-alpha)*y_hat[-1]
    eh[i] = yh[i] - yh_hat[i]
eh2 = np.square(eh)
# Plot the test set, training set and the h-step forecast in one graph.
plt.plot(y, color='red', marker='o', label='Training set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh, color='blue', marker='x', label='Testing set')
plt.plot(np.arange(len(y), len(y)+len(yh)), yh_hat, color='green', marker='d', label='H-step forecast')
plt.title('11-d. Training set, Testing set, and H-step Forecast with SES Method (alpha=0.5)')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.legend()
plt.show()
## SES 0.5 MSE
mse_prediction = np.mean(e2[2:])
mse_forecast = np.mean(eh2)
table = [["MSE of prediction errors (SES alpha 0.5)","MSE of forecast errors (SES alpha 0.5)"], [mse_prediction, mse_forecast]]
print(table)
rmse_prediction = np.sqrt(mse_prediction)
rmse_forecast = np.sqrt(mse_forecast)
table = [["RMSE of prediction errors (SES alpha 0.5)","RMSE of forecast errors (SES alpha 0.5)"], [rmse_prediction, rmse_forecast]]
print(table)

# compare the SARIMA model performance with the base model predication
# # SARIMA model
# sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit()
# sarima_pred = sarima_model.forecast(len(test))
# sarima_rmse = rmse(test, sarima_pred)
#
# # Compare the performance of the base models and SARIMA model
# # print('Naive model RMSE:', naive_rmse)
# # print('Simple average model RMSE:', avg_rmse)
# # print('Drift model RMSE:', drift_rmse)
# # print('Simple exponential smoothing model RMSE:', ses_rmse)
# # print("Holt's exponential smoothing model RMSE:", holt_rmse)
# print('SARIMA model RMSE:', sarima_rmse)

# 12. multiple linear regression
# Perform one-step ahead prediction and compare the performance versus the test set
# 1-step ahead prediction (training set)
model = sm.OLS(y_train_std, sm.add_constant(X_train_std)).fit()
# 1-step ahead predictions for training set
y_pred_train = model.predict(sm.add_constant(X_train_std))
# 1-step ahead predictions for test set
y_pred_test = model.predict(sm.add_constant(X_test_std))
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(y_train_std, color='red', marker='o', label='Training set')
ax.plot(y_test_std, color='blue', marker='x', label='Testing set')
ax.plot(y_train_std.index, y_pred_train, color='green', marker='d', label='1-step prediction')
ax.set_title('12-a-1. Training set, Testing set, and 1-step ahead prediction with Multiple Linear Regression')
ax.set_xlabel('Time')
ax.set_ylabel('Observation')
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(y_train_std, color='red', marker='o', label='Training set')
ax.plot(y_test_std, color='blue', marker='x', label='Testing set')
ax.plot(y_test_std.index, y_pred_test, color='green', marker='d', label='h-step prediction')
ax.set_title('12-a-2. Training set, Testing set, and h-step ahead prediction with Multiple Linear Regression')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()

mse_train = mean_squared_error(y_train_std, y_pred_train)
print(f"12-a. MSE (One-step Ahead Prediction): {mse_train:.4f}")
y_pred_test = model_fit.predict(start=len(y_train_std), end=len(y_train_std)+len(y_test_std)-1)
mse_test = mean_squared_error(y_test_std, y_pred_test)
print(f"12-a. MSE (Test: Forecasting): {mse_test:.4f}")


# 12-b. F-test, t-test
t_values = model.tvalues
p_values = model.pvalues
result_t = pd.DataFrame({'t-values': t_values, 'p-values': p_values})
print(f'12-b. {result_t}')
f_test = model.f_test(np.identity(len(model.params)))
f_value = f_test.fvalue
p_value = f_test.pvalue
print(f'12-b. f-values: {f_value}, p-values: {p_value}')

# 12-c. AIC, BIC, RMSE, and Adjusted R-squared
print(model.summary())
print("12-c. AIC: ", model.aic)
print("12-c. BIC: ", model.bic)
mse = mean_squared_error(y_train_std, y_pred_train)
rmse = np.sqrt(mse)
print("12-c. RMSE: ", rmse)
print("12-c. Adjusted R-squared: ", model.rsquared_adj)

# 12-d. ACF of residuals.
# prediction errors and plot the ACF of prediction errors
y_train_std1 = y_train_std.to_numpy().flatten()
y_pred_train1 = y_pred_train.to_numpy()
residuals = y_train_std1 - y_pred_train1
def acf_fuc(y, lag):
    y1 = y[lag:]
    y2 = y[:(len(y)-lag)]
    denominator = np.sum((y1 - np.mean(y)) * (y2 - np.mean(y)))
    numerator = (np.var(y)*len(y))
    ans = denominator/numerator
    return ans
acf_list = []
for i in range(21):
    acf_list.append(acf_fuc(residuals, i))
new_acf_list = acf_list[::-1] + acf_list[1:]
lag = 20
x = np.arange(-lag, lag + 1)
y = np.array(new_acf_list)
plt.stem(x, y, markerfmt='ro', basefmt='b')
conf_int = 0.05
upper_CI = 1.96 / math.sqrt(len(residuals))
lower_CI = -1.96 / math.sqrt(len(residuals))
plt.fill_between(x, lower_CI, upper_CI, color='blue', alpha=0.2)
plt.title('12-d. ACF of Prediction Errors')
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# 12-e. Q-value
acf = np.zeros(lag+1)
for i in range(len(acf)):
    acf[i] = acf_fuc(residuals[2:],i)
acf2 = np.square(acf)
Q = len(residuals[2:])*np.sum(acf2[1:])
print("12-e. Q value is", Q)

# 12-f. Mean and variance of the residuals.
mean_residuals = np.mean(residuals)
var_residuals = np.var(residuals)
print("12-f. Mean of the residuals is", mean_residuals)
print("12-f. Variance of the residuals is", var_residuals)

# 13. ARMA and ARIMA and SARIMA model order determination
# 13-a.	Preliminary model development procedures and results (ARMA model order determination).
from statsmodels.tsa.stattools import acf
# ry = acf(y_train_std, nlags=50)
ry = acf(y_train_std, nlags=50)
ry2 = np.concatenate((ry[::-1], ry[1:]))
c = len(ry2)//2
# display GPAC table for the default values of k and j
def compute_gpac_table(data,j_max=15, k_max=15):
    gpac_array = np.zeros((j_max, k_max))
    for j in range(0, j_max):
        for k in range(1, k_max+1):
            if k == 1:
                gpac_array[j, k-1] = ry2[c+j+1] / ry2[c+j]
            else:
                denom = np.zeros((k, k))
                for row_denom in range(k):
                    end = c - j + k - row_denom
                    start = c - j - row_denom
                    row = ry2[start:end]
                    denom[row_denom, :] = row
                numer = denom.copy()
                start = c + j + 1
                end = start + k
                numer[:, -1] = ry2[start:end]
                if np.linalg.det(denom) == 0:
                    pai = np.inf
                else:
                    pai = np.linalg.det(numer)/np.linalg.det(denom)
                gpac_array[j, k-1] = pai
    return gpac_array

gpac_array = compute_gpac_table(ry)
col_labels = [k for k in range(1, gpac_array.shape[1]+1)]
row_labels = [j for j in range(0, gpac_array.shape[0])]
sns.set(font_scale=1)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(gpac_array, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=col_labels, yticklabels=row_labels, ax=ax)
ax.set_title('13-a. Generalized Partial Autocorrelation(GPAC) Table: ARMA', fontsize=16)
plt.show()
print("Picked orders using GPAC table are (4,0), (4,1), (4,4), (8,0), (8,1)")

# 13-b.	Should include discussion of the autocorrelation function and the GPAC.
# Include a plot of the autocorrelation function and the GPAC table within this section).
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def ACF_PACF_Plot(y,lags):
 acf = sm.tsa.stattools.acf(y, nlags=lags)
 pacf = sm.tsa.stattools.pacf(y, nlags=lags)
 fig = plt.figure()
 plt.subplot(211)
 plt.title(f'13-b. ACF/PACF of the data)')
 plot_acf(y, ax=plt.gca(), lags=lags)
 plt.subplot(212)
 plot_pacf(y, ax=plt.gca(), lags=lags)
 fig.tight_layout(pad=3)
 plt.show()
ACF_PACF_Plot(y_train_std,50)

# 13-c. Include the GPAC table in your report and highlight the estimated order.

# 14. Estimate ARMA model parameters using the Levenberg Marquardt algorithm.
# Display the parameter estimates, the standard deviation of the parameter estimates and confidence intervals.

orders = [(4, 0), (4, 4), (8, 0), (8, 4)]
mse_results = []
for order in orders:
    ar_order = order[0]
    ma_order = order[1]
    model = sm.tsa.ARIMA(y_train_std, order=(ar_order, 0, ma_order)).fit()
    coefficients = model.params.round(3)
    y_train_pred = model.predict()
    mse_train = mean_squared_error(y_train_std, y_train_pred)
    y_test_pred = model.predict(start=len(y_train_std), end=len(y_train_std) + len(y_test_std)-1)
    mse_test = mean_squared_error(y_test_std, y_test_pred)
    mse_results.append((order, mse_train, mse_test))
    print(f"\n14. Estimated parameters for ARMA({ar_order},{ma_order}):")
    print(f"\n14. Estimated parameters for ARMA({ar_order},{ma_order}):")
    print("14. Estimated AR coefficients:\n", coefficients[1:ar_order + 1])
    print("14. Estimated MA coefficients:\n", coefficients[ar_order + 1:-1])
    n = ar_order + ma_order
    CI = np.zeros((n, 2))
    std_err = np.zeros(n)
    for i in range(n):
        std_err[i] = np.sqrt(model.cov_params().iloc[i, i])
        CI[i, 0] = coefficients[i] - 2 * std_err[i]
        CI[i, 1] = coefficients[i] + 2 * std_err[i]
        print(f"14-{i}. {i}th standard deviation: {round(std_err[i], 3)}")
        print(f"14-{i}. CI of {i}th coefficient {coefficients[i]}: [{round(CI[i, 0], 3)}, {round(CI[i, 1], 3)}]")
    print(model.summary())

print("14 & 17. MSE Results of ARMA:")
for result in mse_results:
    print(f"ARMA{result[0]}: Train MSE (Prediction) = {result[1]}, Test MSE (Forecasting) = {result[2]}\n\n")
print("\n\n 14. Conclusion: ARMA(4,4) works well! ARMA(4,0) will be also considered in SARIMA.")

# 15. Diagnostic Analysis
# 15-a. Diagnostic tests (confidence intervals, zero/pole cancellation, chi-square test).
ar_order = 4
ma_order = 4
model = sm.tsa.ARIMA(y_train_std, order=(ar_order, 0, ma_order)).fit()
coefficients = model.params.round(3)
n = ar_order + ma_order
CI = np.zeros((n, 2))
std_err = np.zeros(n)
for i in range(n):
    std_err[i] = np.sqrt(model.cov_params().iloc[i, i])
    CI[i, 0] = coefficients[i] - 2 * std_err[i]
    CI[i, 1] = coefficients[i] + 2 * std_err[i]
    print(f"15-a. {i}th standard deviation: {round(std_err[i], 3)}")
    print(f"15-a. CI of {i}th coefficient {coefficients[i]}: [{round(CI[i, 0], 3)}, {round(CI[i, 1], 3)}]")
    if CI[i, 0] > 0 or CI[i, 1] < 0:
        print(f"15-a. {i}th coefficient is statistically important.")
    else:
        print(f"15-a. {i}th coefficient is statistically not important.")
# check for zero/pole cancellation
poles = np.roots(np.append(1, -model.arparams))
zeros = np.roots(np.append(1, model.maparams))
print("15-a. Poles - AR roots: ", poles)
print("15-a. Zeros - MA roots: ", zeros)
# chi-square test for residuals
from statsmodels.stats.diagnostic import acorr_ljungbox
residuals = model.resid
lags = min(10, len(residuals)-1)
lbvalue, pvalue = acorr_ljungbox(residuals, lags=lags, boxpierce=False)
# print("15-a. Ljung-Box (L%d) (Q): %.2f" % (lags, lbvalue[-1]))

# 15-b. the estimated variance of the error and the estimated covariance of the estimated parameters.
error_var = np.round(model.scale, 3)
print(f"15-b. Estimated variance of the error for ARMA({ar_order},{ma_order}): {error_var}")
# Estimated covariance of the estimated parameters
param_cov = np.round(model.cov_params().to_numpy(), 3)
print(f"15-b. Estimated covariance of the estimated parameters for ARMA({ar_order},{ma_order}): \n {param_cov}")

# 15-c.
print(model.summary())
print("the constant of coefficent is closed to zero and CI includes zero, which means this is unbiased.")

# 15-d.
var_residuals = np.var(residuals)
print("15-d. Variance of the residual errors is", var_residuals)
y_test_std = y_test_std.to_numpy().flatten()
y_pred_test = y_pred_test.to_numpy()
forecast_errors = y_test_std - y_pred_test
var_forecast_errors = np.var(forecast_errors)
print("15-d. Variance of the forecast errors is", var_forecast_errors)

# # 15-e. ARIMA or SARIMA model may better represents the dataset
# # 15-e. ex1.
# model = sm.tsa.SARIMAX(y_train_std, order=(0,0,0), seasonal_order=(1,0,0,4))
# model_fit = model.fit()
# y_model_hat = model_fit.predict(start=1, end=len(y_train_std)-1)
# plt.figure(figsize=(10, 6))
# plt.plot(y_train_std[1:], label='y')
# plt.plot(y_model_hat, label='1-step')
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.title('15-e. EX1. y and 1-Step Ahead Prediction with SARIMA')
# plt.legend()
# plt.show()
# print(model_fit.summary())
#
# y_pred_train = y_model_hat.to_numpy()
# residuals = y_train_std1[1:] - y_pred_train
# acf_list = []
# lag = 60
# for i in range(lag+1):
#     acf_list.append(acf_fuc(residuals, i))
# new_acf_list = acf_list[::-1] + acf_list[1:]
# x = np.arange(-lag, lag + 1)
# y = np.array(new_acf_list)
# plt.stem(x, y, markerfmt='ro', basefmt='b')
# conf_int = 0.05
# upper_CI = 1.96 / math.sqrt(len(residuals))
# lower_CI = -1.96 / math.sqrt(len(residuals))
# plt.fill_between(x, lower_CI, upper_CI, color='blue', alpha=0.2)
# plt.title('15-e. EX1. ACF of Prediction Errors')
# plt.xlabel('Lags')
# plt.ylabel('Magnitude')
# plt.tight_layout()
# plt.show()
#
# mse_train = mean_squared_error(y_train_std[1:], y_model_hat)
# print(f"15-a. EX5-a. MSE of the SARIMA model (Training: Prediction): {mse_train:.4f}")
# y_pred_test = model_fit.predict(start=len(y_train_std), end=len(y_train_std)+len(y_test_std)-1)
# mse_test = mean_squared_error(y_test_std, y_pred_test)
# print(f"15-e. EX5-b. MSE of the SARIMA model (Test: Forecasting): {mse_test:.4f}")
#
# # 15-e. ex2.
# model = sm.tsa.SARIMAX(y_train_std, order=(0,0,0), seasonal_order=(1,0,1,4))
# model_fit = model.fit()
# y_model_hat = model_fit.predict(start=1, end=len(y_train_std)-1)
# plt.figure(figsize=(10, 6))
# plt.plot(y_train_std[1:], label='y')
# plt.plot(y_model_hat, label='1-step')
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.title('15-e. EX2. y and 1-Step Ahead Prediction with SARIMA')
# plt.legend()
# plt.show()
# print(model_fit.summary())
#
# y_pred_train = y_model_hat.to_numpy()
# residuals = y_train_std1[1:] - y_pred_train
# acf_list = []
# lag = 60
# for i in range(lag+1):
#     acf_list.append(acf_fuc(residuals, i))
# new_acf_list = acf_list[::-1] + acf_list[1:]
# x = np.arange(-lag, lag + 1)
# y = np.array(new_acf_list)
# plt.stem(x, y, markerfmt='ro', basefmt='b')
# conf_int = 0.05
# upper_CI = 1.96 / math.sqrt(len(residuals))
# lower_CI = -1.96 / math.sqrt(len(residuals))
# plt.fill_between(x, lower_CI, upper_CI, color='blue', alpha=0.2)
# plt.title('15-e. EX2. ACF of Prediction Errors')
# plt.xlabel('Lags')
# plt.ylabel('Magnitude')
# plt.tight_layout()
# plt.show()
#
# mse_train = mean_squared_error(y_train_std[1:], y_model_hat)
# print(f"15-e. EX5-a. MSE of the SARIMA model (Training: Prediction): {mse_train:.4f}")
# y_pred_test = model_fit.predict(start=len(y_train_std), end=len(y_train_std)+len(y_test_std)-1)
# mse_test = mean_squared_error(y_test_std, y_pred_test)
# print(f"15-e. EX5-b. MSE of the SARIMA model (Test: Forecasting): {mse_test:.4f}")
#
# # 15-e. ex3.
# model = sm.tsa.SARIMAX(y_train_std, order=(4,0,0), seasonal_order=(0,0,0,0))
# model_fit = model.fit()
# y_model_hat = model_fit.predict(start=1, end=len(y_train_std)-1)
# plt.figure(figsize=(10, 6))
# plt.plot(y_train_std[1:], label='y')
# plt.plot(y_model_hat, label='1-step')
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.title('15-e. EX3. y and 1-Step Ahead Prediction with SARIMA')
# plt.legend()
# plt.show()
# print(model_fit.summary())
#
# y_pred_train = y_model_hat.to_numpy()
# residuals = y_train_std1[1:] - y_pred_train
# acf_list = []
# lag = 60
# for i in range(lag+1):
#     acf_list.append(acf_fuc(residuals, i))
# new_acf_list = acf_list[::-1] + acf_list[1:]
# x = np.arange(-lag, lag + 1)
# y = np.array(new_acf_list)
# plt.stem(x, y, markerfmt='ro', basefmt='b')
# conf_int = 0.05
# upper_CI = 1.96 / math.sqrt(len(residuals))
# lower_CI = -1.96 / math.sqrt(len(residuals))
# plt.fill_between(x, lower_CI, upper_CI, color='blue', alpha=0.2)
# plt.title('15-e. EX3. ACF of Prediction Errors')
# plt.xlabel('Lags')
# plt.ylabel('Magnitude')
# plt.tight_layout()
# plt.show()
#
# mse_train = mean_squared_error(y_train_std[1:], y_model_hat)
# print(f"15-e. EX5-a. MSE of the SARIMA model (Training: Prediction): {mse_train:.4f}")
# y_pred_test = model_fit.predict(start=len(y_train_std), end=len(y_train_std)+len(y_test_std)-1)
# mse_test = mean_squared_error(y_test_std, y_pred_test)
# print(f"15-e. EX5-b. MSE of the SARIMA model (Test: Forecasting): {mse_test:.4f}")
#
# 15-e. ex4.
model = sm.tsa.SARIMAX(y_train_std, order=(4,0,4), seasonal_order=(0,0,0,0))
model_fit = model.fit()
y_model_hat = model_fit.predict(start=1, end=len(y_train_std)-1)
plt.figure(figsize=(10, 6))
plt.plot(y_train_std[1:], label='y')
plt.plot(y_model_hat, label='1-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('15-e. EX4. y and 1-Step Ahead Prediction with SARIMA')
plt.legend()
plt.show()
print(model_fit.summary())

y_pred_train = y_model_hat.to_numpy()
residuals = y_train_std1[1:] - y_pred_train
acf_list = []
lag = 60
for i in range(lag+1):
    acf_list.append(acf_fuc(residuals, i))
new_acf_list = acf_list[::-1] + acf_list[1:]
x = np.arange(-lag, lag + 1)
y = np.array(new_acf_list)
plt.stem(x, y, markerfmt='ro', basefmt='b')
conf_int = 0.05
upper_CI = 1.96 / math.sqrt(len(residuals))
lower_CI = -1.96 / math.sqrt(len(residuals))
plt.fill_between(x, lower_CI, upper_CI, color='blue', alpha=0.2)
plt.title('15-e. EX4. ACF of Prediction Errors')
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

mse_train = mean_squared_error(y_train_std[1:], y_model_hat)
print(f"15-e. EX4-a. MSE of the SARIMA model (Training: Prediction): {mse_train:.4f}")
y_pred_test = model_fit.predict(start=len(y_train_std), end=len(y_train_std)+len(y_test_std)-1)
mse_test = mean_squared_error(y_test_std, y_pred_test)
print(f"15-e. EX4-b. MSE of the SARIMA model (Test: Forecasting): {mse_test:.4f}")

# 15-e. ex5.
model = sm.tsa.SARIMAX(y_train_std, order=(4,0,4), seasonal_order=(1,0,0,28))
model_fit = model.fit()
y_model_hat = model_fit.predict(start=1, end=len(y_train_std)-1)
plt.figure(figsize=(10, 6))
plt.plot(y_train_std[1:], label='y')
plt.plot(y_model_hat, label='1-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('15-e. EX5. y and 1-Step Ahead Prediction with SARIMA (periodicity=28)')
plt.legend()
plt.show()
print(model_fit.summary())

y_pred_train = y_model_hat.to_numpy()
residuals = y_train_std1[1:] - y_pred_train
acf_list = []
lag = 60
for i in range(lag+1):
    acf_list.append(acf_fuc(residuals, i))
new_acf_list = acf_list[::-1] + acf_list[1:]
x = np.arange(-lag, lag + 1)
y = np.array(new_acf_list)
plt.stem(x, y, markerfmt='ro', basefmt='b')
conf_int = 0.05
upper_CI = 1.96 / math.sqrt(len(residuals))
lower_CI = -1.96 / math.sqrt(len(residuals))
plt.fill_between(x, lower_CI, upper_CI, color='blue', alpha=0.2)
plt.title('15-e. EX5. ACF of Prediction Errors')
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

mse_train = mean_squared_error(y_train_std[1:], y_model_hat)
print(f"15-e. EX5-a. MSE of the SARIMA model (Training: Prediction): {mse_train:.4f}")
y_pred_test = model_fit.predict(start=len(y_train_std), end=len(y_train_std)+len(y_test_std)-1)
mse_test = mean_squared_error(y_test_std, y_pred_test)
print(f"15-e. EX5-b. MSE of the SARIMA model (Test: Forecasting): {mse_test:.4f}")
#
# # 15-e. ex6.
# model = sm.tsa.SARIMAX(y_train_std, order=(4,0,4), seasonal_order=(1,1,1,28))
# model_fit = model.fit()
# y_model_hat = model_fit.predict(start=1, end=len(y_train_std)-1)
# plt.figure(figsize=(10, 6))
# plt.plot(y_train_std[1:], label='y')
# plt.plot(y_model_hat, label='1-step')
# plt.xlabel('Sample')
# plt.ylabel('Value')
# plt.title('15-e. EX6. y and 1-Step Ahead Prediction with SARIMA (periodicity=28)')
# plt.legend()
# plt.show()
# print(model_fit.summary())
#
# y_pred_train = y_model_hat.to_numpy()
# residuals = y_train_std1[1:] - y_pred_train
# acf_list = []
# lag = 60
# for i in range(lag+1):
#     acf_list.append(acf_fuc(residuals, i))
# new_acf_list = acf_list[::-1] + acf_list[1:]
# x = np.arange(-lag, lag + 1)
# y = np.array(new_acf_list)
# plt.stem(x, y, markerfmt='ro', basefmt='b')
# conf_int = 0.05
# upper_CI = 1.96 / math.sqrt(len(residuals))
# lower_CI = -1.96 / math.sqrt(len(residuals))
# plt.fill_between(x, lower_CI, upper_CI, color='blue', alpha=0.2)
# plt.title('15-e. EX6. ACF of Prediction Errors')
# plt.xlabel('Lags')
# plt.ylabel('Magnitude')
# plt.tight_layout()
# plt.show()
#
# mse_train = mean_squared_error(y_train_std[1:], y_model_hat)
# print(f"15-e. EX6-a. MSE of the SARIMA model (Training: Prediction): {mse_train:.4f}")
# y_pred_test = model_fit.predict(start=len(y_train_std), end=len(y_train_std)+len(y_test_std)-1)
# mse_test = mean_squared_error(y_test_std, y_pred_test)
# print(f"15-e. EX6-b. MSE of the SARIMA model (Test: Forecasting): {mse_test:.4f}")

# 17. Final Model selection
# 18. Forecast function & 19. h-step ahead Predictions
model = sm.tsa.SARIMAX(y_train_std, order=(4,0,4), seasonal_order=(1,0,0,28))
model_fit = model.fit()
y_pred_train = model_fit.predict(start=1, end=len(y_train_std)-1)
plt.figure(figsize=(10, 6))
plt.plot(y_train_std[1:], color='red', label='y train')
plt.plot(y_pred_train, color='green', label='1-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('19. y and 1-Step Ahead Prediction with the Final Model')
plt.legend()
plt.show()
print(model_fit.summary())

y_pred_test = model_fit.predict(start=1, end=len(y_test_std)-1)
plt.figure(figsize=(10, 6))
plt.plot(y_test_std[1:], color='blue', label='y test')
plt.plot(y_pred_test.values.flatten(), color='green', label='h-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('19. y test and h-Step Ahead Prediction with the Final Model')
plt.legend()
plt.show()
print(model_fit.summary())

plt.figure(figsize=(18, 6))
plt.plot(y_train_std, color='red', label='Train')
plt.plot(np.arange(len(y_train_std), len(y_train_std)+len(y_test_std[1:])),y_test_std[1:], color='blue', label='Test')
plt.plot(np.arange(len(y_train_std), len(y_train_std)+len(y_test_std[1:])),y_pred_test.values.flatten(), color='green', label='H-step')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('19. h-Step Ahead Prediction with the Final Model')
plt.legend()
plt.show()
print(model_fit.summary())

# 16. Deep Learning Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

X_train_std = X_train_std.values.reshape(X_train_std.shape[0], X_train_std.shape[1], 1)
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_std.shape[1], 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train_std, y_train_std, epochs=100, batch_size=32, verbose=0)

# define function to make h-step ahead predictions
def lstm_forecast(model, X_test_std, h):
    # create empty array to hold forecasted values
    forecast = []
    # initial input to the model is the last sequence in the training data
    last_sequence = X_test_std[0]
    for i in range(h):
        # make one-step prediction
        y_pred = model.predict(last_sequence.reshape(1, X_test_std.shape[1], 1))
        # append predicted value to forecast array
        forecast.append(y_pred[0,0])
        # update last sequence with predicted value
        last_sequence = np.append(last_sequence[1:], y_pred)
    return forecast

# h-step predictions on test set
h = len(y_test_std)
X_test_std = X_test_std.values.reshape(X_test_std.shape[0], X_test_std.shape[1], 1)
forecast = lstm_forecast(model, X_test_std, h)
mse_test = mean_squared_error(y_test_std, forecast)
print("16. Mean squared error (MSE) of LSTM (Test: Forecasting): {:.4f}".format(mse_test))


