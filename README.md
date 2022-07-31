# BasicTimeSeries Automation
This part has three classes: TimeSeries, ARIMA and SARIMA classes.
TimeSeries is the parent class that takes care of common time series tasks such as:
1) Time Series Plots
2) Decomposition plots
3) Stationarity Checks
4) acf and pacf
For best results, rigorously clean your data and make sure there are no missing dates
Note: The data should already have its index set to the time it represents, e.g using 
```bash
pd.set_index('date_time_column', inplace = True)
```
And also make sure that the data's datetime variable is in datetime format e.g by doing this:
```bash
data["date_column"] = pd.to_datetime(data["date"])
# Or if its yearly data
data["year"] = pd.to_datetime(data["year"], format="%Y")
```
ARIMAModels is a class that inherits from the TimeSeries class, with some specific functionalities that include fitting an ARIMA model, checking the performance of the ARIMA model and forecasting using the ARIMA model thats fitted.
Below is a limited sample of how to use the ARIMAModels class:
```bash
arima_model = ARIMAModels()
arima_model.trend_visualizations(data)
arima_model.decomposition_plot(data, model='additive', period='None') # The period is None and the model is additive by default. The period should be adjusted according to data. The data can be a pandas DataFrame or a pandas Series.
arima_model.correlation_function(data, lags=10) # lags=10 is the default, and can be changed if need be.
arima_model.stationarity_check(data, signif=0.05) # signif=0.05 is the alpha value, 
# can be changed if need be. In this  case stationarity is being checked at 95% confidence level.
arima_model.model_evaluation(train, test, p_values, d_values, q_values) 
# train parameter should the a Pandas Series, which is the training dataset; 
# test parameter should the a Pandas Series, which is the training dataset; 
# p_values is a list of the range of values that you wish to check for the AR part of ARIMA model
# d_values is a list of the range of values that you wish to check for the differentiation part
# q_values is a list of the range of values that you wish to check for the MA part of ARIMA model
arima_model.best_model(train_data, order) # train_data represents the data to be modelled and should have already been split; order parameter is the order of the ARIMA model.
arima_model.results_diagnostics(lags) # The lags will be used to perform the ljung-Box test
# The residuals and fitted model are already provided by the class so you dont need to reprovision
arima_model.prediction_check(test_data=test)
# The test dataset must be provided
# Checks how the model's prediction performs compared to the test data.
# Note: arima_model.prediction_check method must only be called after the arima_model.best_model
arima_model.forecasts(data, steps) # data should be the full dataset(neither train nor test)
# Note: arima_model.forecasts method must only be called after the arima_model.best_model
```
SARIMAModels is a class that inherits from the TimeSeries class, with some specific functionalities that include fitting a SARIMA model, checking the performance of the SARIMA model and forecasting using the SARIMA model thats fitted.
Below is a limited sample of how to use the SARIMAModels class:
```bash
sarima_model = sARIMAModels()
sarima_model.trend_visualizations(data)
sarima_model.decomposition_plot(data, model='additive', period='None') # The period is None and the model is additive by default. The period should be adjusted according to data. The data can be a pandas DataFrame or a pandas Series.
sarima_model.correlation_function(data, lags=10) # lags=10 is the default, and can be changed if need be.
sarima_model.stationarity_check(data, signif=0.05) # signif=0.05 is the alpha value, 
# can be changed if need be. In this  case stationarity is being checked at 95% confidence level.
sarima_model.model_evaluation(series, p_values, d_values, q_values) # series parameter should the a Pandas Series; 
# p_values is a list of the range of values that you wish to check for the AR part of ARIMA model
# d_values is a list of the range of values that you wish to check for the differentiation part
# q_values is a list of the range of values that you wish to check for the MA part of ARIMA model
sarima_model.best_model(train_data, order) # train_data represents the data to be modelled and should have already been split; order parameter is the order of the ARIMA model.
sarima_model.results_diagnostics(lags) # The lags will be used to perform the ljung-Box test
# The residuals and fitted model are already provided by the class so you dont need to reprovision
sarima_model.prediction_check(test_data=test)
# The test dataset must be provided
# Checks how the model's prediction performs compared to the test data.
# Note: sarima_model.prediction_check method must only be called after the sarima_model.best_model
sarima_model.forecasts(data, steps) # data should be the full dataset(neither train nor test)
# Note: sarima_model.forecasts method must only be called after the sarima_model.best_model