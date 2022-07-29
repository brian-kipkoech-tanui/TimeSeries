# BasicTimeSeries
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
Below is a limited example of how to use the TimeSeries class:
```bash
timeseries = TimeSeries()
timeseries.visualize(data)
timeseries.decomposition_plot(data, model='additive', period='None') # The period is None and the model is additive by default. The period should be adjusted according to data.


