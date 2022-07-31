# General
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import warnings

# stats
# from statsmodels.tsa.seasonal import seasonal_decompose
# import statsmodels.graphics.tsaplots as sgt
# from statsmodels.tsa.arima_model import ARIMA
# from scipy.stats.distributions import chi2
# import statsmodels.tsa.stattools as sts
# import statsmodels.api as sm
# from sklearn.metrics import mean_squared_error

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Settings
# Ignore harmless warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (15, 4)

class TimeSeries:
    def trend_visualizations(self, data):
        plt.style.use('dark_background')
        """Plots all time series plots for all the variables
        """
        try:
            # If the time series data is a pandas DataFrame, it should visualize all variables
            for i in range(len(list(data.columns))):
                df = list(data.columns)[i]
                data[df].plot(figsize=(12, 5), color = "white") # Plotting the time series plots
                plt.title(df, size=14)
                plt.show()
        except AttributeError:
            # If the Time series data is a pandas Series, It should not get an AttributeError
            data.plot(figsize=(20, 5), color="white")  # Plotting the time series plots
            plt.title("Time Plot", size=10)
            plt.show()
            
    def decomposition_plot(self, data, model="additive", period=None):
        plt.style.use('dark_background')
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        try:
            # If the time series data is a pandas DataFrame, it should visualize all
            # Plotting the decomposition plots; trend, seasonality and randomness
            for i in range(len(list(data.columns))):
                df = list(data.columns)[i]
                decomposition = seasonal_decompose(data[df],
                                            model=model,
                                            period=period)
                
                # Seasonality plot
                seasonality = decomposition.seasonal
                df = list(data.columns)[i]
                seasonality.plot(color='white')
                plt.title(f'seasonality of {df}', size=14)
                plt.show()
                
                # Residual Plot
                resid = decomposition.resid
                resid.plot(color='white')
                plt.title(f"Residuals {df}", size=14)
                plt.show()
        except AttributeError:
            # If the Time series data is a pandas Series
            decomposition = seasonal_decompose(data,
                                            model=model,
                                            period=period)
            # Seasonality plot
            seasonality = decomposition.seasonal
            seasonality.plot(color='white')
            plt.title('seasonality', size=14)
            plt.show()
            
            # Residual Plot
            resid = decomposition.resid
            resid.plot(color='white')
            plt.title("Residuals", size=14)
            plt.show()
            
    def correlation_function(self, data, lags=10):
        import statsmodels.graphics.tsaplots as sgt
        try:
            # If the time series data is a pandas DataFrame, it should visualize all
            # Plotting the decomposition plots; trend, seasonality and randomness
            for i in range(len(list(data.columns))):
                df = list(data.columns)[i]
                # The ACF
                sgt.plot_acf(data[df], lags=lags,
                            #zero=False,
                            color="darkslategrey",
                            title=f"ACF of {df}")
                plt.show()
                # The PACF
                sgt.plot_pacf(data[df], alpha=0.05,
                            lags=lags, #zero=False,
                            method=("ols"),
                            color="darkslategrey",
                            title=f"PACF of {df}")
                plt.show()
        except AttributeError:
            # The ACF
            sgt.plot_acf(data, lags=lags,
                        #zero=False,
                        color="darkslategrey",
                        title="ACF")
            plt.show()
            # The PACF
            sgt.plot_pacf(data, alpha=0.05,
                        lags=lags, #zero=False,
                        method=("ols"),
                        color="darkslategrey",
                        title="PACF")
            plt.show()
        
    def stationarity_check(self, data, signif=0.05):
        
        """
        Perform ADFuller to test for Stationarity of given series and print report. 
        Data Must be  a pandas DataFrame
        param data: pandas DataFrame or pandas Series
        param signif: The significance level(e.g 0.1, 0.05)
        """
        from statsmodels.tsa.stattools import adfuller, kpss
        try:
            for name, column in data.iteritems():
                r = adfuller(column, autolag='AIC')
                output = {'test_statistic':round(r[0], 4), 
                            'pvalue':round(r[1], 4), 
                            'n_lags':round(r[2], 4), 
                            'n_obs':r[3]}
                p_value = output['pvalue'] 
                def adjust(val, length= 6): return str(val).ljust(length)

                # Print Summary
                print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
                print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
                print(f' Significance Level    = {signif}')
                print(f' Test Statistic        = {output["test_statistic"]}')
                print(f' No. Lags Chosen       = {output["n_lags"]}')

                for key,val in r[4].items():
                    print(f' Critical value {adjust(key)} = {round(val, 3)}')

                if p_value <= signif:
                    print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
                    print(f" => Series is Stationary.")
                else:
                    print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
                    print(f" => Series is Non-Stationary.")
                    print('\n')
            
        except ValueError:
        
            p_value = adfuller(data)[1]
            test_statistic = adfuller(data)[0]
            lag = adfuller(data)[2]
            observations = adfuller(data)[3]
            critical_values = adfuller(data)[4]
            maximized_criteria = adfuller(data)[5]

            print("Augmented Dickey Fuller Test", "\n")
            print("null-hypothesis: The data is non-stationary" )
            print("alternative-hypothesis: The data is stationary","\n", '-'*47)
            print(f"p-value:                = {p_value}")
            print(f"test-statistic:         = {test_statistic}")
            print(f"Number of lag chosen:   = {lag}")
            print(f"observation:            = {observations}")
            print(f"critical-values:        = {critical_values}")
            print(f"maximized-criteria:     = {maximized_criteria}")
            if test_statistic < critical_values['5%']:
                print(f"=> The data is stationary. The p-value is {round(p_value, 3)} thus we reject the null hypothesis")
            else:
                print(f"=> The data is non-stationary.The p-value is {round(p_value, 3)} thus we fail to reject the null hypothesis")
                
            print('\n')
            
            # KPSS test
            print('-'*47, "\n",'-'*47,"\n")
            print("KPSS Test for Stationarity","\n")
            print("H0: The time series data is stationary")
            print("H1:The time series data is non-stationary", "\n",'-'*47)
            statistic, p_value, n_lags, critical_values = kpss(data)
            # Format Output
            print(f'KPSS Statistic: ={statistic}')
            print(f'p-value:        ={p_value}')
            print(f'num lags:       ={n_lags}')
            print('Critial Values:')
            for key, value in critical_values.items():
                print(f'   {key} :  ={value}')
            print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')
            
    def split_data(self, data, ratio=0.90):
        """_summary_
        A method to split data
        Args:
            data (pandas series): The data to split into Train and Test sets
            ratio (float, optional): The ratio used to split the data. Defaults to 0.90.

        Returns:
            train, test: The train and the test datasets
        """
        self.__ratio = ratio
        train_size = int(len(data) * self.__ratio)
        self.__train, self.__test = data[0:train_size], data[train_size:]
        return self.__train, self.__test 
            
    def model_evaluation(self, **params):
        raise NotImplementedError
    
    def best_model(self, **params):
        """_summary_
        Uses statsmodels to compute the results of the best fitted model

        Raises:
            NotImplementedError: If there is no code implementation in the child class
        """
        raise NotImplementedError
    
    def results_diagnostics(self, **params):
        """_summary_
        Perform diagnostics to check if the model fulfills the time series assumptions

        Raises:
            NotImplementedError: If there is no code implementation in the child class
        """
        raise NotImplementedError
    
    def prediction_check(self, **params):
        """_summary_

        Args:
            test_data(pandas series or DataFrame): The data to benchmark the predictions from the model
            start_date (datetime string): The start_date for the prediction, should coincide with test data
            end_date (datetime string): The start_date for the prediction, should coincide with test data
            

        Raises:
            NotImplementedError: If no code is provided in the child class
        """
        raise NotImplementedError
    
    def forecasts(self, steps):
        """_summary_
        Make Predictions from the fitted model.

        Args:
            steps (integer): The steps into the future to be forecasted.

        Raises:
            NotImplementedError: If there is no code implementation in the child class
        """
        raise NotImplementedError
    
class ARIMAModels(TimeSeries):
    def model_evaluation(self, series, p_values, d_values, q_values):
        """_summary_
        Uses grid_search to locate the best fitting model

        Args:
            series (Pandas Series): The data to be modelled on the time series. Should be a Pandas Series
            p_values (list): A list containing the order for the AR part of ARIMA
            d_values (list): A list containing the order for the differentiation part of ARIMA
            q_values (list): A list containing the order of MA part of ARIMA
        Examples:
            1) model_evaluation(data["column_name"], 
                                p_values=[1,2,3,4,5], 
                                d_values=[1,2],
                                q_values=[1,2,3,4,5])
        """
        import statsmodels.api as sm
        from sklearn.metrics import mean_squared_error
        dataset = series.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        def evaluate_arima_model(dataset, order):
                            # prepare training dataset
                            train_size = int(len(dataset) * self.__ratio)
                            train, test = dataset[0:train_size], dataset[train_size:]
                            history = [dataset for dataset in train]
                            # make predictions
                            predictions = list()
                            for t in range(len(test)):
                                model = sm.tsa.arima.ARIMA(history, order=order)
                                model_fit = model.fit()
                                yhat = model_fit.forecast()[0]
                                predictions.append(yhat)
                                history.append(test[t])
                            # calculate out of sample error
                            error = mean_squared_error(test, predictions)
                            return error
                        mse=evaluate_arima_model(dataset, order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.6f' % (order, mse))
                    except:
                        continue
        print('Best ARIMA%s MSE=%.6f' % (best_cfg, best_score))
        
    def best_model(self, train_data, order):
        """_summary_
        This is a function to fit the best model based on aic and llf obtained from
        run_time_series function
        Args:
        series(pandas series): A pandas series containing the data to be fitted on the model
        order (tuple): A tuple containing the order for the ARIMA model.
        Example:
        ARIMAModel.best_model(data["column_name"], order=(2,1,1))
        """
        warnings.filterwarnings("ignore")
        import statsmodels.api as sm
        self.__train = train_data
        self.__order = order
        model = sm.tsa.arima.ARIMA(self.__train, order=self.__order)
        self.__results = model.fit()
        print(self.__results.summary().tables[1])
        return self.__results.resid
    
    
    def results_diagnostics(self, lags):
        """_summary_
        Check if the model fulfils regression and Time Series Assumptions
        By Plotting the QQ-plot and performing the Ljung-Box test on the data
        Args:
        lags (integer): The number of lags to be used in the Ljung-Box test
        """
        import statsmodels.api as sm
        self.__results.plot_diagnostics(figsize=(16, 8))
        plt.show()
        print('\n')
        print('-'*47, "\n",'-'*47,"\n")
        print("Ljung-Box Test - Checks for Serial Correlation", "\n")
        print("null-hypothesis:         The residuals are independently distributed.")
        print("alternative-hypothesis:  The residuals are not independently distributed","\n", '-'*47)
        print(sm.stats.acorr_ljungbox(self.__results.resid, lags=lags, return_df=True))
        
    def prediction_check(self, test_data, start_date, end_date):
        """_summary_
        Plot How the predictions compares with the test data

        Args:
        test_data(pandas Series): The pandas series to be compared with the prediction
        start_date (datetime string): The date that the model should start predicting, should be Y-m-d
        end_date (datetime string): When the model should end its prediction, should be Y-m-d
        """
        self.__test = test_data
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        df_pred = self.__results.predict(start=start_date, end=end_date)
        # Visualizing the predictions
        # df_pred[self.start_date:self.end_date].plot(figsize=(20, 5), color="red")
        # self.data_test[self.start_date:self.end_date].plot(figsize=(20, 5), color="blue")
        plt.rcParams["figure.figsize"] = (20, 6)
        plt.plot(df_pred, label="prediction", color="red")
        plt.plot(self.__test[start_date:end_date], label="actual", color="midnightblue")
        plt.xlabel("Date")
        plt.title("Predictions VS Actual", size=24)
        plt.legend(loc="upper right")
        plt.show()
    
    def forecasts(self,data, steps):
        """_summary_
        Gets the forecasts of the model

        Args:
            data (_type_): _description_
            steps (_type_): _description_
        """
        import statsmodels.api as sm
        model = sm.tsa.arima.ARIMA(data, order=self.__order)
        results = model.fit()
        pred_uc = results.get_forecast(steps=steps)
        pred_ci = pred_uc.conf_int()
        ax = data.plot(label='observed', figsize=(14, 7), color="white")
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color="red")
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(data.name)
        plt.title(data.name)
        plt.legend()
        plt.show()
        return pred_uc.summary_frame()
    
class SARIMAModels(TimeSeries):
    def model_evaluation(self, train_data, order_limit, season=12):
        """_summary_
        This is a function to make the SARIMA process easy and fast
        to come up with the optimum parameters to fit the best model

        Args:
        train_data(pandas series): subset of the dataset after splitting into train and test
        order_limit (Integer): The upper bound of the orders for the seasonal and non-seasonal parts
        season(integer, optional): Should be changed according to frequency of the data, i.e
                                   if weekly, it should be set accordingly.
        """
        import statsmodels.api as sm
        print("Choosing the best model from the range provided", "\n", '-'*47)
        p = d = q = range(0, order_limit)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], season) for x in list(itertools.product(p, d, q))]
        best_score, best_order, best_seas_order = float("inf"), None, None

        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train_data,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)
                    results = mod.fit()
                    aic = results.aic
                    if aic < best_score:
                        best_score, best_order, best_seas_order = results.aic, param, param_seasonal

                    print(f'ARIMA{param}x{param_seasonal}season:{season} - AIC:{results.aic} - LLF:{results.llf}')
                except:
                    continue
        print('Best Order%s Seasonal_order%s AIC=%.3f' % (best_order, best_seas_order, best_score))
        
    def best_model(self, train_data, order, seasonal_order):
        """
        This is a function to fit the best model based on aic and llf obtained from
        run_time_series function
        :param order: This is the ARIMA order that best suits the data. Entered as a tuple
        :param seasonal_order: The order of the seasonal part of the data. Entered as tuple
        :param enforce_stationarity: To be described
        :param enforce_invertibility:To be described
        :return: results, results_resid
        """
        self.__order=order
        self.__seasonal_order=seasonal_order
        import statsmodels.api as sm
        model = sm.tsa.statespace.SARIMAX(train_data,
                                          order=order,
                                          seasonal_order=seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)

        self.__results = model.fit()
        print(self.__results.summary().tables[1])
        self.__results_resid = self.__results.resid
        return self.__results_resid
    
    def results_diagnostics(self, lags):
        """_summary_
        Check if the model fulfils regression and Time Series Assumptions
        By Plotting the QQ-plot and performing the Ljung-Box test on the data
        Args:
        lags (integer): The number of lags to be used in the Ljung-Box test
        """
        import statsmodels.api as sm
        self.__results.plot_diagnostics(figsize=(16, 8))
        plt.show()
        print('\n')
        print('-'*47, "\n",'-'*47,"\n")
        print("Ljung-Box Test - Checks for Serial Correlation", "\n")
        print("null-hypothesis:         The residuals are independently distributed.")
        print("alternative-hypothesis:  The residuals are not independently distributed","\n", '-'*47)
        print(sm.stats.acorr_ljungbox(self.__results_resid, lags=lags, return_df=True))
        
    def prediction_check(self, test_data, start_date, end_date):
        """_summary_
        Plot How the predictions compares with the test data

        Args:
        test_data(pandas Series): The pandas series to be compared with the prediction
        start_date (datetime string): The date that the model should start predicting, should be Y-m-d
        end_date (datetime string): When the model should end its prediction, should be Y-m-d
        """
        self.__test = test_data
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        df_pred = self.__results.predict(start=start_date, end=end_date)
        # Visualizing the predictions
        # df_pred[self.start_date:self.end_date].plot(figsize=(20, 5), color="red")
        # self.data_test[self.start_date:self.end_date].plot(figsize=(20, 5), color="blue")
        plt.rcParams["figure.figsize"] = (20, 6)
        plt.plot(df_pred, label="prediction", color="red")
        plt.plot(self.__test[start_date:end_date], label="actual", color="midnightblue")
        plt.xlabel("Date")
        plt.title("Predictions VS Actual", size=24)
        plt.legend(loc="upper right")
        plt.show()
        
    def forecasts(self,data, steps):
        """_summary_
        Gets the forecasts of the model

        Args:
            data (_type_): _description_
            steps (_type_): _description_
        """
        import statsmodels.api as sm
        model = sm.tsa.statespace.SARIMAX(data,
                                          order=self.__order,
                                          seasonal_order=self.__seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)
        results = model.fit()
        pred_uc = results.get_forecast(steps=steps)
        pred_ci = pred_uc.conf_int()
        ax = data.plot(label='observed', figsize=(14, 7), color="white")
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color="red")
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel(data.name)
        plt.title(data.name)
        plt.legend()
        plt.show()
        return pred_uc.summary_frame()
        
    
    
    
        
        
        
        
            
    