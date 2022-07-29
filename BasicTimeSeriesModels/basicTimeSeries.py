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
    def visualizations(self, data):
        plt.style.use('dark_background')
        """Plots all time series plots for all the variables
        """
        try:
            # If the time series data is a pandas DataFrame, it should visualize all
            for i in range(len(list(data.columns))):
                df = list(data.columns)[i]
                data[df].plot(figsize=(12, 5), color = "white") # Plotting the time series plots
                plt.title(df, size=14)
                plt.show()
        except AttributeError:
            # If the Time series data is a pandas Series
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
            
    
