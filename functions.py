import numpy as np
import pandas as pd

# definitions come from various website as :
# Binance
# coinacademy.fr
# tradingview.com
# crypto.com
# zoneboourse.com
# coinbase.com

# this list is nit exhaustive


def calculate_all_indicators_optimised(dataset) :
    """
    function that allows to program a set of different functions corresponding to each indicators
    
    Parameters:
        dataset (dataframe): original dataset composed of columnss High, Low, Open, Close and Volume)

    Returns:
        return a dataframe with all the indicators calculated and added per new columns
    """
    #MOMENTUM
    dataset = ta_williams_percent_r(dataset,14)
    dataset = ta_roc(dataset,14)
    dataset = ta_rsi(dataset,7)
    dataset = ta_rsi(dataset,14)
    dataset = ta_rsi(dataset,28)
    dataset = ta_macd(dataset, 8, 21)
    dataset = ta_bbands(dataset,20)
    
    #TREND
    dataset = ta_ichimoku_cloud(dataset)
    dataset = ta_ema(dataset, 3)
    dataset = ta_ema(dataset, 8)
    dataset = ta_ema(dataset, 15)
    dataset = ta_ema(dataset, 50)
    dataset = ta_ema(dataset, 100)
    dataset = ta_adx(dataset, 14)
    dataset = ta_donchian(dataset, 10)
    dataset = ta_donchian(dataset, 20)
    dataset = ta_alma(dataset, 10)
    dataset = ta_tsi(dataset, 13, 25)
    dataset = ta_zscore(dataset, 20)
    dataset = ta_log_return(dataset, 10)
    dataset = ta_log_return(dataset, 20)
    dataset = ta_vortex(dataset, 7)
    dataset = ta_aroon(dataset, 16)
    dataset = ta_ebsw(dataset, 14)
    dataset = ta_accbands(dataset, 20)
    dataset = ta_short_run(dataset, 14)
    dataset = ta_bias(dataset, 26)
    dataset = ta_ttm_trend(dataset, 5, 20)
    dataset = ta_percent_return(dataset, 10)
    dataset = ta_percent_return(dataset, 20)
    dataset = ta_kurtosis(dataset, 5)
    dataset = ta_kurtosis(dataset, 10)
    dataset = ta_kurtosis(dataset, 20)
    dataset = ta_eri(dataset, 13)
    
    #VOLATILITY
    dataset = ta_atr(dataset, 14)
    dataset = ta_keltner_channels(dataset, 20)
    dataset = ta_chaikin_volatility(dataset, 10)
    dataset = ta_stdev(dataset, 5)
    dataset = ta_stdev(dataset, 10)
    dataset = ta_stdev(dataset, 20)
    dataset = ta_vix(dataset, 21)
    
    # VOLUME
    dataset = ta_obv(dataset, 10)
    dataset = ta_chaikin_money_flow(dataset, 5)
    dataset = ta_volume_price_trend(dataset, 7)
    dataset = ta_accumulation_distribution_line(dataset, 3)
    dataset = ta_ease_of_movement(dataset, 14)
    
    return dataset

#-----------------------------------------------------------------------------------------------------------------------------------

def calculate_all_indicators(dataset) :
    """
    function that allows to program a set of different functions corresponding to each indicators
    
    Parameters:
        dataset (dataframe): original dataset composed of columnss High, Low, Open, Close and Volume)

    Returns:
        return a dataframe with all the indicators calculated and added per new columns
    """
    #MOMENTUM : 8
    dataset = ta_williams_percent_r(dataset,14)
    dataset = ta_roc(dataset,14)
    dataset = ta_rsi(dataset,7)
    dataset = ta_rsi(dataset,14)
    dataset = ta_rsi(dataset,28)
    dataset = ta_stochastic(dataset, 3, 3)
    dataset = ta_macd(dataset, 8, 21)
    dataset = ta_bbands(dataset,20)
    
    #TREND 33
    dataset = ta_ichimoku_cloud(dataset)
    dataset = ta_ema(dataset, 3)
    dataset = ta_ema(dataset, 8)
    dataset = ta_ema(dataset, 15)
    dataset = ta_ema(dataset, 50)
    dataset = ta_ema(dataset, 100)
    dataset = ta_adx(dataset, 14)
    dataset = ta_donchian(dataset, 10)
    dataset = ta_donchian(dataset, 20)
    dataset = ta_eri(dataset, 13)
    dataset = ta_alma(dataset, 10)
    dataset = ta_tsi(dataset, 13, 25)
    dataset = ta_zscore(dataset, 20)
    dataset = ta_log_return(dataset, 5)
    dataset = ta_log_return(dataset, 10)
    dataset = ta_log_return(dataset, 20)
    dataset = ta_kurtosis(dataset, 5)
    dataset = ta_kurtosis(dataset, 10)
    dataset = ta_kurtosis(dataset, 20)
    dataset = ta_vortex(dataset, 7)
    dataset = ta_aroon(dataset, 16)
    dataset = ta_ebsw(dataset, 14)
    dataset = ta_accbands(dataset, 20)
    dataset = ta_short_run(dataset, 14)
    dataset = ta_bias(dataset, 26)
    dataset = ta_ttm_trend(dataset, 5, 20)
    dataset = ta_percent_return(dataset, 1)
    dataset = ta_percent_return(dataset, 5)
    dataset = ta_percent_return(dataset, 10)
    dataset = ta_percent_return(dataset, 20)
    dataset = ta_stdev(dataset, 5)
    dataset = ta_stdev(dataset, 10)
    dataset = ta_stdev(dataset, 20)
    
    #VOLATILITY  :5
    dataset = ta_vix(dataset, 21)
    dataset = ta_chaikin_volatility(dataset, 10)
    dataset = ta_atr(dataset, 14)
    dataset = ta_chaikin_oscillator(dataset, 3)
    dataset = ta_keltner_channels(dataset, 20)
    
    # VOLUME : 6
    dataset = ta_obv(dataset, 10)
    dataset = ta_chaikin_money_flow(dataset, 5)
    dataset = ta_volume_price_trend(dataset, 7)
    dataset = ta_accumulation_distribution_line(dataset, 3)
    dataset = ta_money_flow_index(dataset, 14)
    dataset = ta_ease_of_movement(dataset, 14)
    
    return dataset

#---------------------------------------------------------------------------------------------------------------------------

def ta_cci(dataset, window=20, constant=0.015):
    typical_price = (dataset["High"] + dataset["Low"] + dataset["Close"]) / 3
    mean_deviation = abs(typical_price - typical_price.rolling(window=window).mean()).rolling(window=window).mean()
    dataset["CCI{}".format(window)] = (typical_price - typical_price.rolling(window=window).mean()) / (constant * mean_deviation)
    return dataset

def ta_williams_percent_r(dataset, window=14):
    """
    Calculate Williams %R for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price information
    - window (int): The window size for calculating the highest high and lowest low. Default is 14
    
    Returns:
    - dataset with williams %R values as a new column
    """
    # Calculate the highest high and lowest low over the specified window
    highest_high = dataset["High"].rolling(window=window).max()
    lowest_low = dataset["Low"].rolling(window=window).min()
    # Calculate Williams %R and add it as a new column
    dataset["Williams_%R{}".format(window)] = -((highest_high - dataset["Close"]) / (highest_high - lowest_low)) * 100
    return dataset

def ta_roc(dataset, window=14):
    """
    Calculate Roc
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Rolling window. Default is 14
    
    Returns:
    - dataset with new column
    """    
    # Calculate Rate of Change
    dataset["ROC_{}".format(window)] = (dataset["Close"] / dataset["Close"].shift(window) - 1) * 100
    return dataset

def ta_rsi(dataset, window=14) : #14,28
    """
    Calculate Relative Strength Index (RSI) in a DataFrame.
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Rolling window for average gain and loss calculations. Default is 14
    
    Returns:
    - dataset with new column
    """
    # Calculate daily price changes
    delta = dataset["Close"].diff(1)
    # Separate gains and losses
    gains = delta.where(delta>0,0)
    losses = -delta.where(delta<0,0)
    # Calculate average gains and losses over the specified window
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss
    # Calculate the RSI
    dataset["rsi_{}".format(window)] = 100 - (100 / (1 + rs))
    return dataset

def ta_stochastic(dataset, k_period=3, d_period=3):
    """
    Calculate Stochastic Oscillator for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - k_period: Stochastic %K period. Default is 3
    - d_period: Stochastic %D period. Default is 3
    
    Returns:
    - dataset with additional columns "Stochastic_K" and "Stochastic_D"
    """
    # Calculate %K
    lowest_low = dataset["Close"].rolling(window=k_period).min()
    highest_high = dataset["Close"].rolling(window=k_period).max()
    stochastic_k = 100 * ((dataset["Close"] - lowest_low) / (highest_high - lowest_low))
    # Calculate %D
    stochastic_d = stochastic_k.rolling(window=d_period).mean()
    # Add "Stochastic_K" and "Stochastic_D" columns to the dataset
    dataset["Stochastic_K{}".format(k_period)] = stochastic_k
    dataset["Stochastic_D{}".format(d_period)] = stochastic_d
    return dataset

def ta_macd(dataset, short_window=8, long_window=21, signal_window=9):
    """
    Calculate MACD for a given column in a DataFrame with specified short, long, and signal windows
    
    Parameters:
    - dataset: Pandas DataFrame
    - short_window: Short-term window for the EMA. Default is 8
    - long_window: Long-term window for the EMA. Default is 21
    - signal_window: Signal line window for the EMA. Default is 9
    
    Returns:
    - dataset with additional columns for MACD, Signal Line, and MACD Histogram
    """
    # Calculate short-term and long-term EMAs
    short_ema = dataset["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = dataset["Close"].ewm(span=long_window, adjust=False).mean()
    # Calculate MACD line
    macd_line = short_ema - long_ema
    # Calculate Signal Line
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    # Calculate MACD Histogram
    macd_histogram = macd_line - signal_line
    # Add new columns to the original DataFrame
    dataset["MACD_Line"] = macd_line
    dataset["Signal_Line"] = signal_line
    dataset["MACD_Histogram"] = macd_histogram
    return dataset

def ta_bbands(dataset, window=20, num_std_dev=2) :
    """
    Calculate Bollinger Bands for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Rolling window for moving average calculation. Default is 20
    - num_std_dev: Number of standard deviations for upper and lower bands. Default is 2
    
    Returns:
    - return a dataset with "Upper Band", "Middle Band" (MA), and "Lower Band" columns
    """
    # Calculate the rolling mean (Middle Band)
    dataset["midlle_band"] = dataset["Close"].rolling(window=window).mean()
    # Calculate the standard deviation
    dataset["std"] = dataset["Close"].rolling(window=window).std()
    # Calculate upper and lower Bollinger Bands
    dataset["upper_band{}".format(window)] = dataset["midlle_band"] + (num_std_dev * dataset["std"])
    dataset["lower_band{}".format(window)] = dataset["midlle_band"] - (num_std_dev * dataset["std"])
    # Drop intermediate columns if needed
    dataset.drop(["std"], axis=1, inplace=True)   
    return dataset

#TREND

def ta_ichimoku_cloud(dataset, window_tenkan=9, window_kijun=26, window_senkou_span_b=52, window_chikou=26):
    """
    Calculate Ichimoku Cloud components for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price information
    - window_tenkan (int): The window size for Tenkan-sen. Default is 9
    - window_kijun (int): The window size for Kijun-sen. Default is 26
    - window_senkou_span_b (int): The window size for Senkou Span B. Default is 52
    - window_chikou (int): The window size for Chikou Span. Default is 26
    
    Returns:
    - dataset with Ichimoku Cloud components as new columns.
    """
    # Calculate Tenkan-sen
    tenkan_sen = (dataset["Close"].rolling(window=window_tenkan).max() + dataset["Close"].rolling(window=window_tenkan).min()) / 2
    # Calculate Kijun-sen
    kijun_sen = (dataset["Close"].rolling(window=window_kijun).max() + dataset["Close"].rolling(window=window_kijun).min()) / 2
    # Calculate Senkou Span A
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window_kijun)
    # Calculate Senkou Span B
    senkou_span_b = (dataset["Close"].rolling(window=window_senkou_span_b).max() + dataset["Close"].rolling(window=window_senkou_span_b).min()) / 2
    # Calculate Chikou Span
    chikou_span = dataset["Close"].shift(-window_chikou)
    # Add Ichimoku Cloud components as new columns
    dataset["Tenkan_sen"] = tenkan_sen
    dataset["Kijun_sen"] = kijun_sen
    dataset["Senkou_Span_A"] = senkou_span_a
    dataset["Senkou_Span_B"] = senkou_span_b
    dataset["Chikou_Span"] = chikou_span
    return dataset

def ta_ema(dataset, window=8) : #8,21,50
    """
    Calculate Exponential Moving Average (EMA) for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - span: Span for the exponential smoothing. Default is 5
    
    Returns:
    - dataset with new column
    """
    dataset["ema_{}".format(window)] = dataset["Close"].ewm(span=window, adjust=False).mean()
    return dataset

def ta_sma(dataset, window=10): #50,100
    """
    Calculate Simple Moving Average (SMA) for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Window size for the moving average. Default is 10
    
    Returns:
    - dataset with new column
    """
    dataset["sma_{}".format(window)] = dataset["Close"].rolling(window=window, min_periods=1).mean()
    return dataset

def ta_adx(dataset, window=14): #14
    """
    Calculate adx for a given column in a DataFrame
    
    parameters:
        dataset (_type_): Pandas DataFrame
        window (int, optional): _description_. Default is 14

    Returns:
 -      dataset with new column
    """
    # Calculate True Range
    dataset["TR"] = abs(dataset["High"] - dataset["Low"]).combine_first(abs(dataset["High"] - dataset["Close"].shift(1))).combine_first(abs(dataset["Low"] - dataset["Close"].shift(1)))
    # Calculate Directional Movement
    dataset["DMplus"] = (dataset["High"] - dataset["High"].shift(1)).apply(lambda x: x if x > 0 else 0)
    dataset["DMminus"] = (dataset["Low"].shift(1) - dataset["Low"]).apply(lambda x: x if x > 0 else 0)
    # Calculate Smoothed ATR and Directional Indicators
    dataset["ATR"] = dataset["TR"].rolling(window=window).mean()
    dataset["DIplus"] = (dataset["DMplus"].rolling(window=window).mean() / dataset["ATR"]) * 100
    dataset["DIminus"] = (dataset["DMminus"].rolling(window=window).mean() / dataset["ATR"]) * 100
    # Calculate ADX
    dataset["DX"] = abs(dataset["DIplus"] - dataset["DIminus"]) / (dataset["DIplus"] + dataset["DIminus"]) * 100
    dataset["ADX_{}".format(window)] = dataset["DX"].rolling(window=window).mean()
    # Drop intermediate columns if needed
    dataset.drop(["TR", "DMplus", "DMminus", "ATR", "DIplus", "DIminus", "DX"], axis=1, inplace=True)
    return dataset

def ta_donchian(dataset, window=10):
    """
    Calculate Donchian Channel for a given column in a DataFrame with a specified window size
    
    Parameters:
    - dataset: Pandas DataFrame
    - window_size: Size of the rolling window. Default is 10
    
    Returns:
    - dataset with two additional columns for Donchian Channel Upper and Lower bands
    """
    # Calculate highest high and lowest low within the rolling window
    highest_high = dataset["Close"].rolling(window=window).max()
    lowest_low = dataset["Close"].rolling(window=window).min()
    # Add new columns to the original DataFrame
    dataset["Donchian_Upper_{}".format(window)] = highest_high
    dataset["Donchian_Lower_{}".format(window)] = lowest_low
    return dataset

def ta_eri(dataset, window=13):
    """
    Calculate Elder's Force Index (ERI) for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - period: Number of periods for calculating the change in force. Default is 13
    
    Returns:
    - dataset with an additional column "ERI"
    """
    # Calculate price change
    price_change = dataset["Close"].diff()
    # Calculate Force Index
    force_index = price_change * dataset["Volume"]
    # Calculate ERI
    eri = force_index.ewm(span=window, adjust=False).mean()
    # Add "ERI" column to the dataset
    dataset["ERI_{}".format(window)] = eri
    return dataset

def ta_alma(dataset, window=10, sigma=6, offset=0.85):
    """
    Calculate Arnaud Legoux Moving Average (ALMA) for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Window size for the moving average. Default is 10
    - sigma: Standard deviation factor. Default is 6
    - offset: Offset factor. Default is 0.85
    
    Returns:
    - dataset with an additional column "ALMA"
    """
    # Calculate the weighting function
    m = np.linspace(-offset*(window-1), offset*(window-1), window)
    w = np.exp(-0.5 * (m / sigma) ** 2)
    w /= w.sum()
    # Calculate ALMA
    alma_values = np.convolve(dataset["Close"].values, w, mode="valid")
    # Add NaN values to match the length of the original dataset
    alma_values = np.concatenate([np.full(window-1, np.nan), alma_values])
    # Add "ALMA" column to the dataset
    dataset["ALMA_{}".format(window)] = alma_values
    return dataset

def ta_tsi(dataset, short_period=13, long_period=25):
    """
    Calculate True Strength Index (TSI) for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - short_period: Short EMA period. Default is 13
    - long_period: Long EMA period. Default is 25
    
    Returns:
    - Dataset with an additional column "TSI"
    """
    # Calculate price difference
    price_diff = dataset["Close"].diff(1)
    # Calculate double smoothed price difference
    double_smoothed = price_diff.ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    # Calculate double smoothed absolute price difference
    double_smoothed_abs = price_diff.abs().ewm(span=short_period, min_periods=1, adjust=False).mean().ewm(span=long_period, min_periods=1, adjust=False).mean()
    # Calculate TSI
    tsi_values = 100 * double_smoothed / double_smoothed_abs
    # Add "TSI" column to the dataset
    dataset["TSI_{}_{}".format(short_period, long_period)] = tsi_values
    return dataset

def ta_zscore(dataset, window=20):
    """
    Calculate Z-Score for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window_size: Rolling window size for calculating mean and standard deviation. Default is 20
    
    Returns:
    - Dataset with an additional column "Z_Score"
    """
    # Calculate mean and standard deviation
    rolling_mean = dataset["Close"].rolling(window=window).mean()
    rolling_std = dataset["Close"].rolling(window=window).std()
    # Calculate Z-Score
    z_score = (dataset["Close"] - rolling_mean) / rolling_std
    # Add "Z_Score" column to the dataset
    dataset["Z_Score_{}".format(window)] = z_score
    return dataset

def ta_log_return(dataset, window=5):
    """
    Calculate the log return for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Window size for calculating the log return. Default is 1
    
    Returns:
    - dataset with an additional column "LogReturn"
    """
    # Calculate log return
    dataset["LogReturn_{}".format(window)] = dataset["Close"].pct_change(window).apply(lambda x: 0 if pd.isna(x) else x)
    return dataset

def ta_kurtosis(dataset, window=20):
    """
    Calculate kurtosis for a given column in a DataFrame with a specified window size
    
    Parameters:
    - dataset: Pandas DataFrame
    - window_size: Size of the rolling window. Default is 20
    
    Returns:
    - dataset with an additional column "kurtosis"
    """
    # Calculate kurtosis using a rolling window and create a new column
    dataset["kurtosis_{}".format(window)] = dataset["Close"].rolling(window=window).apply(lambda x: np.nan if x.isnull().any() else x.kurt())
    return dataset

def ta_vortex(dataset, window=7): #14?
    """
    Calculate Vortex Indicator for a given column in a DataFrame with a specified window size
    
    Parameters:
    - dataset: Pandas DataFrame
    - window_size: Size of the rolling window. Default is 14
    
    Returns:
    - dataset with two additional columns for Positive Vortex Movement (VM+) and Negative Vortex Movement (VM-)
    """
    # Calculate True Range (TR)
    high_low = dataset["High"] - dataset["Low"]
    high_close_previous = abs(dataset["High"] - dataset["Close"].shift(1))
    low_close_previous = abs(dataset["Low"] - dataset["Close"].shift(1))
    true_range = pd.concat([high_low, high_close_previous, low_close_previous], axis=1).max(axis=1)
    # Calculate Positive Vortex Movement (VM+)
    positive_vm = abs(dataset["High"].shift(1) - dataset["Low"])
    # Calculate Negative Vortex Movement (VM-)
    negative_vm = abs(dataset["Low"].shift(1) - dataset["High"])
    # Calculate rolling sum for TR, VM+, and VM-
    true_range_sum = true_range.rolling(window=window).sum()
    positive_vm_sum = positive_vm.rolling(window=window).sum()
    negative_vm_sum = negative_vm.rolling(window=window).sum()
    # Calculate Vortex Indicator components
    positive_vi = positive_vm_sum / true_range_sum
    negative_vi = negative_vm_sum / true_range_sum
    # Add new columns to the original DataFrame
    dataset["Positive_VI_{}".format(window)] = positive_vi
    dataset["Negative_VI_{}".format(window)] = negative_vi
    return dataset

def ta_aroon(dataset, window=16):
    """
    Calculate Aroon Indicator in a DataFrame with a specified window size
    
    Parameters:
    - dataset: Pandas DataFrame
    - window_size: Size of the rolling window. Default is 16
    
    Returns:
    - dataset with two additional columns for Aroon Up and Aroon Down
    """
    high_prices = dataset["High"]
    low_prices = dataset["Low"]
    aroon_up = []
    aroon_down = []
    for i in range(window, len(high_prices)):
        high_period = high_prices[i - window:i + 1]
        low_period = low_prices[i - window:i + 1]
        high_index = window - high_period.values.argmax() - 1
        low_index = window - low_period.values.argmin() - 1
        aroon_up.append((window - high_index) / window * 100)
        aroon_down.append((window - low_index) / window * 100)

    # Pad with NaN values for the first "period" rows
    aroon_up = [None] * window + aroon_up
    aroon_down = [None] * window + aroon_down

    # Add new columns to the original DataFrame
    dataset["Aroon_Up_{}".format(window)] = aroon_up
    dataset["Aroon_Down_{}".format(window)] = aroon_down
    return dataset

def ta_ebsw(dataset, window=14):
    """
    Calculate Elder"s Bull Power and Bear Power for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - ema_window: Window for the Exponential Moving Average. Default is 14
    
    Returns:
    - dataset with additional columns for Bull Power and Bear Power
    """
    # Calculate Exponential Moving Average
    ema = dataset["Close"].ewm(span=window, adjust=False).mean()
    # Calculate Bull Power
    bull_power = dataset["High"] - ema
    # Calculate Bear Power
    bear_power = dataset["Low"] - ema
    # Add new columns to the original DataFrame
    dataset["Bull_Power_{}".format(window)] = bull_power
    dataset["Bear_Power_{}".format(window)] = bear_power
    return dataset

def ta_accbands(dataset, window=20, acceleration_factor=0.02):
    """
    Calculate Acceleration Bands for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - sma_window: Window for the Simple Moving Average. Default is 20
    - acceleration_factor: Acceleration factor for the bands. Dfault is 0.02
    
    Returns:
    - dataset with additional columns for Upper Band, Lower Band, and Middle Band (SMA)
    """
    # Calculate Simple Moving Average (SMA)
    sma = dataset["Close"].rolling(window=window).mean()
    # Calculate the difference between the Upper and Lower Bands
    band_difference = dataset["Close"] * acceleration_factor
    # Calculate Upper and Lower Bands
    upper_band = sma + band_difference
    lower_band = sma - band_difference
    # Add new columns to the original DataFrame
    dataset["Upper_Band_{}".format(window)] = upper_band
    dataset["Lower_Band_{}".format(window)] = lower_band
    dataset["Middle_Band_{}".format(window)] = sma
    return dataset

def ta_short_run(dataset, window=14):
    """
    Calculate Short Run for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Window for finding the lowest closing price. Default is 14
    
    Returns:
    - Original DataFrame with an additional column for Short Run
    """
    # Calculate Short Run as the difference between the current closing price and the lowest closing price
    short_run = dataset["Close"] - dataset["Close"].rolling(window=window).min()
    # Add the new column to the original DataFrame
    dataset["Short_Run_{}".format(window)] = short_run
    return dataset

def ta_bias(dataset, window=26):
    """
    Calculate Bias for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - moving_average_window: Window for the moving average. Default is 26
    
    Returns:
    - Original DataFrame with an additional column for Bias
    """
    # Calculate the moving average
    moving_average = dataset["Close"].rolling(window=window).mean()
    # Calculate Bias as the percentage difference
    bias = ((dataset["Close"] - moving_average) / moving_average) * 100
    # Add the new column to the original DataFrame
    dataset["Bias_{}".format(window)] = bias
    return dataset

def ta_ttm_trend(dataset, short_window=5, long_window=20):
    """
    Calculate TTM Trend for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - short_window: Window for the short-term EMA. Default is 5
    - long_window: Window for the long-term EMA. Default is 20
    
    Returns:
    - dataset with an additional column for TTM Trend
    """
    # Calculate short-term and long-term EMAs
    short_ema = dataset["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = dataset["Close"].ewm(span=long_window, adjust=False).mean()
    # Calculate TTM Trend as the difference between short-term and long-term EMAs
    ttm_trend = short_ema - long_ema
    # Add the new column to the original DataFrame
    dataset["TTM_Trend_{}_{}".format(short_window, long_window)] = ttm_trend
    return dataset

def ta_percent_return(dataset, window=1): #5/10/20
    """
    Calculate percent return for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Size of the rolling window. Default is 1
    
    Returns:
    - dataset with an additional column for percent return
    """
    # Calculate percent return as the percentage change
    percent_return = dataset["Close"].pct_change().rolling(window=window).mean() * 100
    # Add the new column to the original DataFrame
    dataset["Percent_Return_{}".format(window)] = percent_return
    return dataset

def ta_stdev(dataset, window=1): #5/10/20
    """
    Calculate standard deviation with a specified rolling window for a given column in a DataFrame
    
    Parameters:
    - dataset: Pandas DataFrame
    - window: Size of the rolling window. Default is 1
    
    Returns:
    - dataset with an additional column for standard deviation
    """
    # Calculate standard deviation with a rolling window
    stdev_column = dataset["Close"].rolling(window=window).std()
    # Add the new column to the original DataFrame
    dataset["Stdev_{}".format(window)] = stdev_column
    return dataset

# VOLATILITY

def ta_keltner_channels(dataset, period=20, multiplier=2):
    """
    Calculate Keltner Channels
    
    Parameters:
    - dataset: DataFrame with columns "Date", "High", "Low", "Close"
    - period: Window size for calculating moving averages. Default: 20
    - multiplier: Multiplier for ATR to set upper and lower bands. Default: 2
    
    Returns:
    - Modified dataset with new columns: "Middle Band", "Upper Band", "Lower Band"
    """
    # Calculate ATR
    dataset["TR"] = dataset.apply(lambda row: max(row["High"] - row["Low"], abs(row["High"] - row["Close"]), abs(row["Low"] - row["Close"])), axis=1)
    dataset["ATR"] = dataset["TR"].rolling(window=period).mean()
    # Calculate Keltner Channels
    dataset["Middle Band"] = dataset["Close"].rolling(window=period).mean()
    dataset["Upper Band"] = dataset["Middle Band"] + multiplier * dataset["ATR"]
    dataset["Lower Band"] = dataset["Middle Band"] - multiplier * dataset["ATR"]
    return dataset

def ta_vix(dataset, window=21):
    """
    Calculate a basic form of the Volatility Index (VIX) for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price information
    - window (int): The window size for volatility calculation. Default is 21
    
    Returns:
    - dataset with a new "VIX" column representing volatility
    """
    # Calculate daily returns
    returns = dataset["Close"].pct_change().dropna()
    # Calculate the rolling standard deviation of returns
    rolling_std = returns.rolling(window=window).std()
    # Calculate the VIX as a simple measure of volatility
    vix = rolling_std * np.sqrt(252) * 100  # Adjust for annualization
    # Add VIX as a new column to the original dataset
    dataset["VIX_{}".format(window)] = vix
    return dataset

def ta_chaikin_volatility(dataset, window=10):
    """
    Calculate Chaikin Volatility for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price information
    - close_column (str): The column name for closing prices. Default is "Close"
    - window (int): The window size for calculating Chaikin Volatility. Default is 10
    
    Returns:
    - dataset with a new column "Chaikin Volatility"
    """
    # Calculate daily percentage change
    daily_returns = dataset["Close"].pct_change()
    # Calculate Chaikin Volatility
    chaikin_volatility = daily_returns.rolling(window=window).std() * (252 ** 0.5)
    # Add Chaikin Volatility as a new column
    dataset["Chaikin_Volatility_{}".format(window)] = chaikin_volatility
    return dataset

def ta_atr(dataset, window=14):
    """
    Calculate ATR for a given dataset

    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price information
    - window (int): The window size for calculating ATR. Default is 14

    Returns:
    - dataset with 4 new columns
    """
    # Calculate True Range
    dataset["High-Low"] = dataset["High"] - dataset["Low"]
    dataset["High-PrevClose"] = abs(dataset["High"] - dataset["Close"].shift(1))
    dataset["Low-PrevClose"] = abs(dataset["Low"] - dataset["Close"].shift(1))
    dataset["TrueRange"] = dataset[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
    # Calculate ATR using the Exponential Moving Average (EMA)
    dataset["atr_{}".format(window)] = dataset["TrueRange"].rolling(window=window, min_periods=1).mean()
    # Drop intermediate columns
    dataset.drop(["High-Low", "High-PrevClose", "Low-PrevClose", "TrueRange"], axis=1, inplace=True)
    return dataset

def ta_chaikin_oscillator(dataset, window=3):
    """
    Calculate Chaikin Oscillator for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price and volume information
    - window (int): The window size for calculating the Chaikin Oscillator. Default is 3
    
    Returns:
    - dataset with Chaikin Oscillator values as a new column in the original dataset
    """
    # Calculate Money Flow Multiplier
    mf_multiplier = ((dataset["Close"] - dataset["Low"]) - (dataset["High"] - dataset["Close"])) / (dataset["High"] - dataset["Low"])
    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * dataset["Volume"]
    # Calculate Accumulation Distribution Line (ADL)
    adl = mf_volume.cumsum()
    # Calculate Chaikin Oscillator
    chaikin_oscillator = adl - adl.rolling(window=window).mean()
    # Add Chaikin Oscillator as a new column
    dataset["Chaikin_Oscillator_{}".format(window)] = chaikin_oscillator
    return dataset

# VOLUME

def ta_obv(dataset, window=10):
    """
    Calculate On-Balance Volume (OBV) for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price and volume information
    - window (int): The window size for calculating OBV. Default is 10
    
    Returns:
    - dataset with OBV values as a new column
    """
    # Calculate daily price changes
    price_changes = dataset["Close"].diff()
    # Assign volume direction based on price changes
    volume_direction = pd.Series(1, index=price_changes.index)
    volume_direction[price_changes < 0] = -1
    # Calculate OBV
    obv = (dataset["Volume"] * volume_direction).cumsum()
    # Smooth OBV using a rolling window
    obv_smoothed = obv.rolling(window=window).mean()
    # Add OBV as a new column
    dataset["OBV_{}".format(window)] = obv_smoothed
    return dataset

def ta_chaikin_money_flow(dataset, window=10):
    """
    Calculate Chaikin Money Flow (CMF) for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price and volume information
    - window (int): The window size for calculating CMF. Default is 10
    
    Returns:
    - dataset with Chaikin Money Flow (CMF) values as a new column
    """
    # Calculate Money Flow Multiplier
    mf_multiplier = ((dataset["Close"] - dataset["Close"].shift(1)) 
                    + (dataset["Close"] - dataset["Close"].shift(1)).abs()) / 2
    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * dataset["Volume"]
    # Calculate Accumulation/Distribution Line (ADL)
    adl = mf_volume.cumsum()
    # Calculate Chaikin Money Flow (CMF)
    cmf = adl.rolling(window=window).mean() / dataset["Volume"].rolling(window=window).mean()
    # Add CMF as a new column
    dataset["CMF_{}".format(window)] = cmf
    return dataset

def ta_volume_price_trend(dataset, window=10):
    """
    Calculate Volume Price Trend (VPT) for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price and volume information
    - window (int): The window size for calculating VPT. Default is 10
    
    Returns:
    - dataset with Volume Price Trend (VPT) values as a new column
    """
    # Calculate percentage price change
    price_change = dataset["Close"].pct_change()
    # Calculate Volume Price Trend (VPT)
    vpt = (price_change * dataset["Volume"].shift(window)).cumsum()
    # Add VPT as a new column
    dataset["VPT_{}".format(window)] = vpt
    return dataset

def ta_accumulation_distribution_line(dataset, window=10):
    """
    Calculate Accumulation/Distribution Line (A/D Line) for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price and volume information
    - window (int): The window size for smoothing the A/D Line. Default is 10
    
    Returns:
    - dataset with Accumulation/Distribution Line (A/D Line) values as a new column
    """
    # Calculate Money Flow Multiplier
    money_flow_multiplier = ((dataset["Close"] - dataset["Close"].shift(1)) - (dataset["Close"].shift(1) - dataset["Close"])) / (dataset["Close"].shift(1) - dataset["Close"])
    # Calculate Money Flow Volume
    money_flow_volume = money_flow_multiplier * dataset["Volume"]
    # Calculate Accumulation/Distribution Line (A/D Line)
    ad_line = money_flow_volume.cumsum()
    # Smooth the A/D Line using a moving average
    ad_line_smoothed = ad_line.rolling(window=window, min_periods=1).mean()
    # Add A/D Line as a new column
    dataset["A/D Line_{}".format(window)] = ad_line_smoothed
    return dataset

def ta_money_flow_index(dataset, window=14):
    """
    Calculate Money Flow Index (MFI) for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price and volume information
    - window (int): The window size for calculating MFI. Default is 14
    
    Returns:
    - dataset with Money Flow Index (MFI) values as a new column
    """
    # Calculate typical price
    typical_price = (dataset["High"] + dataset["Low"] + dataset["Close"]) / 3
    # Calculate raw money flow
    raw_money_flow = typical_price * dataset["Volume"]
    # Calculate money flow ratio
    money_flow_ratio = raw_money_flow / raw_money_flow.shift(1)
    # Calculate positive and negative money flows
    positive_money_flow = money_flow_ratio * (money_flow_ratio > 1)
    negative_money_flow = -money_flow_ratio * (money_flow_ratio < 1)
    # Calculate average positive and negative money flows using a window
    average_positive_money_flow = positive_money_flow.rolling(window=window, min_periods=1).mean()
    average_negative_money_flow = negative_money_flow.rolling(window=window, min_periods=1).mean()
    # Calculate Money Flow Index (MFI)
    mfi = 100 - (100 / (1 + average_positive_money_flow / average_negative_money_flow))
    # Add MFI as a new column
    dataset["MFI_{}".format(window)] = mfi
    return dataset

def ta_ease_of_movement(dataset, window=14):
    """
    Calculate Ease of Movement (EOM) for a given dataset
    
    Parameters:
    - dataset (pd.DataFrame): The input dataset containing price and volume information
    - window (int): The window size for calculating EOM. Default is 14
    
    Returns:
    - dataset with Ease of Movement (EOM) values as a new column
    """
    # Calculate Midpoint Move
    midpoint_move = ((dataset["High"] + dataset["Low"]) / 2).diff(1)
    # Calculate Box Ratio
    box_ratio = dataset["Volume"] / 1000000 / (dataset["High"] - dataset["Low"])
    # Calculate Ease of Movement (EOM)
    eom = midpoint_move / box_ratio
    # Smooth EOM using a window
    eom_smoothed = eom.rolling(window=window, min_periods=1).mean()
    # Add EOM as a new column
    dataset["EOM_{}".format(window)] = eom_smoothed
    return dataset


    """
MOMENTUM
Momentum indicators are used to identify the strength or weakness of a trend based on the speed of price changes.

    Relative Strength Index (RSI):
        Measures the speed and change of price movements.
        Values above 70 indicate overbought conditions, while values below 30 suggest oversold conditions.

    Moving Average Convergence Divergence (MACD):
        Compares two moving averages to identify potential buy or sell signals.
        Consists of the MACD line, signal line, and histogram.

    Stochastic Oscillator:
        Measures the closing price relative to the high-low range over a specific period.
        Helps identify overbought and oversold conditions.

    Average True Range (ATR):
        Measures market volatility.
        Reflects the average range between high and low prices over a specified period.

    Momentum Indicator:
        Compares the current closing price to a specific historical price.
        Indicates the strength of a trend.

    Rate of Change (ROC):
        Measures the percentage change in price over a specified period.
        Helps identify the speed of price movements.

    Commodity Channel Index (CCI):
        Measures the current price level relative to an average price level over a given period.
        Identifies overbought and oversold conditions.

    Williams %R:
        Similar to the Stochastic Oscillator, measures the current closing price relative to the high-low range.
        Helps identify potential reversal points.

    Bollinger Bands:
        Consists of a middle band being an N-period simple moving average and upper/lower bands representing N standard deviations from the moving average.
        Useful for identifying volatility and potential reversal points.

    Chaikin Money Flow (CMF):
        Combines price and volume to measure buying and selling pressure.
        Positive values suggest buying pressure, while negative values suggest selling pressure.

TREND
Trend indicators help traders identify the direction of the price movement and whether an asset is in an uptrend, downtrend, or ranging.

    Moving Averages:
        Simple Moving Average (SMA) and Exponential Moving Average (EMA) are commonly used.
        Helps smooth out price data to identify the overall direction of the trend.

    Trendlines:
        Drawn by connecting higher lows in an uptrend or lower highs in a downtrend.
        Provides a visual representation of the trend direction.

    Average Directional Index (ADX):
        Measures the strength of a trend.
        Values above 25 indicate a strong trend, while values below 20 suggest a weak trend.

    Parabolic SAR (Stop and Reverse):
        Places dots above or below the price to indicate potential trend reversals.
        Useful for identifying potential entry and exit points.

    Ichimoku Cloud:
        Consists of several components, including the Kumo (cloud), Tenkan Sen, and Kijun Sen.
        Helps identify trend direction and potential support/resistance levels.

    Moving Average Convergence Divergence (MACD):
        Besides being a momentum indicator, it can also signal changes in trend direction.
        Bullish and bearish crossovers provide trend reversal signals.

    Bollinger Bands:
        Helps identify volatility and potential trend reversals.
        Price tends to stay within the bands during a strong trend.

    Donchian Channels:
        Consists of an upper and lower channel based on the highest high and lowest low over a specified period.
        Identifies the current trend direction.

    Elder-Ray Index:
        Combines the Bull Power and Bear Power indicators.
        Bull Power measures the strength of the bulls during an uptrend, and Bear Power measures the strength of the bears during a downtrend.

    Super Trend:
        Plots a line above or below the price to indicate the current trend direction.
        Flip in direction suggests a potential trend reversal.

OSCILLATOR
Oscillator indicators are designed to identify potential overbought or oversold conditions in the market and help traders gauge the momentum of price movements.

    Relative Strength Index (RSI):
        Measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
        Typically uses a 14-period setting.

    Stochastic Oscillator:
        Consists of %K and %D lines to identify potential reversal points.
        Readings above 80 suggest overbought conditions, while readings below 20 suggest oversold conditions.

    Moving Average Convergence Divergence (MACD):
        Besides being a trend indicator, it also functions as an oscillator.
        The MACD histogram visualizes the difference between the MACD line and the signal line.

    Commodity Channel Index (CCI):
        Measures the deviation of the price from its statistical average.
        Readings above +100 may indicate overbought conditions, while readings below -100 may suggest oversold conditions.

    Williams %R:
        Reflects the level of the close relative to the highest high over a specified period.
        Values above -20 indicate overbought conditions, while values below -80 suggest oversold conditions.

    Average True Range (ATR):
        Measures market volatility.
        Can be used to set stop-loss levels and assess potential trend strength.

    Relative Vigor Index (RVI):
        Measures the conviction of a recent price action and its relationship to the trading range.
        Helps identify potential trend reversals.

    Money Flow Index (MFI):
        Combines price and volume to measure the strength of money flowing in and out of a security.
        Values above 80 may indicate overbought conditions, while values below 20 may suggest oversold conditions.

    Chaikin Oscillator:
        Combines the Accumulation/Distribution Line with an exponential moving average.
        Helps identify changes in the momentum of buying and selling pressure.

    Detrended Price Oscillator (DPO):
        Measures the difference between a past price and a simple moving average.
        Aims to eliminate the trend and highlight short-term cycles.

VOLATILITY
Volatility indicators are crucial for assessing the price fluctuations and risk associated with an asset.

    Average True Range (ATR):
        Measures market volatility by calculating the average range between the high and low prices over a specified period.
        A higher ATR suggests higher volatility.

    Bollinger Bands:
        Consist of a middle band (simple moving average) and upper/lower bands that represent volatility.
        Wider bands indicate higher volatility, while narrower bands suggest lower volatility.

    Volatility Index (VIX):
        Known as the "Fear Index," VIX measures market expectations for future volatility.
        Higher VIX values indicate higher expected volatility.

    Chaikin Volatility:
        Compares the spread between high and low prices to the trading range.
        Higher values suggest higher volatility.

    Historical Volatility (HV):
        Measures the actual price changes observed in the past.
        Comparing current volatility to historical levels helps assess the market's current state.

    Keltner Channels:
        Similar to Bollinger Bands but use the Average True Range to set channel boundaries.
        Wider channels indicate higher volatility.

    Standard Deviation:
        Measures the dispersion of prices from their average.
        A higher standard deviation indicates higher volatility.

    Ulcer Index:
        Measures the depth and duration of drawdowns in prices from earlier highs.
        A higher Ulcer Index indicates higher volatility.

    Volatility Smile:
        Commonly used in options pricing, it represents implied volatility across different strike prices.
        Helps assess market sentiment regarding future price movements.

    Hindenburg Omen:
        Flags potential market crashes based on a series of technical analysis criteria.
        It's more of a bearish signal related to market internals but can be considered as a volatility indicator.

VOLUME
Volume indicators are essential for analyzing trading activity and market sentiment.

    On-Balance Volume (OBV):
        Measures cumulative buying and selling pressure based on the volume.
        Rising OBV suggests buying interest, while falling OBV indicates selling pressure.

    Chaikin Money Flow (CMF):
        Combines price and volume to assess the flow of money in or out of a security.
        Positive CMF indicates buying pressure, while negative CMF suggests selling pressure.

    Accumulation/Distribution Line:
        Measures the cumulative flow of money into or out of a security based on the close.
        Rising line suggests accumulation, while falling line indicates distribution.

    Volume Price Trend (VPT):
        Combines percentage price change and volume to assess the strength of a trend.
        Rising VPT confirms an uptrend, while falling VPT confirms a downtrend.

    Money Flow Index (MFI):
        Combines price and volume to measure the strength of buying and selling pressure.
        Values over 80 suggest overbought conditions, while values below 20 suggest oversold conditions.

    Ease of Movement (EOM):
        Measures the relationship between price change and volume.
        Helps assess the ease with which prices move in a particular direction.

    Volume Weighted Average Price (VWAP):
        Calculates the average price based on both volume and price.
        Useful for identifying significant price levels based on volume.

    Force Index:
        Multiplies price change and volume to assess the force behind a price movement.
        Positive Force Index indicates buying pressure, while negative Force Index suggests selling pressure.

    Demand Index:
        Compares price and volume to identify potential demand and supply imbalances.
        Rising Demand Index indicates increasing buying interest.

    Price Volume Trend (PVT):
        Measures the relationship between price and volume, similar to the On-Balance Volume.
        Helps identify the strength of a price trend.

    """

#--------------------------------------------------------------------------------------------------------------

def calendar_features(dataset) :
    """
    Create new calendar features from date (index)
    
    Parameters:
        dataset (dataframe): original dataset
        
    Returns:
        dataset: dataset with new columns containing target prices
    """
    dataset["month"] = pd.DatetimeIndex(dataset.index).month
    dataset["day"] = pd.DatetimeIndex(dataset.index).day
    dataset["year"] = pd.DatetimeIndex(dataset.index).year
    dataset["day_of_week"] = pd.DatetimeIndex(dataset.index).day_of_week
    dataset.sort_index(ascending=False)
    return dataset

def create_targets(dataset, horizon=3) :
    """
    Create targets data (price to predict) from close price and horizon of time
    
    Parameters:
        dataset (dataframe): original dataset
        horizon (int): horizon of time for witch we want a target price. Defaults to 7
        
    Returns:
        dataset: dataset with new columns containing target prices
    """
    for i in range(1,horizon+1,1) :
        dataset["targetvalue_j{}".format(i)] = dataset["Close"].shift(-i)
    dataset["target_night"] = dataset["Open"].shift(-1)
    return dataset