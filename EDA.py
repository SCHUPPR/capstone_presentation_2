# Package imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import holidays
# Dataset imports
from ETL_Preprocessing import demand_df_ts, merged_ml_ts, get_classical_partitions, get_ml_partitions

# Preparing graphics settings


sns.set_theme(
    style="whitegrid",
    context="paper",
    palette="colorblind"
)

plt.rcParams.update({
    "figure.figsize": (8,5),
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

'''
Matplotlib.pyplot color strings:
b: blue, g: green, r: red, c: cyan, m: magenta, y: yellow, k: black, w: white
'''

'''
General Exploratory Data Analysis
'''

def get_line_graph(df, item_category, line_color, file_name=None):
    '''
    generates line graphs of demand time series
    '''
    plt.figure(figsize=(10,5))
    # Converting period series to timestamp dtype
    plt.plot(df.index.to_timestamp(), df[item_category], marker="o",
             linestyle="-", color=line_color, label=f"{item_category} Units Ordered")
    # Formatting
    plt.title(f"{item_category} Time Series", fontsize=16)
    plt.xlabel("YearMonth Period", fontsize=12)
    plt.ylabel("Units Ordered", fontsize=12)
    # Setting grid background
    plt.grid(True, linestyle="--", alpha=0.6)
    # Rotating date labels for readability
    plt.xticks(rotation=45)
    # Setting tight layout to prevent label cutoff
    plt.tight_layout
    # Save plot
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    # Displaying plot
    plt.show()
    
    
def verify_df_partition(df, expected_periods):
    '''Verifys dataset partitioning'''
    min_period = min(df.index)
    max_period = max(df.index)
    # Checking number of periods
    if df.shape[0] != expected_periods:
        print(f"Partition failure, check partitions\nPeriod Range: {min_period} - {max_period}\nPeriod Count: {len(df.index)}")
    else:
        print(f"Partition successful.\nPeriod Range: {min_period} - {max_period}\nPeriod Count: {len(df.index)}")


# Creating training set partition
train_set_classical, _ = get_classical_partitions(31, demand_df_ts)
# Verifying test partition
#verify_df_partition(train_set_classical, 31)

item_categories = [col for col in demand_df_ts.columns if col != "keg_sales"]

''' Get demand line graphs
# Line graphs to visualize time series, identify trend and seasonality
for cat in item_categories:
    file_name = cat + " Line Graph"
    get_line_graph(train_set_classical, cat, "b", file_name=file_name)
'''
 
'''
Beer: exhibits clear seasonality, consistent peaks during summer periods and trough in dec-jan, potential + trend starting in 2020,
    seasonal magnitude is consistent (additive)
Liquor: exhibits clear seasonality, large peaks at year end followed by consistent drop offs, potential + trend starting in 2020
    seasonal magnitude is seemingly consistent (additive)
Non-Alcoholic: No clear seasonality, potential + trend starting  mid 2019
Wine: clear seasonality, consistent peaks at year end and ~May, no clear trend, seasonal magnitutde appears consistent (additive)
'''

'''
Classical Model EDA
- Determine if data exhibits seasonality ARIMA vs SARIMA
- Determine if ts is stationary (assumption of ARIMA and SARIMA), if not, difference
- Can check stationarity with Augmented Dickey-Fuller test (significant = stationary)
'''

# Creating training set partition
train_set_classical, _ = get_classical_partitions(31, demand_df_ts)
# Verifying test partition
#verify_df_partition(train_set_classical, 31)
    
'''
Time Series Decomposition
'''

def get_ts_decomposition_plot(df, item_category, file_name=None):
    '''
    plots time series decompositions
    '''
    # Creating copy of df to avoid any changes outside of function
    df_copy = df.copy()
    # Convert index to timestamp
    df_copy.index = df_copy.index.to_timestamp()
    df_copy = df_copy.asfreq("MS")
    # Interpolating missing value (STL would fail with the missing value)
    df_copy = df_copy.interpolate()
    stl = STL(df_copy[item_category],period=12)
    stl_result = stl.fit()
    stl_result.plot()
    # Rotating date labels for readability
    plt.xticks(rotation=45)
    # Save plot
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    

''' Get Time Series Decomposition
for cat in item_categories:
    file_name = cat + " Decomposition"
    get_ts_decomposition_plot(train_set_classical, cat, file_name=file_name)
'''
    
'''
Beer: Weak increasing trend, clear seasonality around summer, residuals show consistent variance
Liquor: No trend -> weak increasing trend, clear seasonality around December, constant variance of residuals
Non-Alcoholic: Increasing trend then plateaus, seasonal decrease around Jan-Feb, constant variance of residuals
Wine: Weak decreasing trend followed by plateau, clear seasonality around EOY, constant variance of residuals

Overall, all ts seem to be stationary, with all datasets but maybe non-alcohol exhibiting clear seasonality and no distinct trend
'''

'''
Stationary vs Non-Stationary Check
'''

# Will also use the Augmented Dickey-Fuller hypothesis test

def get_adfuller_test(df, item_category):
    '''
    ADFuller hypothesis test for stationarity
    '''
    test_results = adfuller(df[item_category])
    if test_results[1] <= 0.05:
        signif = "Stationary"
    else:
        signif = "Non-Stationary"
    return test_results[1], signif


#for cat in item_categories:
#    print(cat, get_adfuller_test(train_set_classical, cat))

'''
Augmented Dickey-Fuller hypothesis test suggest all four ts are stationary
'''

# In addition to ADF, We'll use the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test

def get_kpss_test(df, item_category):
    test_results = kpss(df[item_category], regression="c", nlags="auto")
    test_statistic, p_val = test_results[0], test_results[1]
    if p_val <= 0.05:
        signif="Non Stationary"
    else:
        signif="Stationary"
    return test_statistic, p_val, signif

#for cat in item_categories:
#    print(cat, get_kpss_test(train_set_classical, cat))

'''
KPSS hypothesis test also suggests all four ts are stationary
'''

'''
Autocorrelations (ACF) and Partial Autocorrelations (PACF)
'''

def get_acf_pacf_plots(df, item_category, acf_lags=None, pacf_lags=None, file_name=None):
    '''
    generates acf and pacf plots
    '''
    train_set_copy = df.copy()
    if acf_lags is None:
        acf_lags = 12
    if pacf_lags is None:
        pacf_lags = 12
    fig, axes = plt.subplots(2,1,figsize=(12,12))
    plot_acf(train_set_copy[item_category],lags=acf_lags,ax=axes[0])
    plot_pacf(train_set_copy[item_category],lags=pacf_lags,ax=axes[1])
    axes[0].set_title(f"{item_category} ACF Plot")
    axes[1].set_title(f"{item_category} PACF Plot")
    # Save plot
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.show()

#train_set_classical_interpolated = train_set_classical.interpolate()
#plot_acf(train_set_classical_interpolated["liquor_sales"], lags=30)
#plot_pacf(train_set_classical_interpolated["liquor_sales"], lags=12)

'''#Getting ACF and PACF Plots
for cat in item_categories:
    file_name = cat + " ACF PCF Plots"
    get_acf_pacf_plots(train_set_classical, cat, file_name=file_name)
'''
    
'''
Beer: Significant lag in both plots at lag=1, partial autocorrelation lags at 3,4,7
Liquor: Significant lag in both plots at lag=12
Non-Alcoholic: No significant lags in either plot
Wine: Significant lag=12 in autocorrelation plot, none in partial autocorrelations

With the ACF, PACF plots, and the Augmented Dickey-Fuller tests, it would suggest
that the time series (ts) do not need to be differenced prior to model fitting
'''

'''
Machine Learning Model EDA
'''

# Creating training set partition
train_set_ml, _ = get_ml_partitions(31, merged_ml_ts)
# Verifying test partition
#verify_df_partition(train_set_classical, 31)

# Repurposed line graph function for supporting variables
def get_line_graph_supp(df, column, line_color):
    plt.figure(figsize=(10,5))
    # Converting period series to timestamp dtype
    plt.plot(df.index.to_timestamp(), df[column], marker="o",
             linestyle="-", color=line_color, label=f"{column}")
    # Formatting
    plt.title(f"{column} Time Series", fontsize=16)
    plt.xlabel("YearMonth Period", fontsize=12)
    # Setting grid background
    plt.grid(True, linestyle="--", alpha=0.6)
    # Rotating date labels for readability
    plt.xticks(rotation=45)
    # Setting tight layout to prevent label cutoff
    plt.tight_layout
    # Displaying plot
    plt.show()

    
'''#Generate line graphs of supporting variables
for col in train_set_ml:
    if col in ["beer_sales", "liquor_sales", "nonalcoholic_sales", "keg_sales", "wine_sales"]:
        pass
    else:
        get_line_graph_supp(train_set_ml,col,"b")
'''

''' Line Graph Observations
CPI variables increased over time (alcohol plateaus from 2018-05 - 2020-08)
Employment has a sharp decrease in 2020-04, Unemployment has a increase during the same period
Most employment/unemployment variables seem to capture the same pattern / are redundant
Temperature variables fluctuate/cycle by season
Precipitation has no distinct trend or seasonality, average around 5 inches / month
'''

# Now we'll create scatterplots to visualize the relationships between supporting variables and demand

def get_scatter_plot_demand(df, alc_col_list, supp_col_list):
    # iterate through all combinations
    for alc in alc_col_list:
        for supp in supp_col_list:            
            plt.figure(figsize=(10,5))          
            plt.scatter(df[supp], df[alc], marker="o")
            plt.title(f"{alc} by {supp}", fontsize=16)
            plt.xlabel(f"{supp}", fontsize=12)
            plt.ylabel(f"{alc}", fontsize=12)
            # Setting grid background
            plt.grid(True, linestyle="--", alpha=0.6)
            # Rotating date labels for readability
            plt.xticks(rotation=45)
            # Setting tight layout to prevent label cutoff
            plt.tight_layout
            # Displaying plot
            plt.show()


# Creating lists of supporting variable groupings
supp_var_list = ["all_cpi", "alcohol_cpi", "employment", "unemployment", "avg_temp_f", "precipitation_inches"]
cpi_var_list = ["all_cpi", "alcohol_cpi"]
employment_var_list = ["employment", "unemployment"]
weather_var_list = ["avg_temp_f", "precipitation_inches"]
redundant_var_list = [col for col in train_set_ml.columns if col not in supp_var_list and col not in item_categories]

'''# Generate Scatterplots
# CPI scatterplots
get_scatter_plot_demand(train_set_ml, item_categories, cpi_var_list)
# Employment scatterplots
get_scatter_plot_demand(train_set_ml, item_categories, employment_var_list)
# Weather scatterplots
get_scatter_plot_demand(train_set_ml, item_categories, weather_var_list)
'''

''' Scatterplot Observations
No clear relationship with all_cpi or alc_cpi
No clear relationship with employment
No distinct relationship with precipitation; beer increase with temp
'''

'''
Correlation Matrix
'''

'''
# Generate Corrlation Matrix
# Rounding for the sake of visualization
ml_correlation_matrix = train_set_ml.corr().round(2)
# Plotting the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(ml_correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Demand & Supporting Variable Corr Matrix")
# Save plot
plt.savefig("base_correlation_matrix", dpi=300, bbox_inches="tight")
plt.show()
'''


''' Correlation Matrix Observations
Confirmed that all employment variables are highly correlated with one another and unemployment
min, max, and avg temperatures are also almost perfectly correlated unsurprisingly
Beer has a high correlation with temperature variables
Both cpi variables are highly correlated
These leads me to believe variable reduction techniques will be useful (PCA)
'''

# Creating separate time series dataframes for each drink type
train_set_wine_ml = train_set_ml.copy().drop(columns=["beer_sales", "liquor_sales", "nonalcoholic_sales"])
train_set_beer_ml = train_set_ml.copy().drop(columns=["wine_sales", "liquor_sales", "nonalcoholic_sales"])
train_set_liquor_ml = train_set_ml.copy().drop(columns=["wine_sales", "beer_sales", "nonalcoholic_sales"])
train_set_nonalcoholic_ml = train_set_ml.copy().drop(columns=["wine_sales", "beer_sales", "liquor_sales"])


'''
Feature Engineering
'''

# Function to create lag demand values
# Will consider using only small lags, such as 1 and/or 3 due to small training size
def add_lag_columns(df, lag_list):
    '''
    Input
        df: dataframe for ML forecasting
        lag_list: list of desired lag values (type int)
    Output: dataframe with added lag columns of sales
    '''
    df_w_lags = df.copy()
    
    if lag_list:
        # Filter just to the demand/sales column
        product_col = [col for col in df.columns if col[-5:] == "sales"]
        for lag in lag_list:
            col_name = "sales_lag_" + str(lag)
            df_w_lags[col_name] = df_w_lags[product_col].shift(lag)
    
    return df_w_lags


# Function to create rolling mean/moving average columns
def add_roll_mean_columns(df, roll_list):
    '''
    Input
        df: dataframe for ML forecasting
        roll_list: list of ranges to calculate the rolling mean for (type int)
    Output: dataframe with added columns for the respective list of rolling means
    '''
    df_w_roll_means = df.copy()
    
    if roll_list:
        # Filter just to the demand/sales column
        product_col = [col for col in df.columns if col[-5:] == "sales"]
        for roll in roll_list:
            col_name = "rolling_mean_" + str(roll)
            # Shifting first to correctly calculate mean of past values
            df_w_roll_means[col_name] = df_w_roll_means[product_col].shift(1).rolling(3).mean()
    
    return df_w_roll_means


# Cosine and Sine functions to model seasonality (better than one-hot encoding monthly values)
# Used ChatGPT to help create the mathematical functions themselves, not write the code to create the columns or Python function
def add_cosine_sine_columns(df):
    '''
    Input: dataframe for ML forecasting
    Output: dataframe with added sine and cosine columns or seasonal modeling
    '''
    df_new_cols = df.copy()
    df_new_cols["month"] = df_new_cols.index.month
    df_new_cols["month_sin"] = np.sin(2 * np.pi * df_new_cols["month"] / 12)
    df_new_cols["month_cos"] = np.cos(2 * np.pi * df_new_cols["month"] / 12)
    # Dropping the month column
    df_new_cols.drop(columns=["month"], inplace=True)
    
    return df_new_cols


# Function to create holiday count columns (national and state)
def add_holiday_count_columns(df):
    '''
    Input: dataframe for ML forecasting
    Output: dataframe with added holiday_count column
    '''
    # Creating holiday package object of US holidays
    us_holidays = holidays.US(years=range(2017,2021))
    # Dict for storing holiday counts by month
    holiday_count = {}
    for date, name in us_holidays.items():
        # Filter out duplicate (observed) holidays
        if name[-1:] == ")":
            pass
        else:
            key = date.strftime("%B %Y")
            if key not in holiday_count.keys():
                holiday_count[key] = 1
            elif key in holiday_count.keys():
                holiday_count[key] += 1
                
    # Mapping dict values to dataframe
    df_new_cols = df.copy()
    df_new_cols["month_year"] = df_new_cols.index.to_timestamp().strftime("%B %Y")
    # Value is zero if key doesn't exist
    df_new_cols["holiday_count"] = df_new_cols["month_year"].map(holiday_count).fillna(0)
    df_new_cols.drop(columns=["month_year"], inplace=True)
    
    return df_new_cols


def add_supporting_var_lag_columns(df, supporting_var_list=[]):
    '''
    As we can't use future predictor variable values to predict the next month,
    we have to lag them all by one month. This makes it so that the model will predict next month's
    demand using the 'current' month's values
    These variables include weather, economic, and employment
    '''
    df_new_cols = df.copy()
    # If there's a list of variables, create lags and then remove base variable
    if len(supporting_var_list) >= 1:
        for supporting_var in supporting_var_list:
            supporting_var_lag1 = supporting_var + "_lag1"
            if supporting_var in df_new_cols.columns:
                df_new_cols[supporting_var_lag1] = df_new_cols[supporting_var].shift(1)
                
        # Remove the non-lag supporting variable columns to prevent data leakage
        df_new_cols.drop(columns=supporting_var_list, inplace=True)
    
    return df_new_cols
    

# Function that aggregates all column-add functions
def add_all_eng_columns(df, lag_list=None, roll_list=None, supporting_var_list=[]):
    '''
    Inputs
        df: dataframe for ML forecasting
        lag_list: list of int lag values (months to lag by)
        roll_list: list of int rolling mean values (months to go back and avg)
    Output: df_new_cols: dataframe with added columns from feature engineering portion
    '''
    df_new_cols = df.copy()
    if lag_list:
        df_new_cols = add_lag_columns(df_new_cols, lag_list)
    if roll_list:
        df_new_cols = add_roll_mean_columns(df_new_cols, roll_list)
    df_new_cols = add_cosine_sine_columns(df_new_cols)
    df_new_cols = add_holiday_count_columns(df_new_cols)
    df_new_cols = add_supporting_var_lag_columns(df_new_cols, supporting_var_list)
    
    return df_new_cols


# Testing functions
#print(add_all_eng_columns(train_set_beer_ml, lag_list=[1,3], roll_list=[3]).head())

'''
Due to the small train partition size, and dataset size in general, I'll be creating
two dataframes for ML testing. One will have lag and roll mean features up to 3, 
and the other will have no lag or roll mean features. This will be for testing
if omitting 3 months worth of data is worth it in terms of model performance.
'''

train_set_ml_eng = add_all_eng_columns(train_set_wine_ml, lag_list=[1,3], roll_list=[3])

train_set_ml_eng.drop(columns=["alcohol_cpi", "labor_force_participation_rate", "employment_population_ratio",
                               "min_temp_f", "max_temp_f", "unemployment_rate"], inplace=True)
train_set_ml_eng = train_set_ml_eng.dropna()

'''
Engineered Correlation Matrix
'''

'''# Generate Corrlation Matrix
# Rounding for the sake of visualization
ml_eng_correlation_matrix = train_set_ml_eng.corr().round(2)
# Plotting the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(ml_eng_correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Demand & Supporting Variable Corr Matrix (Engineered Features)")
# Save plot
plt.savefig("engineered_correlation_matrix", dpi=300, bbox_inches="tight")
plt.show()
'''

'''
Generating datasets for model building
'''

wine_ml_ts = merged_ml_ts.copy().drop(columns=["beer_sales", "liquor_sales", "nonalcoholic_sales"])
beer_ml_ts = merged_ml_ts.copy().drop(columns=["wine_sales", "liquor_sales", "nonalcoholic_sales"])
liquor_ml_ts = merged_ml_ts.copy().drop(columns=["wine_sales", "beer_sales", "nonalcoholic_sales"])
nonalcoholic_ml_ts = merged_ml_ts.copy().drop(columns=["wine_sales", "beer_sales", "liquor_sales"])

''' Adding this section due to small dataset size and issues with ML models
Data back-extending with seasonal imputation
'''

def extend_time_series_backward_with_trend(df, n_periods, noise_scale=0.05, add_noise=True, add_flag=True, trend_strength=1.0):
    """
    Extend a monthly PeriodIndex dataframe backward using:
    - Monthly averages (seasonality)
    - Linear trend extrapolation
    - Optional noise

    Parameters:
    -----------
    df : pd.DataFrame
        Must have a PeriodIndex with freq='M'
    n_periods : int
        Number of months to extend backward
    noise_scale : float
        Noise level as a fraction of each column's std (default 0.1)
    add_noise : bool
        Whether to add Gaussian noise
    add_flag : bool
        Whether to add 'is_synthetic' column
    trend_strength : float
        Multiplier for trend effect (1.0 = full trend, 0 = no trend)

    Returns:
    --------
    pd.DataFrame
        Extended dataframe
    """

    df = df.copy()

    # --- Validate index ---
    if not isinstance(df.index, pd.PeriodIndex):
        raise ValueError("Index must be a pandas PeriodIndex")
    if df.index.freq != 'M':
        df.index = df.index.asfreq('M')
    # --- Monthly averages ---
    monthly_avg = df.groupby(df.index.month).mean()
    # --- Create backward index ---
    start = df.index.min()
    new_index = pd.period_range(end=start - 1, periods=n_periods, freq='M')
    # --- Time index for trend (numeric) ---
    t = np.arange(len(df))
    synth_data = {}

    for col in df.columns:
        y = df[col].values

        # --- Fit linear trend ---
        if np.all(np.isfinite(y)) and len(y) > 1:
            slope, intercept = np.polyfit(t, y, 1)
        else:
            slope, intercept = 0, np.nanmean(y)

        col_std = np.nanstd(y)
        col_noise = noise_scale * col_std if add_noise else 0

        values = []

        # Generate backward points
        for i, p in enumerate(reversed(new_index)):
            # Position BEFORE the start of the series
            t_new = - (i + 1)
            # Trend component
            trend_val = intercept + slope * t_new
            # Seasonal component
            seasonal_val = monthly_avg.loc[p.month, col]
            # Combine (center seasonal around trend)
            value = seasonal_val + trend_strength * (trend_val - np.mean(y))
            # Add noise
            if add_noise:
                value += np.random.normal(0, col_noise)

            values.append(value)

        # Reverse back to chronological order
        values = list(reversed(values))
        synth_data[col] = values

    df_synth = pd.DataFrame(synth_data, index=new_index)
    # --- Optional flag ---
    if add_flag:
        df_synth['is_synthetic'] = 1
        df['is_synthetic'] = 0
    # --- Combine ---
    df_extended = pd.concat([df_synth, df]).sort_index()

    return df_extended


# Imputing missing demand point
def impute_missing_demand(df, col_name):
    df_imp = df.copy()
    df_imp.loc[df_imp[col_name].isna(),col_name] = df_imp[col_name].shift(12)
    return df_imp

wine_ml_ts = impute_missing_demand(wine_ml_ts, "wine_sales")
beer_ml_ts = impute_missing_demand(beer_ml_ts, "beer_sales")
liquor_ml_ts = impute_missing_demand(liquor_ml_ts, "liquor_sales")
nonalcoholic_ml_ts = impute_missing_demand(nonalcoholic_ml_ts, "nonalcoholic_sales")

'''
Creating Extended Dataset
'''
wine_ml_ts_imputed = extend_time_series_backward_with_trend(wine_ml_ts, 60, add_flag=False)
beer_ml_ts_imputed = extend_time_series_backward_with_trend(beer_ml_ts, 60, add_flag=False)
liquor_ml_ts_imputed = extend_time_series_backward_with_trend(liquor_ml_ts, 60, add_flag=False)
nonalcoholic_ml_ts_imputed = extend_time_series_backward_with_trend(nonalcoholic_ml_ts, 60, add_flag=False)

# Initalizing datasets with engineered features
# Creating list of supporting/predictor variables to create a lag of
supporting_var_lag1_list = merged_ml_ts.columns[4:]

'''
Normal Lagged Datasets
'''
wine_ml_ts_lag3 = add_all_eng_columns(wine_ml_ts, lag_list=[1,3], roll_list=[3], supporting_var_list=supporting_var_lag1_list)
wine_ml_ts_lag1 = add_all_eng_columns(wine_ml_ts, lag_list=[1], supporting_var_list=supporting_var_lag1_list)
beer_ml_ts_lag3 = add_all_eng_columns(beer_ml_ts, lag_list=[1,3], roll_list=[3], supporting_var_list=supporting_var_lag1_list)
beer_ml_ts_lag1 = add_all_eng_columns(beer_ml_ts, lag_list=[1], supporting_var_list=supporting_var_lag1_list)
liquor_ml_ts_lag3 = add_all_eng_columns(liquor_ml_ts, lag_list=[1,3], roll_list=[3], supporting_var_list=supporting_var_lag1_list)
liquor_ml_ts_lag1 = add_all_eng_columns(liquor_ml_ts, lag_list=[1], supporting_var_list=supporting_var_lag1_list)
nonalcoholic_ml_ts_lag3 = add_all_eng_columns(nonalcoholic_ml_ts, lag_list=[1,3], roll_list=[3], supporting_var_list=supporting_var_lag1_list)
nonalcoholic_ml_ts_lag1 = add_all_eng_columns(nonalcoholic_ml_ts, lag_list=[1], supporting_var_list=supporting_var_lag1_list)

'''
Extended datasets
'''
wine_ml_ts_lag12 = add_all_eng_columns(wine_ml_ts, lag_list=[1,3,6,12], roll_list=[2,3,6], supporting_var_list=supporting_var_lag1_list)
wine_ml_ts_lag12_imputed = add_all_eng_columns(wine_ml_ts_imputed, lag_list=[1,3,6,12], roll_list=[2,3,6], supporting_var_list=supporting_var_lag1_list)
beer_ml_ts_lag12_imputed = add_all_eng_columns(beer_ml_ts_imputed, lag_list=[1,3,6,12], roll_list=[2,3,6], supporting_var_list=supporting_var_lag1_list)
liquor_ml_ts_lag12_imputed = add_all_eng_columns(liquor_ml_ts_imputed, lag_list=[1,3,6,12], roll_list=[2,3,6], supporting_var_list=supporting_var_lag1_list)
nonalcoholic_ml_ts_lag12_imputed = add_all_eng_columns(nonalcoholic_ml_ts_imputed, lag_list=[1,3,6,12], roll_list=[2,3,6], supporting_var_list=supporting_var_lag1_list)



