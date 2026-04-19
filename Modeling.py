import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import random

# Importing datasets
from EDA import (demand_df_ts, wine_ml_ts_lag1, wine_ml_ts_lag3, beer_ml_ts_lag1, beer_ml_ts_lag3,
                 liquor_ml_ts_lag3, liquor_ml_ts_lag1, nonalcoholic_ml_ts_lag3, nonalcoholic_ml_ts_lag1,
                 wine_ml_ts_lag12, wine_ml_ts_lag12_imputed, beer_ml_ts_lag12_imputed,
                 liquor_ml_ts_lag12_imputed, nonalcoholic_ml_ts_lag12_imputed)
from ETL_Preprocessing import get_classical_partitions, get_ml_partitions

# Changing global settings to allow all rows and columns to display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''
Imputing demand data points in classical dataset
'''
# Reindexing because NaN month 2020/08 was dropped
complete_index = pd.period_range(start=demand_df_ts.index.min(),end=demand_df_ts.index.max(),
                                 freq="M")
demand_df_ts = demand_df_ts.reindex(complete_index)
# Imputing the missing values in 2020/08 with the same value from the last period
for col in demand_df_ts.columns:
    demand_df_ts.loc[demand_df_ts[col].isna(),col] = demand_df_ts[col].shift(12)
# Verify no missing values
#print(demand_df_ts.isna())


'''
Metric Tracking
'''
model_evaluation_metrics = {"dataset": [], "model": [], "RMSE": [], "MAPE": []}

def store_evaluation_metrics(df, sales_col, dataset_name, model_name):
    # Ensures it's just the predicted periods
    df_copy = df.copy()
    df_copy.dropna()
    rmse = np.sqrt(((df_copy[sales_col] - df_copy["predicted_sales"]) ** 2).mean())
    mape = (np.abs((df_copy[sales_col] - df_copy["predicted_sales"]) / df_copy[sales_col]).mean()) * 100
    # store values
    model_evaluation_metrics["dataset"].append(dataset_name)
    model_evaluation_metrics["model"].append(model_name)
    model_evaluation_metrics["RMSE"].append(rmse)
    model_evaluation_metrics["MAPE"].append(mape)


'''
Classical Modeling Section
'''
# Initializing classical train:test splits
wine_classical_train, wine_classical_test = get_classical_partitions(31, demand_df_ts["wine_sales"])
beer_classical_train, beer_classical_test = get_classical_partitions(31, demand_df_ts["beer_sales"])
liquor_classical_train, liquor_classical_test = get_classical_partitions(31, demand_df_ts["liquor_sales"])
nonalcoholic_classical_train, nonalcoholic_classical_test = get_classical_partitions(31, demand_df_ts["nonalcoholic_sales"])

'''
SARIMA
'''


def grid_search_sarima_cv(train_set, p=[0], d=[0], q=[0], P=[0], D=[0], Q=[0], period=[12], n_splits=5):
    '''GridSearch for SARIMA parameters
    Returns the best order and seasonal order'''

    # Ensure 1D array
    train_values = np.array(train_set).reshape(-1)
    # Generate parameter combinations
    pdq_combinations = list(itertools.product(p, d, q))
    PDQ_combinations = list(itertools.product(P, D, Q, period))

    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = np.inf
    best_order = None
    best_seasonal_order = None

    for pdq in pdq_combinations:
        for PDQ in PDQ_combinations:
            fold_errors = []
            # Walk-forward CV
            for train_index, val_index in tscv.split(train_values):
                train_fold = train_values[train_index]
                val_fold = train_values[val_index]
                try:
                    model = sm.tsa.statespace.SARIMAX(
                        train_fold,
                        order=pdq,
                        seasonal_order=PDQ,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )

                    results = model.fit(disp=False)
                    # Forecast for validation fold
                    start = len(train_fold)
                    end = len(train_fold) + len(val_fold) - 1

                    predictions = results.predict(start=start, end=end)
                    rmse = np.sqrt(mean_squared_error(val_fold, predictions))
                    fold_errors.append(rmse)

                except:
                    # Skip parameter sets that fail to converge
                    fold_errors.append(np.inf)

            avg_score = np.mean(fold_errors)

            if avg_score < best_score:
                best_score = avg_score
                best_order = pdq
                best_seasonal_order = PDQ

    return best_order, best_seasonal_order, best_score


# Getting optimal parameters for each time series    
wine_order, wine_seasonal_order, wine_aic = grid_search_sarima_cv(wine_classical_train, p=[0,1,2], d=[0], q=[0,1,2], P=[0,1,2], D=[1], Q=[0,1,2], period=[12])
beer_order, beer_seasonal_order, beer_aic = grid_search_sarima_cv(beer_classical_train, p=[0,1,2], d=[0], q=[0,1,2], P=[0,1,2], D=[1], Q=[0,1,2], period=[12])
liquor_order, liquor_seasonal_order, liquor_aic = grid_search_sarima_cv(liquor_classical_train, p=[0,1,2], d=[0], q=[0,1,2], P=[0,1,2], D=[1], Q=[0,1,2], period=[12])
nonalcoholic_order, nonalcoholic_seasonal_order, nonalcoholic_aic = grid_search_sarima_cv(nonalcoholic_classical_train, p=[0,1,2], d=[0], q=[0,1,2], P=[0,1,2], D=[0,1], Q=[0,1,2], period=[12])


# Best order, seasonal orders (wine, beer, liquor, nonalcoholic)
'''
(0, 0, 0) (0, 1, 2, 12)
(1, 0, 2) (0, 1, 1, 12)
(2, 0, 2) (2, 0, 2, 12)
(2, 0, 2) (0, 1, 1, 12)

IMPORTANT NOTE: SEASONAL DIFFERENCING SHOULD ONLY BE APPLIED TO THOSE WITH SEASONAL COMPONENTS
IT WAS NOT VISUALLY APPARENT THAT NONALCOHOLIC DRINKS HAVE A SEASONALITY TO THEM
OPTIMAL SEASONAL ORDER SUGGESTS IT SHOULD BE DIFFERENCED
ALSO LIQUOR HAS NO SEASONAL DIFFERENCING EVEN THOUGH IT HAD A STRONG SEASONAL COMPONENT
'''

'''
From ACF/PACF Plots:
Beer: (1,0,(0,1)) ((0,1),(0,1),(0,1),12)
Wine: (0,0,0) (1,1,(1,0),12)
Liquor: (0,0,0) (1,1,1,12)
Nonalcholic: ((0,1,2),0,(0,1,2)) (0,0,0,12)
'''

# Initializing Models
wine_sarima_model = sm.tsa.statespace.SARIMAX(wine_classical_train, order=wine_order, seasonal_order=wine_seasonal_order,
                                  enforce_stationarity=False, enforce_invertibility=False)
wine_sarima_results = wine_sarima_model.fit(method="Powell", disp=False)

beer_sarima_model = sm.tsa.statespace.SARIMAX(beer_classical_train, order=beer_order, seasonal_order=beer_seasonal_order,
                                  enforce_stationarity=False, enforce_invertibility=False)
beer_sarima_results = beer_sarima_model.fit(method="Powell", disp=False)

liquor_sarima_model = sm.tsa.statespace.SARIMAX(liquor_classical_train, order=liquor_order, seasonal_order=liquor_seasonal_order,
                                  enforce_stationarity=False, enforce_invertibility=False)
liquor_sarima_results = liquor_sarima_model.fit(method="Powell", disp=False)

nonalcoholic_sarima_model = sm.tsa.statespace.SARIMAX(nonalcoholic_classical_train, order=nonalcoholic_order, seasonal_order=nonalcoholic_seasonal_order,
                                  enforce_stationarity=False, enforce_invertibility=False)
nonalcoholic_sarima_results = nonalcoholic_sarima_model.fit(method="Powell", disp=False)



def get_model_fit_graph(classical_train_dataset, sarima_model_results):
    # Joining fitted values with actual training values
    demand_df = pd.DataFrame(classical_train_dataset)
    demand_cat_column = demand_df.columns[0]
    demand_df["fitted"] = sarima_model_results.fittedvalues.reindex(demand_df.index)
    demand_df.index = demand_df.index.to_timestamp()
    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(demand_df[demand_cat_column], label="Actual")
    plt.plot(demand_df["fitted"], label="Fitted", linestyle="--")
    plt.legend()
    plt.title(f"SARIMAX Fit vs Actual {demand_cat_column}")
    plt.show()
    
#get_model_fit_graph(nonalcoholic_classical_train, nonalcoholic_sarima_results)


def onestep_predict_sarima(sarima_model, train_set, test_set):
    '''
    Performs one-step prediction using a SARIMA model
    Returns a dataframe with a newly appended column for test set forecasts and the rmse
    '''
    # initializing a dataframe to later include predicted scores
    demand_df = pd.DataFrame(pd.concat([train_set, test_set]))
    # Used to enforce 1d
    historical_demand = list(train_set.values)
    test_set_vals = test_set.values
    # storing order and seasonal order
    order = sarima_model.order
    seasonal_order = sarima_model.seasonal_order
    # Prediction
    predictions = []
    
    for i in range(len(test_set_vals)):
        sarima_model = sm.tsa.statespace.SARIMAX(historical_demand, order=order, seasonal_order=seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False)
        results=sarima_model.fit(method="Powell", disp=False)
        
        # Perform one-step forecast
        pred = results.forecast(steps=1)[0]
        predictions.append(pred)
        # Update training set with new step value
        historical_demand.append(test_set_vals[i])
        
    # Compute evaluation metrics
    prediction_rmse = np.sqrt(mean_squared_error(test_set,predictions))
    # Adding preicted scores to demand dataframe
    pred_series = pd.Series(predictions, index=test_set.index)
    demand_df["predicted_sales"] = pred_series
    return demand_df, prediction_rmse
    

# Initializing SARIMA prediction dataframes and scores
wine_sarima_prediction_df, wine_sarima_rmse = onestep_predict_sarima(wine_sarima_model, wine_classical_train, wine_classical_test)
beer_sarima_prediction_df, beer_sarima_rmse = onestep_predict_sarima(beer_sarima_model, beer_classical_train, beer_classical_test)
liquor_sarima_prediction_df, liquor_sarima_rmse = onestep_predict_sarima(liquor_sarima_model, liquor_classical_train, liquor_classical_test)
nonalcoholic_sarima_prediction_df, nonalcoholic_sarima_rmse = onestep_predict_sarima(nonalcoholic_sarima_model,
                                                                                     nonalcoholic_classical_train,
                                                                                     nonalcoholic_classical_test)

# Appending scores to evaluation dict
store_evaluation_metrics(wine_sarima_prediction_df, "wine_sales", "Wine (Classical)", "SARIMA")
store_evaluation_metrics(beer_sarima_prediction_df, "beer_sales", "Beer (Classical)", "SARIMA")
store_evaluation_metrics(liquor_sarima_prediction_df, "liquor_sales", "Liquor (Classical)", "SARIMA")
store_evaluation_metrics(nonalcoholic_sarima_prediction_df, "nonalcoholic_sales", "Nonalcoholic (Classical)", "SARIMA")


''' RMSE SARIMA Scores
Wine: 4125.90665
Beer: 21156.396299
Liquor: 4121.220161
Nonalcoholic: 1311.630905
'''

def get_classical_prediction_linegraph(prediction_df, sales_col, save_image=False, model_type=None):
    '''
    Function that generates a line graph of the true and predicted time series
    Takes in a prediction dataframe (output from one-step forecast functions)
    '''
    df_copy = prediction_df.copy()
    df_copy.index = df_copy.index.to_timestamp()
    # Highlighting start of prediction sequence
    start_pred = df_copy["predicted_sales"].first_valid_index()
    plt.figure(figsize=(10,5))
    # Plotting true sales numbers
    plt.plot(df_copy.index, df_copy[sales_col], label="Actual "+sales_col)
    #Plotting predicted values
    plt.plot(df_copy.index, df_copy["predicted_sales"], label="Predicted "+sales_col, color="red")
    # Plotting prediction start
    plt.axvline(x=start_pred, linestyle=":", label="Forecast Start", linewidth=2, color="green")
    plt.legend()
    plt.title(f"Actual vs Predicted {sales_col} -- {model_type}")
    plt.xlabel("Time Period (m)")
    plt.ylabel("Sales")
    # Saving image if desired
    if save_image==True:
        plt.savefig(f"{sales_col}_{model_type}_linegraph")
        
    plt.show()


'''# Generating and saving prediction vs actual SARIMA line plots
get_classical_prediction_linegraph(wine_sarima_prediction_df, "wine_sales", save_image=True, model_type="SARIMA")
get_classical_prediction_linegraph(beer_sarima_prediction_df, "beer_sales", save_image=True, model_type="SARIMA")
get_classical_prediction_linegraph(liquor_sarima_prediction_df, "liquor_sales", save_image=True, model_type="SARIMA")
get_classical_prediction_linegraph(nonalcoholic_sarima_prediction_df, "nonalcoholic_sales", save_image=True, model_type="SARIMA")
'''


'''
Holt-Winters
'''

def fit_holt_winters_model(train_set, trend=["add", None], seasonal="add", seasonal_periods=12, method="Powell"):
    # .values to ensure train_set is 1d
    train_set = train_set.values
    # Iterate through trend values to optimize (although we didn't not a distinct trend in any ts)
    best_aic = np.inf
    best_model = None
    best_parameters = None
    
    for t in trend:
        model = ExponentialSmoothing(train_set, trend=t, seasonal=seasonal, seasonal_periods=seasonal_periods
                                     ).fit(optimized=True, method=method)
        if model.aic < best_aic:
            best_aic = model.aic
            best_model = model
            best_parameters = (f"Trend: {t}")
    
    return best_model, best_aic, best_parameters


'''# Dict to store all item category results
holt_winters_dict = {"Var":["AIC", "Params"],"Wine":[],"Beer":[],"Liquor":[],"NonAlcoholic":[]}
# Fitting wine model
wine_holtwinters_model, wine_holtwinters_aic, wine_holtwinters_trend = fit_holt_winters_model(wine_classical_train)
holt_winters_dict["Wine"].extend([wine_holtwinters_aic, wine_holtwinters_trend])
# Fitting beer model
beer_holtwinters_model, beer_holtwinters_aic, beer_holtwinters_trend = fit_holt_winters_model(beer_classical_train)
holt_winters_dict["Beer"].extend([beer_holtwinters_aic, beer_holtwinters_trend])
# Fitting liquor model
liquor_holtwinters_model, liquor_holtwinters_aic, liquor_holtwinters_trend = fit_holt_winters_model(liquor_classical_train)
holt_winters_dict["Liquor"].extend([liquor_holtwinters_aic, liquor_holtwinters_trend])
# Fitting nonalcoholic model
nonalcoholic_holtwinters_model, nonalcoholic_holtwinters_aic, nonalcoholic_holtwinters_trend = fit_holt_winters_model(nonalcoholic_classical_train)
holt_winters_dict["NonAlcoholic"].extend([nonalcoholic_holtwinters_aic, nonalcoholic_holtwinters_trend])
'''


def onestep_predict_holt_winters(train_set, test_set):
    '''
    Performs one-step forecasting using holt winters
    Returns prediction dataframe and RMSE
    '''
    # initializing a dataframe to later include predicted scores
    demand_df = pd.DataFrame(pd.concat([train_set, test_set]))
    # Used to enforce 1d
    historical_demand = list(train_set.values)
    test_set_vals = test_set.values
    # Prediction
    predictions = []
    
    for i in range(len(test_set_vals)):
        model = ExponentialSmoothing(historical_demand, trend=None, seasonal="add", seasonal_periods=12)
        fitted_model = model.fit(optimized=True, method="Powell")
        # Forecast
        pred = fitted_model.forecast(1)[0]
        # store pred
        predictions.append(pred)
        # expanding demand training
        historical_demand.append(test_set_vals[i])
        
    # Compute evaluation metrics
    prediction_rmse = np.sqrt(mean_squared_error(test_set,predictions))
    # Adding preicted scores to demand dataframe
    pred_series = pd.Series(predictions, index=test_set.index)
    demand_df["predicted_sales"] = pred_series
    return demand_df, prediction_rmse


# Initializing Holt-Winters prediction dataframes and scores
wine_holtwinters_prediction_df, wine_holtwinters_rmse = onestep_predict_holt_winters(wine_classical_train, wine_classical_test)
beer_holtwinters_prediction_df, beer_holtwinters_rmse = onestep_predict_holt_winters(beer_classical_train, beer_classical_test)
liquor_holtwinters_prediction_df, liquor_holtwinters_rmse = onestep_predict_holt_winters(liquor_classical_train, liquor_classical_test)
nonalcoholic_holtwinters_prediction_df, nonalcoholic_holtwinters_rmse = onestep_predict_holt_winters(nonalcoholic_classical_train,
                                                                                 nonalcoholic_classical_test)

# Storing evaluation metrics
store_evaluation_metrics(wine_holtwinters_prediction_df, "wine_sales", "Wine (Classical)", "Holt-Winters")
store_evaluation_metrics(beer_holtwinters_prediction_df, "beer_sales", "Beer (Classical)", "Holt-Winters")
store_evaluation_metrics(liquor_holtwinters_prediction_df, "liquor_sales", "Liquor (Classical)", "Holt-Winters")
store_evaluation_metrics(nonalcoholic_holtwinters_prediction_df, "nonalcoholic_sales", "Nonalcoholic (Classical)", "Holt-Winters")


'''# Generating and saving prediction vs actual Holt-Winters line plots
get_classical_prediction_linegraph(wine_holtwinters_prediction_df, "wine_sales", save_image=True, model_type="Holt-Winters")
get_classical_prediction_linegraph(beer_holtwinters_prediction_df, "beer_sales", save_image=True, model_type="Holt-Winters")
get_classical_prediction_linegraph(liquor_holtwinters_prediction_df, "liquor_sales", save_image=True, model_type="Holt-Winters")
get_classical_prediction_linegraph(nonalcoholic_holtwinters_prediction_df, "nonalcoholic_sales", save_image=True, model_type="Holt-Winters")
'''


def get_joined_classical_linegraph(sarima_df, holtwinters_df, sales_col, save_image=False):
    '''
    Linegraph function to plot both classical model forecast for a given product category
    '''

    sarima_new, holtwinters_new = sarima_df.copy(), holtwinters_df.copy()
    sarima_new.rename(columns={"predicted_sales": "SARIMA Forecast"}, inplace=True)
    holtwinters_new.rename(columns={"predicted_sales": "Holt-Winters Forecast"}, inplace=True)
    sarima_new["Holt-Winters Forecast"] = holtwinters_new["Holt-Winters Forecast"]
    sarima_new.index = sarima_new.index.to_timestamp()
    # Highlighting forecast start
    start_pred = sarima_new["SARIMA Forecast"].first_valid_index()
    plt.figure(figsize=(10,5))
    # Plotting true
    plt.plot(sarima_new.index, sarima_new[sales_col], label="Actual " +sales_col, color="black")
    # Plotting predicted
    plt.plot(sarima_new.index, sarima_new["SARIMA Forecast"], label="SARIMA Forecast", color="red", alpha=0.6)
    plt.plot(sarima_new.index, sarima_new["Holt-Winters Forecast"], label="Holt-Winters Forecast", color="c", alpha=0.8)
    # Plotting prediction start
    plt.axvline(x=start_pred, linestyle=":", label="Forecast Start", linewidth=2, color="green")
    plt.legend()
    plt.title(f"Actual vs Predicted {sales_col} -- Classical Models")
    plt.xlabel("Time Period (m)")
    plt.ylabel("Sales")
    # Saving image if desired
    if save_image==True:
        plt.savefig(f"{sales_col}_classical_model_forecast_linegraph")
        
    plt.show()
    

'''# Getting classical linegraphs
get_joined_classical_linegraph(wine_sarima_prediction_df, wine_holtwinters_prediction_df, "wine_sales", save_image=True)
get_joined_classical_linegraph(beer_sarima_prediction_df, beer_holtwinters_prediction_df, "beer_sales", save_image=True)
get_joined_classical_linegraph(liquor_sarima_prediction_df, liquor_holtwinters_prediction_df, "liquor_sales", save_image=True)
get_joined_classical_linegraph(nonalcoholic_sarima_prediction_df, nonalcoholic_holtwinters_prediction_df, "nonalcoholic_sales", save_image=True)
'''


'''
Machine Learning Modeling Section
'''

'''
ML Preprocessing
'''

# Initializing machine learning train:test splits
wine_ml_lag1_train, wine_ml_lag1_test = get_ml_partitions(31, wine_ml_ts_lag1)
wine_ml_lag3_train, wine_ml_lag3_test = get_ml_partitions(31, wine_ml_ts_lag3)
beer_ml_lag1_train, beer_ml_lag1_test = get_ml_partitions(31, beer_ml_ts_lag1)
beer_ml_lag3_train, beer_ml_lag3_test = get_ml_partitions(31, beer_ml_ts_lag3)
liquor_ml_lag1_train, liquor_ml_lag1_test = get_ml_partitions(31, liquor_ml_ts_lag1)
liquor_ml_lag3_train, liquor_ml_lag3_test = get_ml_partitions(31, liquor_ml_ts_lag3)
nonalcoholic_ml_lag1_train, nonalcoholic_ml_lag1_test = get_ml_partitions(31, nonalcoholic_ml_ts_lag1)
nonalcoholic_ml_lag3_train, nonalcoholic_ml_lag3_test = get_ml_partitions(31, nonalcoholic_ml_ts_lag3)

# Backwards Imputation initialization
wine_ml_lag12_train, wine_ml_lag12_test = get_ml_partitions(31, wine_ml_ts_lag12)
wine_ml_lag12_imp_train, wine_ml_lag12_imp_test = get_ml_partitions(91, wine_ml_ts_lag12_imputed)
beer_ml_lag12_imp_train, beer_ml_lag12_imp_test = get_ml_partitions(91, beer_ml_ts_lag12_imputed)
liquor_ml_lag12_imp_train, liquor_ml_lag12_imp_test = get_ml_partitions(91, liquor_ml_ts_lag12_imputed)
nonalcoholic_ml_lag12_imp_train, nonalcoholic_ml_lag12_imp_test = get_ml_partitions(91, nonalcoholic_ml_ts_lag12_imputed)

# Dropping the rows with NaNs as a reult from including lagged features
wine_ml_lag1_train, wine_ml_lag1_test = wine_ml_lag1_train.dropna(), wine_ml_lag1_test.dropna()
wine_ml_lag3_train, wine_ml_lag3_test = wine_ml_lag3_train.dropna(), wine_ml_lag3_test.dropna()
beer_ml_lag3_train, beer_ml_lag3_test = beer_ml_lag3_train.dropna(), beer_ml_lag3_test.dropna()
beer_ml_lag1_train, beer_ml_lag1_test = beer_ml_lag1_train.dropna(), beer_ml_lag1_test.dropna()
liquor_ml_lag1_train, liquor_ml_lag1_test = liquor_ml_lag1_train.dropna(), liquor_ml_lag1_test.dropna()
liquor_ml_lag3_train, liquor_ml_lag3_test = liquor_ml_lag3_train.dropna(), liquor_ml_lag3_test.dropna()
nonalcoholic_ml_lag3_train, nonalcoholic_ml_lag3_test = nonalcoholic_ml_lag3_train.dropna(), nonalcoholic_ml_lag3_test.dropna()
nonalcoholic_ml_lag1_train, nonalcoholic_ml_lag1_test = nonalcoholic_ml_lag1_train.dropna(), nonalcoholic_ml_lag1_test.dropna()

#Backwards Imputation NaN dropping
wine_ml_lag12_train, wine_ml_lag12_test = wine_ml_lag12_train.dropna(), wine_ml_lag12_test.dropna()
wine_ml_lag12_imp_train, wine_ml_lag12_imp_test = wine_ml_lag12_imp_train.dropna(), wine_ml_lag12_imp_test.dropna()
beer_ml_lag12_imp_train, beer_ml_lag12_imp_test = beer_ml_lag12_imp_train.dropna(), beer_ml_lag12_imp_test.dropna()
liquor_ml_lag12_imp_train, liquor_ml_lag12_imp_test = liquor_ml_lag12_imp_train.dropna(), liquor_ml_lag12_imp_test.dropna()
nonalcoholic_ml_lag12_imp_train, nonalcoholic_ml_lag12_imp_test = nonalcoholic_ml_lag12_imp_train.dropna(), nonalcoholic_ml_lag12_imp_test.dropna()


'''
XGBoost
'''

def get_train_test_splits(train, test):
    '''
    Returns train and test splits for ML datasets
    '''
    sales_col = [col for col in train.columns if col[-5:]=="sales"]
    X_train, y_train = train.drop(columns=sales_col[0]).reset_index(drop=True), train[sales_col[0]].reset_index(drop=True)
    X_test, y_test = test.drop(columns=sales_col[0]).reset_index(drop=True), test[sales_col[0]].reset_index(drop=True)
    
    return X_train, y_train, X_test, y_test


def xgboost_gridsearch(train, test, parameter_matrix, validation_splits=5, random_state=7, print_grid_search=False):
    '''
    Performs grid search for XGBoost models
    '''
    X_train, y_train, _, _ = get_train_test_splits(train, test)

    # Initialize model
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state)

    # Time series cross-validation
    time_series_val = TimeSeriesSplit(n_splits=validation_splits)

    # Grid search using RMSE
    xgb_grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=parameter_matrix,
        cv=time_series_val,
        scoring="neg_root_mean_squared_error",
        verbose=1,
        n_jobs=-1)

    # Fit
    xgb_grid_search.fit(X_train, y_train)

    if print_grid_search:
        print("Best Parameters:", xgb_grid_search.best_params_)
        print("Best CV RMSE:", -xgb_grid_search.best_score_)

    return xgb_grid_search.best_params_


# XGBoost parameter space for grid search
xgboost_parameter_dict = {"learning_rate": [0.01, 0.03, 0.05, 0.1],
                          "n_estimators": [100, 200, 300, 500, 750],
                          "max_depth": [3, 4, 5, 6, 7]}



# Performing grid search for XGBoost models
wine_lag1_xgb_params = xgboost_gridsearch(wine_ml_lag1_train, wine_ml_lag1_test, xgboost_parameter_dict, print_grid_search=False)
wine_lag3_xgb_params = xgboost_gridsearch(wine_ml_lag3_train, wine_ml_lag3_test, xgboost_parameter_dict, print_grid_search=False)
beer_lag1_xgb_params = xgboost_gridsearch(beer_ml_lag1_train, beer_ml_lag1_test, xgboost_parameter_dict, print_grid_search=False)
beer_lag3_xgb_params = xgboost_gridsearch(beer_ml_lag3_train, beer_ml_lag3_test, xgboost_parameter_dict, print_grid_search=False)
liquor_lag1_xgb_params = xgboost_gridsearch(liquor_ml_lag1_train, liquor_ml_lag1_test, xgboost_parameter_dict, print_grid_search=False)
liquor_lag3_xgb_params = xgboost_gridsearch(liquor_ml_lag3_train, liquor_ml_lag3_test, xgboost_parameter_dict, print_grid_search=False)
nonalcoholic_lag1_xgb_params = xgboost_gridsearch(nonalcoholic_ml_lag1_train, nonalcoholic_ml_lag1_test, xgboost_parameter_dict, print_grid_search=False)
nonalcoholic_lag3_xgb_params = xgboost_gridsearch(nonalcoholic_ml_lag3_train, nonalcoholic_ml_lag3_test, xgboost_parameter_dict, print_grid_search=False)

# Backwards Imputation xgb grid search
wine_lag12_imp_xgb_params = xgboost_gridsearch(wine_ml_lag12_imp_train, wine_ml_lag12_imp_test, xgboost_parameter_dict, print_grid_search=False)
beer_lag12_imp_xgb_params = xgboost_gridsearch(beer_ml_lag12_imp_train, beer_ml_lag12_imp_test, xgboost_parameter_dict, print_grid_search=False)
liquor_lag12_imp_xgb_params = xgboost_gridsearch(liquor_ml_lag12_imp_train, liquor_ml_lag12_imp_test, xgboost_parameter_dict, print_grid_search=False)
nonalcoholic_lag12_imp_xgb_params = xgboost_gridsearch(nonalcoholic_ml_lag12_imp_train, nonalcoholic_ml_lag12_imp_test, xgboost_parameter_dict, print_grid_search=False)


def plot_xgb_feature_importance(train, test, best_params, product_cat, random_state=7, max_num_features=10, save_image=False):
    """Gets feature importance of xgb model using gain"""

    # Get training data
    X_train, y_train, _, _ = get_train_test_splits(train, test)
    # Initialize model with best parameters
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state, **best_params)

    # Fit model
    model.fit(X_train, y_train)
    # Extract feature importance (gain)
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")

    # Convert to DataFrame
    importance_df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())}).sort_values(by="importance", ascending=False)
    # Keep top features
    importance_df = importance_df.head(max_num_features)

    # Plot
    plt.figure(figsize=(8,5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    
    plt.xlabel("Gain (Feature Importance)")
    plt.ylabel("Feature")
    plt.title(f"XGBoost Feature Importance (Gain) -- {product_cat}")
    
    plt.tight_layout()
    
    # Saving image if desired
    if save_image==True:
        plt.savefig(f"{product_cat}_feature_importance_xgb")
        
    plt.show()

    return importance_df


'''# Plotting XGB feature importance
plot_xgb_feature_importance(wine_ml_lag12_imp_train, wine_ml_lag12_imp_test, wine_lag12_imp_xgb_params, "Wine", save_image=True)
plot_xgb_feature_importance(beer_ml_lag12_imp_train, beer_ml_lag12_imp_test, beer_lag12_imp_xgb_params, "Beer", save_image=True)
plot_xgb_feature_importance(liquor_ml_lag12_imp_train, liquor_ml_lag12_imp_test, liquor_lag12_imp_xgb_params, "Liquor", save_image=True)
plot_xgb_feature_importance(nonalcoholic_ml_lag12_imp_train, nonalcoholic_ml_lag12_imp_test, nonalcoholic_lag12_imp_xgb_params, "Nonalcoholic", save_image=True)
'''


def onestep_predict_xgboost(train_set, test_set, params, sales_col):
    '''
    Performs one-step forecasting using XGB model
    Returns prediction dataframe
    '''
    base_df = train_set.copy()
    predictions = []
    predictor_vars = [col for col in train_set.columns if col != sales_col]
    learning_rate = params["learning_rate"]
    max_depth = params["max_depth"]
    n_estimators = params["n_estimators"]
    
    for i in range(len(test_set)):
        X_train = train_set[predictor_vars]
        y_train = train_set[sales_col]
        
        xgb_model = xgb.XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        xgb_model.fit(X_train, y_train)
        
        X_test = test_set.iloc[i:i+1][predictor_vars]
        
        pred = xgb_model.predict(X_test)[0]
        predictions.append(pred)
        
        base_df = pd.concat([base_df, test_set.iloc[i:i+1]])
        
    pred_series = pd.Series(predictions, index=test_set.index)
    test_df = test_set.copy()
    test_df["predicted_sales"] = pred_series
    pred_df = pd.concat([train_set, test_df])
    
    return pred_df


# Generating XGBoost prediction dataframes
wine1_xgb_prediction_df = onestep_predict_xgboost(wine_ml_lag1_train, wine_ml_lag1_test, wine_lag1_xgb_params, "wine_sales")
wine3_xgb_prediction_df = onestep_predict_xgboost(wine_ml_lag3_train, wine_ml_lag3_test, wine_lag3_xgb_params, "wine_sales")
beer1_xgb_prediction_df = onestep_predict_xgboost(beer_ml_lag1_train, beer_ml_lag1_test, beer_lag1_xgb_params, "beer_sales")
beer3_xgb_prediction_df = onestep_predict_xgboost(beer_ml_lag3_train, beer_ml_lag3_test, beer_lag3_xgb_params, "beer_sales")
liquor1_xgb_prediction_df = onestep_predict_xgboost(liquor_ml_lag1_train, liquor_ml_lag1_test, liquor_lag1_xgb_params, "liquor_sales")
liquor3_xgb_prediction_df = onestep_predict_xgboost(liquor_ml_lag3_train, liquor_ml_lag3_test, liquor_lag3_xgb_params, "liquor_sales")
nonalcoholic1_xgb_prediction_df = onestep_predict_xgboost(nonalcoholic_ml_lag1_train, nonalcoholic_ml_lag1_test, nonalcoholic_lag1_xgb_params, "nonalcoholic_sales")
nonalcoholic3_xgb_prediction_df = onestep_predict_xgboost(nonalcoholic_ml_lag3_train, nonalcoholic_ml_lag3_test, nonalcoholic_lag3_xgb_params, "nonalcoholic_sales")

# Backwards Imputation XGB prediction dataframes
wine12_imp_xgb_prediction_df = onestep_predict_xgboost(wine_ml_lag12_imp_train, wine_ml_lag12_imp_test, wine_lag12_imp_xgb_params, "wine_sales")
beer12_imp_xgb_prediction_df = onestep_predict_xgboost(beer_ml_lag12_imp_train, beer_ml_lag12_imp_test, beer_lag12_imp_xgb_params, "beer_sales")
liquor12_imp_xgb_prediction_df = onestep_predict_xgboost(liquor_ml_lag12_imp_train, liquor_ml_lag12_imp_test, liquor_lag12_imp_xgb_params, "liquor_sales")
nonalcoholic12_imp_xgb_prediction_df = onestep_predict_xgboost(nonalcoholic_ml_lag12_imp_train, nonalcoholic_ml_lag12_imp_test, 
                                                               nonalcoholic_lag12_imp_xgb_params, "nonalcoholic_sales")


# Appending XGBoost evaluation metrics to metric dict
store_evaluation_metrics(wine1_xgb_prediction_df, "wine_sales", "Wine (ML Lag1)", "XGBoost")
store_evaluation_metrics(wine3_xgb_prediction_df, "wine_sales", "Wine (ML Lag3)", "XGBoost")
store_evaluation_metrics(beer1_xgb_prediction_df, "beer_sales", "Beer (ML Lag1)", "XGBoost")
store_evaluation_metrics(beer3_xgb_prediction_df, "beer_sales", "Beer (ML Lag3)", "XGBoost")
store_evaluation_metrics(liquor1_xgb_prediction_df, "liquor_sales", "Liquor (ML Lag1)", "XGBoost")
store_evaluation_metrics(liquor3_xgb_prediction_df, "liquor_sales", "Liquor (ML Lag3)", "XGBoost")
store_evaluation_metrics(nonalcoholic1_xgb_prediction_df, "nonalcoholic_sales", "Nonalcoholic (ML Lag1)", "XGBoost")
store_evaluation_metrics(nonalcoholic3_xgb_prediction_df, "nonalcoholic_sales", "Nonalcoholic (ML Lag3)", "XGBoost")
store_evaluation_metrics(wine12_imp_xgb_prediction_df, "wine_sales", "Wine Imputed (ML Lag12)", "XGBoost")
store_evaluation_metrics(beer12_imp_xgb_prediction_df, "beer_sales", "Beer Imputed (ML Lag12)", "XGBoost")
store_evaluation_metrics(liquor12_imp_xgb_prediction_df, "liquor_sales", "Liquor Imputed (ML Lag12)", "XGBoost")
store_evaluation_metrics(nonalcoholic12_imp_xgb_prediction_df, "nonalcoholic_sales", "Nonalcoholic Imputed (ML Lag12)", "XGBoost")


def get_ml_prediction_linegraph(prediction_df, sales_col, save_image=False, model_type=None):
    '''
    Linegraph generating function specific to ML portion
    Plots true and forecasting time series
    '''
    df_new = prediction_df.copy()
    df_new.index = df_new.index.to_timestamp()
    # Highlighting start of prediction sequence
    start_pred = df_new["predicted_sales"].first_valid_index()
    plt.figure(figsize=(10,5))
    # Plotting true sales numbers
    plt.plot(df_new.index, df_new[sales_col], label="Actual "+sales_col)
    #Plotting predicted values
    plt.plot(df_new.index, df_new["predicted_sales"], label="Predicted "+sales_col, color="red")
    # Plotting prediction start
    plt.axvline(x=start_pred, linestyle=":", label="Forecast Start", linewidth=2, color="green")
    plt.legend()
    plt.title(f"Actual vs Predicted {sales_col} -- {model_type}")
    plt.xlabel("Time Period (m)")
    plt.ylabel("Sales")
    # Saving image if desired
    if save_image==True:
        plt.savefig(f"{sales_col}_{model_type}_linegraph")
        
    plt.show()
    
    
def get_joined_ml_prediction_linegraph(xgboost_df, tcn_df, sales_col, save_image=False):
    '''
    Linegraph function to plot both ml model forecast for a given product category
    '''
    xgboost_new, tcn_new = xgboost_df.copy(), tcn_df.copy()
    xgboost_new.rename(columns={"predicted_sales": "XGBoost Forecast"}, inplace=True)
    tcn_new.rename(columns={"predicted_sales": "TCN Forecast"}, inplace=True)
    xgboost_new["TCN Forecast"] = tcn_new["TCN Forecast"]
    xgboost_new.index = xgboost_new.index.to_timestamp()
    # Highlighting forecast start
    start_pred = xgboost_new["XGBoost Forecast"].first_valid_index()
    plt.figure(figsize=(10,5))
    # Plotting true
    plt.plot(xgboost_new.index, xgboost_new[sales_col], label="Actual " +sales_col, color="black")
    # Plotting predicted
    plt.plot(xgboost_new.index, xgboost_new["XGBoost Forecast"], label="XGBoost Forecast", color="red", alpha=0.6)
    plt.plot(xgboost_new.index, xgboost_new["TCN Forecast"], label="TCN Forecast", color="c", alpha=0.8)
    # Plotting prediction start
    plt.axvline(x=start_pred, linestyle=":", label="Forecast Start", linewidth=2, color="green")
    plt.legend()
    plt.title(f"Actual vs Predicted {sales_col} -- ML Models")
    plt.xlabel("Time Period (m)")
    plt.ylabel("Sales")
    # Saving image if desired
    if save_image==True:
        plt.savefig(f"{sales_col}_ml_model_forecast_linegraph")
        
    plt.show()
    

'''# Generating XGBoost prediction line graphs
get_ml_prediction_linegraph(wine1_xgb_prediction_df, "wine_sales", save_image=True, model_type="XGBoost Lag1")
get_ml_prediction_linegraph(wine3_xgb_prediction_df, "wine_sales", save_image=True, model_type="XGBoost Lag3")
get_ml_prediction_linegraph(beer1_xgb_prediction_df, "beer_sales", save_image=True, model_type="XGBoost Lag1")
get_ml_prediction_linegraph(beer3_xgb_prediction_df, "beer_sales", save_image=True, model_type="XGBoost Lag3")
get_ml_prediction_linegraph(liquor1_xgb_prediction_df, "liquor_sales", save_image=True, model_type="XGBoost Lag1")
get_ml_prediction_linegraph(liquor3_xgb_prediction_df, "liquor_sales", save_image=True, model_type="XGBoost Lag3")
get_ml_prediction_linegraph(nonalcoholic1_xgb_prediction_df, "nonalcoholic_sales", save_image=True, model_type="XGBoost Lag1")
get_ml_prediction_linegraph(nonalcoholic3_xgb_prediction_df, "nonalcoholic_sales", save_image=True, model_type="XGBoost Lag3")


# Backwards Imputed Linegraphs
get_ml_prediction_linegraph(wine12_imp_xgb_prediction_df, "wine_sales", save_image=True, model_type="XGBoost Lag12 Imputed")
get_ml_prediction_linegraph(beer12_imp_xgb_prediction_df, "beer_sales", save_image=True, model_type="XGBoost Lag12 Imputed")
get_ml_prediction_linegraph(liquor12_imp_xgb_prediction_df, "liquor_sales", save_image=True, model_type="XGBoost Lag12 Imputed")
get_ml_prediction_linegraph(nonalcoholic12_imp_xgb_prediction_df, "nonalcoholic_sales", save_image=True, model_type="XGBoost Lag12 Imputed")
'''


'''
Temporal Convolutional Networks
'''

# Need to restructure datasets as sequences

def generate_tcn_sequences(X, y, sequence_len):
    '''
    Generates sequences necessary for TCN
    '''
    # initialize two empty lists
    Xs, ys = [], []
    # iterate through the two dataframes to append lists
    for i in range(len(X) - sequence_len):
        Xs.append(X.iloc[i:i+sequence_len].values)
        ys.append(y.iloc[i+sequence_len])
    # Convert to numpy arrays    
    Xs_array, ys_array = np.array(Xs), np.array(ys)
    
    return Xs_array, ys_array


# Defining TCN class
# Used ChatGPT and notes from Deep Learning course for structure of class
class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        
        layers = []
        num_layers = len(num_channels)
        
        for i in range(num_layers):
            dilation = 2 ** i
            
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) * dilation,
                    dilation=dilation
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # → (batch, features, seq_len)
        x = self.network(x)
        x = x[:, :, -1]  # last timestep
        return self.fc(x)
    

# Hyperparameter search
# small seq_lengths due to size of dataset
parameter_space = {
    "num_channels": [[4, 8], [8, 8], [8,4], [16, 16], [32, 16], [16, 32], [32,32]],
    "kernel_size": [2,3,4,6,8],
    "dropout": [0.05,0.1,0.2,0.4],
    "lr": [0.005, 0.01, 0.02, 0.05, 0.1],
    "seq_length": [4, 6, 8, 12]  
}

# This function does include validation

def tcn_param_search_no_val(train_df, sales_col, param_space, n_trials=15):
    
    
    best_params = None
    best_score = float("inf")
    
    def sample_params():
        return {k: random.choice(v) for k, v in param_space.items()}
    
    for _ in range(n_trials):
        params = sample_params()
        # Split the predictors from target
        X = train_df.drop(columns=sales_col)
        y = train_df[sales_col]
        # Scale predictors
        scaler_X = StandardScaler()
        X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns, index=X.index)
        scaler_y = StandardScaler()
        y_scaled = pd.Series(
            scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten(),
            index=y.index
            )
        # Validation splits
        split_idx = len(train_df) - 5
        # Splitting
        X_tr, X_val = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_tr, y_val = y_scaled.iloc[:split_idx], y_scaled.iloc[split_idx:]
        # Skip when X_val length is too small
        if params["seq_length"] >= len(X_val):
            continue
        # Build sequences
        X_tr_seq, y_tr_seq = generate_tcn_sequences(X_tr, y_tr, params["seq_length"])
        X_val_seq, y_val_seq = generate_tcn_sequences(X_val, y_val, params["seq_length"])
        # Skip invalid configs
        if len(X_tr_seq) < 10 or len(X_val_seq) == 0:
            continue
        # Convert to tensors
        X_tr_tensor = torch.tensor(X_tr_seq, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_tr_seq, dtype=torch.float32).unsqueeze(1)
        
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1)
        
        # Build model
        model = TCN(
            input_size=X_tr_seq.shape[2],
            num_channels=params["num_channels"],
            kernel_size=params["kernel_size"],
            dropout=params["dropout"]
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        loss_fn = torch.nn.MSELoss()
        
        # Train
        best_val_loss = float('inf')
        patience, patience_counter = 5, 0
        
        for epoch in range(30):
            model.train()
            optimizer.zero_grad()
            
            preds = model(X_tr_tensor)
            loss = loss_fn(preds, y_tr_tensor)
            
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_tensor)
                val_loss = loss_fn(val_preds, y_val_tensor).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        if best_val_loss < best_score:
            best_score = best_val_loss
            best_params = params
    
    print("Best params:", best_params)
    print("Best validation loss:", best_score)
    
    return best_params




def tcn_one_step_forecast(train_df, test_df, sales_col, best_params):
    '''
    Performs one-step forecasting using TCN
    '''
    
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler

    # Split features/target
    X_train = train_df.drop(columns=sales_col)
    y_train = train_df[sales_col]
    
    X_test = test_df.drop(columns=sales_col)
    y_test = test_df[sales_col]
    
    # scale features
    scaler_X = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
    
    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test),columns=X_test.columns,index=X_test.index)
    
    # scale target
    scaler_y = StandardScaler()
    
    y_train_scaled = pd.Series(scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(),index=y_train.index)
    
    # Train final model on FULL training data
    X_seq, y_seq = generate_tcn_sequences(X_train_scaled, y_train_scaled, best_params["seq_length"])
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)
    
    model = TCN(input_size=X_seq.shape[2], num_channels=best_params["num_channels"],
                kernel_size=best_params["kernel_size"],dropout=best_params["dropout"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        
        loss.backward()
        optimizer.step()
    
    # Initialize rolling history
    history_X = X_train_scaled.copy()
    history_y = y_train_scaled.copy()
    
    predictions = []
    
    for t in range(len(X_test)):
        
        # Ensure enough history exists
        if len(history_X) < best_params["seq_length"]:
            predictions.append(np.nan)
        else:
            # Use last seq_length observations
            window_X = history_X.iloc[-best_params["seq_length"]:].values
            
            X_input = torch.tensor(
                window_X[np.newaxis, :, :],
                dtype=torch.float32)
            
            model.eval()
            with torch.no_grad():
                yhat = model(X_input).item()
            
            predictions.append(yhat)
        
        # Update history with actual observed test point
        history_X = pd.concat([history_X, X_test_scaled.iloc[t:t+1]])
        
        new_y = y_test.iloc[t:t+1]
        new_y_scaled = scaler_y.transform(new_y.values.reshape(-1, 1)).flatten()
        # appending true value to history for one-step forecasting
        history_y = pd.concat([
            history_y,
            pd.Series(new_y_scaled, index=new_y.index)
        ])
    
    # Inverse transform predictions
    pred_array = np.array(predictions).reshape(-1, 1)
    preds_unscaled = scaler_y.inverse_transform(pred_array).flatten()
    
    pred_series = pd.Series(preds_unscaled, index=test_df.index)
    
    return pred_series

def get_tcn_predictions(train_df, test_df, sales_col, param_space):
    '''
    Function that ties together all TCN function to generate forecasts
    '''
    # Parameter tuning
    best_model_params = tcn_param_search_no_val(train_df, sales_col=sales_col, param_space=param_space)
    # Final model training and evaluation
    predictions = tcn_one_step_forecast(train_df, test_df, sales_col, best_model_params)
    # joining dfs and the prediction series
    merged_df = pd.concat([train_df, test_df])
    merged_df["predicted_sales"] = predictions
    
    return merged_df


drop_cols = ['sales_lag_1', 'sales_lag_3', 'sales_lag_6',
       'sales_lag_12', 'rolling_mean_2', 'rolling_mean_3', 'rolling_mean_6',
       
       'alcohol_cpi_lag1', 'labor_force_participation_rate_lag1',
       'employment_population_ratio_lag1', 'labor_force_lag1',
       'unemployment_rate_lag1',
       'min_temp_f_lag1', 'max_temp_f_lag1',
       ]

# Dropping redundant cols to see if it helps
#wine_ml_lag12_imp_train = wine_ml_lag12_imp_train.drop(columns=drop_cols)
#wine_ml_lag12_imp_test = wine_ml_lag12_imp_test.drop(columns=drop_cols)

# Generating prediction dataframes
wine1_tcn_prediction_df = get_tcn_predictions(wine_ml_lag1_train, wine_ml_lag1_test, "wine_sales", parameter_space)
wine3_tcn_prediction_df = get_tcn_predictions(wine_ml_lag3_train, wine_ml_lag3_test, "wine_sales", parameter_space)
beer1_tcn_prediction_df = get_tcn_predictions(beer_ml_lag1_train, beer_ml_lag1_test, "beer_sales", parameter_space)
beer3_tcn_prediction_df = get_tcn_predictions(beer_ml_lag3_train, beer_ml_lag3_test, "beer_sales", parameter_space)
liquor1_tcn_prediction_df = get_tcn_predictions(liquor_ml_lag1_train, liquor_ml_lag1_test, "liquor_sales", parameter_space)
liquor3_tcn_prediction_df = get_tcn_predictions(liquor_ml_lag3_train, liquor_ml_lag3_test, "liquor_sales", parameter_space)
nonalcoholic1_tcn_prediction_df = get_tcn_predictions(nonalcoholic_ml_lag1_train, nonalcoholic_ml_lag1_test, "nonalcoholic_sales", parameter_space)
nonalcoholic3_tcn_prediction_df = get_tcn_predictions(nonalcoholic_ml_lag3_train, nonalcoholic_ml_lag3_test, "nonalcoholic_sales", parameter_space)


# Backwards Imputed Prediction Dataframes
wine12_imp_tcn_prediction_df = get_tcn_predictions(wine_ml_lag12_imp_train, wine_ml_lag12_imp_test, "wine_sales", parameter_space)
beer12_imp_tcn_prediction_df = get_tcn_predictions(beer_ml_lag12_imp_train, beer_ml_lag12_imp_test, "beer_sales", parameter_space)
liquor12_imp_tcn_prediction_df = get_tcn_predictions(liquor_ml_lag12_imp_train, liquor_ml_lag12_imp_test, "liquor_sales", parameter_space)
nonalcoholic12_imp_tcn_prediction_df = get_tcn_predictions(nonalcoholic_ml_lag12_imp_train, nonalcoholic_ml_lag12_imp_test, "nonalcoholic_sales", parameter_space)


# Appending TCN Evaluation metrics
store_evaluation_metrics(wine1_tcn_prediction_df, "wine_sales", "Wine (ML Lag1)", "TCN")
store_evaluation_metrics(wine3_tcn_prediction_df, "wine_sales", "Wine (ML Lag3)", "TCN")
store_evaluation_metrics(beer1_tcn_prediction_df, "beer_sales", "Beer (ML Lag1)", "TCN")
store_evaluation_metrics(beer3_tcn_prediction_df, "beer_sales", "Beer (ML Lag3)", "TCN")
store_evaluation_metrics(liquor1_tcn_prediction_df, "liquor_sales", "Liquor (ML Lag1)", "TCN")
store_evaluation_metrics(liquor3_tcn_prediction_df, "liquor_sales", "Liquor (ML Lag3)", "TCN")
store_evaluation_metrics(nonalcoholic1_tcn_prediction_df, "nonalcoholic_sales", "Nonalcoholic (ML Lag1)", "TCN")
store_evaluation_metrics(nonalcoholic3_tcn_prediction_df, "nonalcoholic_sales", "Nonalcoholic (ML Lag3)", "TCN")

# Backwards Imputed metrics
store_evaluation_metrics(wine12_imp_tcn_prediction_df, "wine_sales", "Wine Imputed (ML Lag12)", "TCN")
store_evaluation_metrics(beer12_imp_tcn_prediction_df, "beer_sales", "Beer Imputed (ML Lag12)", "TCN")
store_evaluation_metrics(liquor12_imp_tcn_prediction_df, "liquor_sales", "Liquor Imputed (ML Lag12)", "TCN")
store_evaluation_metrics(nonalcoholic12_imp_tcn_prediction_df, "nonalcoholic_sales", "Nonalcoholic Imputed (ML Lag12)", "TCN")

# Initiatlizing Evaluation DF
model_evaluation_metric_df = pd.DataFrame(model_evaluation_metrics)
model_evaluation_metric_df.to_excel("Model Evaluation Metrics.xlsx")


'''# Generating TCN linegraphs
get_ml_prediction_linegraph(wine1_tcn_prediction_df, "wine_sales", save_image=True, model_type="TCN Lag1")
get_ml_prediction_linegraph(wine3_tcn_prediction_df, "wine_sales", save_image=True, model_type="TCN Lag3")
get_ml_prediction_linegraph(beer1_tcn_prediction_df, "beer_sales", save_image=True, model_type="TCN Lag1")
get_ml_prediction_linegraph(beer3_tcn_prediction_df, "beer_sales", save_image=True, model_type="TCN Lag3")
get_ml_prediction_linegraph(liquor1_tcn_prediction_df, "liquor_sales", save_image=True, model_type="TCN Lag1")
get_ml_prediction_linegraph(liquor3_tcn_prediction_df, "liquor_sales", save_image=True, model_type="TCN Lag3")
get_ml_prediction_linegraph(nonalcoholic1_tcn_prediction_df, "nonalcoholic_sales", save_image=True, model_type="TCN Lag1")
get_ml_prediction_linegraph(nonalcoholic3_tcn_prediction_df, "nonalcoholic_sales", save_image=True, model_type="TCN Lag3")


# Backwards Imputed Linegraphs
get_ml_prediction_linegraph(wine12_imp_tcn_prediction_df, "wine_sales", save_image=True, model_type="TCN Lag12 Imputed")
get_ml_prediction_linegraph(beer12_imp_tcn_prediction_df, "beer_sales", save_image=True, model_type="TCN Lag12 Imputed")
get_ml_prediction_linegraph(liquor12_imp_tcn_prediction_df, "liquor_sales", save_image=True, model_type="TCN Lag12 Imputed")
get_ml_prediction_linegraph(nonalcoholic12_imp_tcn_prediction_df, "nonalcoholic_sales", save_image=True, model_type="TCN Lag12 Imputed")
'''

'''
Metrics and Evaluations

'''

'''# Getting the bias of errors
def plot_prediction_error_bias(df, true_sales_col, model_type, save_image=False):
    new_df = df[[true_sales_col, "predicted_sales"]]
    new_df.dropna(inplace=True)
    new_df.index = new_df.index.to_timestamp()
    new_df["prediction_error"] = new_df["predicted_sales"] - new_df[true_sales_col]
    
    bar_colors = ["green" if val >= 0 else "red" for val in new_df["prediction_error"]]
    
    plt.figure(figsize=(8,5))
    plt.bar(new_df.index, new_df["prediction_error"], width=25, color=bar_colors, alpha=0.6)
    
    plt.xlabel("Time Period")
    plt.ylabel("Prediction Error (Predicted - Actual)")
    plt.title(f"{model_type} Forcast Error Bias - {true_sales_col}")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_image==True:
        plt.savefig(f"{true_sales_col}_{model_type}_forecast_error_bias")
    
    plt.show()
'''

# Plotting joined ML linegraphs
get_joined_ml_prediction_linegraph(wine1_xgb_prediction_df, wine12_imp_tcn_prediction_df,
                                   "wine_sales", save_image=True)
get_joined_ml_prediction_linegraph(beer3_xgb_prediction_df, beer12_imp_tcn_prediction_df,
                                   "beer_sales", save_image=True)
get_joined_ml_prediction_linegraph(liquor3_xgb_prediction_df, liquor12_imp_tcn_prediction_df,
                                   "liquor_sales", save_image=True)
get_joined_ml_prediction_linegraph(nonalcoholic1_xgb_prediction_df, nonalcoholic12_imp_tcn_prediction_df,
                                   "nonalcoholic_sales", save_image=True)



def plot_prediction_error_bias(df, true_sales_col, model_type, save_image=False):
    '''
    generates plot of prediction error over monthly periods for bias analysis
    '''
    new_df = df[[true_sales_col, "predicted_sales"]].copy()
    new_df.dropna(inplace=True)
    new_df.index = new_df.index.to_timestamp()
    # calculate error
    new_df["prediction_error"] = new_df["predicted_sales"] - new_df[true_sales_col]
    
    bar_colors = ["green" if val >= 0 else "red" for val in new_df["prediction_error"]]
    
    # Compute symmetric y-axis 
    max_error = new_df["prediction_error"].max()
    min_error = new_df["prediction_error"].min()
    max_abs_error = max(abs(max_error), abs(min_error))
    
    plt.figure(figsize=(8,5))
    plt.bar(new_df.index, new_df["prediction_error"], width=25, color=bar_colors, alpha=0.6)
    
    # Apply symmetric limits
    plt.ylim(-max_abs_error, max_abs_error)    
    # add zero reference line
    plt.axhline(0, linewidth=1)
    
    plt.xlabel("Time Period")
    plt.ylabel("Prediction Error (Predicted - Actual)")
    plt.title(f"{model_type} Forecast Error Bias - {true_sales_col}")
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_image:
        plt.savefig(f"{true_sales_col}_{model_type}_forecast_error_bias")
    
    plt.show()

'''
# Generating Forecast Error Bias plots
# SARIMA forecast error bias figure plotting and saving
plot_prediction_error_bias(wine_sarima_prediction_df, "wine_sales", "SARIMA", save_image=True)
plot_prediction_error_bias(beer_sarima_prediction_df, "beer_sales", "SARIMA", save_image=True)
plot_prediction_error_bias(liquor_sarima_prediction_df, "liquor_sales", "SARIMA", save_image=True)
plot_prediction_error_bias(nonalcoholic_sarima_prediction_df, "nonalcoholic_sales", "SARIMA", save_image=True)

# Holt Winters forecast error bias figure plotting and saving
plot_prediction_error_bias(wine_holtwinters_prediction_df, "wine_sales", "Holt-Winters", save_image=True)
plot_prediction_error_bias(beer_holtwinters_prediction_df, "beer_sales", "Holt-Winters", save_image=True)
plot_prediction_error_bias(liquor_holtwinters_prediction_df, "liquor_sales", "Holt-Winters", save_image=True)
plot_prediction_error_bias(nonalcoholic_holtwinters_prediction_df, "nonalcoholic_sales", "Holt-Winters", save_image=True)

# XGBoost forecast error bias figure plotting and saving
plot_prediction_error_bias(wine3_xgb_prediction_df, "wine_sales", "XGBoost (Lag3)", save_image=True)
plot_prediction_error_bias(beer3_xgb_prediction_df, "beer_sales", "XGBoost (Lag3)", save_image=True)
plot_prediction_error_bias(liquor3_xgb_prediction_df, "liquor_sales", "XGBoost (Lag3)", save_image=True)
plot_prediction_error_bias(nonalcoholic3_xgb_prediction_df, "nonalcoholic_sales", "XGBoost (Lag3)", save_image=True)

plot_prediction_error_bias(wine12_imp_xgb_prediction_df, "wine_sales", "XGBoost (Lag12 Imputed)", save_image=True)
plot_prediction_error_bias(beer12_imp_xgb_prediction_df, "beer_sales", "XGBoost (Lag12 Imputed)", save_image=True)
plot_prediction_error_bias(liquor12_imp_xgb_prediction_df, "liquor_sales", "XGBoost (Lag12 Imputed)", save_image=True)
plot_prediction_error_bias(nonalcoholic12_imp_xgb_prediction_df, "nonalcoholic_sales", "XGBoost (Lag12 Imputed)", save_image=True)

# TCN forecast error bias figure plotting and saving

plot_prediction_error_bias(wine3_tcn_prediction_df, "wine_sales", "TCN (Lag3)", save_image=True)
plot_prediction_error_bias(beer3_tcn_prediction_df, "beer_sales", "TCN (Lag3)", save_image=True)
plot_prediction_error_bias(liquor3_tcn_prediction_df, "liquor_sales", "TCN (Lag3)", save_image=True)
plot_prediction_error_bias(nonalcoholic3_tcn_prediction_df, "nonalcoholic_sales", "TCN (Lag3)", save_image=True)

plot_prediction_error_bias(wine12_imp_tcn_prediction_df, "wine_sales", "XGBoost (Lag12 Imputed)", save_image=True)
plot_prediction_error_bias(beer12_imp_tcn_prediction_df, "beer_sales", "XGBoost (Lag12 Imputed)", save_image=True)
plot_prediction_error_bias(liquor12_imp_tcn_prediction_df, "liquor_sales", "XGBoost (Lag12 Imputed)", save_image=True)
plot_prediction_error_bias(nonalcoholic12_imp_tcn_prediction_df, "nonalcoholic_sales", "XGBoost (Lag12 Imputed)", save_image=True)
'''

mean_error_dict = {"Model Type": [], "Product Category": [], "Mean Error": [], "Fill Rate": [], "Avg Monthly Inventory": []}

def get_mean_error(df, model_type, sales_col):
    '''
    function that returns the mean error for bias analysis
    '''
    df_new = df.copy()
    df_new.dropna(inplace=True)
    # Calculating error
    df_new["error"] = df_new["predicted_sales"] - df_new[sales_col]
    mean_error = df_new["error"].sum() / len(df_new)
    mean_error_dict["Model Type"].append(model_type)
    mean_error_dict["Product Category"].append(sales_col)
    mean_error_dict["Mean Error"].append(mean_error)
    # Calculating fill rate
    df_new["fulfilled_units"] = df_new[[sales_col, "predicted_sales"]].min(axis=1)
    fill_rate = df_new["fulfilled_units"].sum() / df_new[sales_col].sum()
    mean_error_dict["Fill Rate"].append(fill_rate)
    # Calculating Avg Monthly Inventory
    avg_monthly_inventory = df_new["predicted_sales"].sum() / len(df_new)
    mean_error_dict["Avg Monthly Inventory"].append(avg_monthly_inventory)
    
    
get_mean_error(wine_sarima_prediction_df, "SARIMA", "wine_sales")
get_mean_error(beer_sarima_prediction_df, "SARIMA", "beer_sales")
get_mean_error(liquor_sarima_prediction_df, "SARIMA", "liquor_sales")
get_mean_error(nonalcoholic_sarima_prediction_df, "SARIMA", "nonalcoholic_sales")

get_mean_error(wine_holtwinters_prediction_df, "Holt-Winters", "wine_sales")
get_mean_error(beer_holtwinters_prediction_df, "Holt-Winters", "beer_sales")
get_mean_error(liquor_holtwinters_prediction_df, "Holt-Winters", "liquor_sales")
get_mean_error(nonalcoholic_holtwinters_prediction_df, "Holt-Winters", "nonalcoholic_sales")

get_mean_error(wine12_imp_tcn_prediction_df, "TCN", "wine_sales")
get_mean_error(beer12_imp_tcn_prediction_df, "TCN", "beer_sales")
get_mean_error(liquor12_imp_tcn_prediction_df, "TCN", "liquor_sales")
get_mean_error(nonalcoholic12_imp_tcn_prediction_df, "TCN", "nonalcoholic_sales")

get_mean_error(wine12_imp_xgb_prediction_df, "XGBoost", "wine_sales")
get_mean_error(beer12_imp_xgb_prediction_df, "XGBoost", "beer_sales")
get_mean_error(liquor12_imp_xgb_prediction_df, "XGBoost", "liquor_sales")
get_mean_error(nonalcoholic12_imp_xgb_prediction_df, "XGBoost", "nonalcoholic_sales")

mean_error_df = pd.DataFrame(mean_error_dict)
# Exporting to excel
'''
mean_error_df.to_excel("mean_error_dataframe.xlsx")
'''