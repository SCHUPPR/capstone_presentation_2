import pandas as pd

# Changing global settings to allow all rows and columns to display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# defining dataset paths for df creation
demand_file_path = "Warehouse_and_Retail_Sales.csv"
employment_file_path = "Maryland_Employment_Data.xlsx"
alc_cpi_file_path = "Maryland_Urban_AlcoholicBeverage_CPI.xlsx"
all_cpi_file_path = "Maryland_Urban_All_CPI.xlsx"
weather_file_path = "Montgom_County_Weather_Data.xlsx"

# initializing dataframes
demand_df_raw = pd.read_csv(demand_file_path)
employment_df_raw = pd.read_excel(employment_file_path)
alc_cpi_df_raw = pd.read_excel(alc_cpi_file_path)
all_cpi_df_raw = pd.read_excel(all_cpi_file_path)
weather_df_raw = pd.read_excel(weather_file_path)


# function to generate row and column count summary of dfs
def summarize_df_size(df_dict):
    '''
    Parameters
    ----------
    df_dict : dict
        dict with keys=df name and values=df.

    Returns
    -------
    df_summary_df : Pandas dataframe
        Pandas dataframe that summarize the row and column counts of all datasets used in the study
    '''
    df_summary_dict = {"Df Name":list(df_dict.keys()),
                       "Df Column Count":[df.shape[1] for df in df_dict.values()],
                       "Df Row Count":[len(df) for df in df_dict.values()]
                       }
    
    df_summary_df = pd.DataFrame(df_summary_dict)
    return df_summary_df


# function to generate column summary df for each df
def summarize_df_columns(df):
    '''
    Parameters
    ----------
    df : Pandas dataframe
        An individual dataframe representing one of the datasets in the study.

    Returns
    -------
    df_summary_df : Pandas dataframe
        A dataframe with characteristics of the input dataframe, such as dtypes, missing data, etc.
    '''
    df_summary_dict = {"Column Name":list(df.columns),
                       "Data Type":list(df.dtypes),
                       "Unique Values":[df[col].nunique(dropna=True) for col in list(df.columns)],
                       "Missing Values":[df[col].isna().sum() for col in list(df.columns)],
                       "Example Value":[df[col][0] for col in list(df.columns)]
                       }
    
    df_summary_df = pd.DataFrame(df_summary_dict)
    return df_summary_df


'''
General Dataset Structure and Characterization
'''

# Defining dictionary with dataframes for size summary function
dataframe_dict = {"Demand Dataset": demand_df_raw, "Employment Dataset": employment_df_raw,
         "Alcohol CPI Dataset": alc_cpi_df_raw, "All CPI Dataset": all_cpi_df_raw,
         "Weather Dataset": weather_df_raw}

# initializing size summary df
dataframe_size_summary_df = summarize_df_size(dataframe_dict)
# initializing column summary dict
dataframe_column_summary_df_list = {df[0]:summarize_df_columns(df[1]) for df in dataframe_dict.items()}
#print(dataframe_size_summary_df)


'''
Demand Dataset Characterization and Transformations
'''

# Summary of just Demand Dataset (9 columns and 307,645 initial rows)
#print(dataframe_size_summary_df[dataframe_size_summary_df["Df Name"]=="Demand Dataset"])
#print(dataframe_column_summary_df_list["Demand Dataset"])

# There are three columns with missing values: Supplier (167), Item Type (1), and Retail Sales (3)
# Supplier doesn't matter to the study and we'll be removing the column, additionally retial transfers
demand_df = demand_df_raw.drop(columns=["SUPPLIER","RETAIL TRANSFERS"])

# Item type does matter as we'll be aggregating individual SKUs up to their respective type.
# Examining row with the missing item type, followed by its removal (low volume doesn't justify imputation)
# print(demand_df[demand_df["ITEM TYPE"].isna()])
demand_df.dropna(subset=["ITEM TYPE"], inplace=True)
# Examing rows where retail sales is missing
# print(demand_df[demand_df["RETAIL SALES"].isna()])
# These rows showcase that the dataset contains coupon/rebate items, which will not be included in the study
demand_df.dropna(subset=["RETAIL SALES"], inplace=True)

# Now we'll be creating our time period column from YEAR and MONTH
# Using PeriodIndex to create a monthly period field for the time series alignment
demand_df["Time Period M"] = pd.PeriodIndex(year=demand_df["YEAR"], month=demand_df["MONTH"], freq="M")

# We will also be merging the retail and warehouse sales column and the project doesn't differentiate these
demand_df["Sales"] = demand_df["RETAIL SALES"] + demand_df["WAREHOUSE SALES"]
# Then we'll drop the two individual sales columns
demand_df.drop(columns=["RETAIL SALES", "WAREHOUSE SALES"], inplace=True)

# As this is a time series study, we'll now aggregate sales up to the item type level, and create one row for each time period
# This can be done easily using pivot_table()
demand_df_ts = demand_df.pivot_table(index="Time Period M", columns="ITEM TYPE",
                                     values="Sales", aggfunc="sum")
# print(demand_df_ts.head())
# Removing the REF, STR_SUPPLIES, and DUNNAGE columns as they're not the main product types
demand_df_ts.drop(columns=["REF", "STR_SUPPLIES", "DUNNAGE"], inplace=True)

# Checking if any product areas are missing sales data for a certain period (none found)
# print(summarize_df_columns(demand_df_ts))
# print(demand_df_ts)
# We notice that there are gaps within the time periods (notably in 2018 and 2020)
# data ranges from 2017-06 to 2020-09

# Renaming columns to follow standard conventions
demand_df_ts.rename(columns={"BEER": "beer_sales", "KEGS": "keg_sales", "LIQUOR": "liquor_sales",
                             "NON-ALCOHOL": "nonalcoholic_sales", "WINE": "wine_sales"}, inplace=True)


'''
Employment Dataset Characterization and Transformations
'''

# Summary of the employment dataset (8 columns and 131 initial rows)
#print(dataframe_size_summary_df[dataframe_size_summary_df["Df Name"]=="Employment Dataset"])
#print(dataframe_column_summary_df_list["Employment Dataset"])

employment_df = employment_df_raw.copy()

# Converting the Period column to a datetime in order to access the int representation of month
# Used ChatGPT solely to get the correct format string for the month formatting: "%b"
employment_df["Month"] = pd.to_datetime(employment_df["Period"], format="%b").dt.month
employment_df["Time Period M"] = pd.PeriodIndex(year=employment_df["Year"], month=employment_df["Month"], freq="M")
# The demand dateset is limited to 2017-06 and 2020-09 inclusive, and we won't be imputing before or after
# Remove anything before or after these period
employment_df = employment_df[(employment_df["Time Period M"] >= pd.Period("2017-06", freq="M")) &
                              (employment_df["Time Period M"] <= pd.Period("2020-09", freq="M"))]

# Replacing the index with Time Period M
employment_df_ts = employment_df.reset_index(drop=True).set_index("Time Period M")
# Dropping year and month columns
employment_df_ts.drop(columns=["Year", "Period", "Month"], inplace=True)

# Rechecking for missing values (none found)
#print(summarize_df_columns(employment_df_ts))

# Renaming columns to follow standard conventions
employment_df_ts.rename(columns={"labor force participation rate": "labor_force_participation_rate",
                                "employment-population ratio": "employment_population_ratio",
                                "labor force": "labor_force",
                                "employment": "employment",
                                "unemployment": "unemployment",
                                "unemployment rate": "unemployment_rate"}, inplace=True)


'''
All CPI Dataset Characterization and Transformations
'''

# Summary of the All CPI dataset ()
# print(dataframe_size_summary_df[dataframe_size_summary_df["Df Name"]=="All CPI Dataset"])
# print(dataframe_column_summary_df_list["All CPI Dataset"])

# Dataset is in wide format, will need to use pd.melt() to change to long
all_cpi_df = all_cpi_df_raw.melt(id_vars="Year", var_name="Month", value_name="all_cpi")
#print(all_cpi_df)
# There are annual, and half values that need to be removed
# Getting indices
drop_indices = all_cpi_df[(all_cpi_df["Month"]=="Annual")|(all_cpi_df["Month"]=="HALF1")|(all_cpi_df["Month"]=="HALF2")].index
all_cpi_df.drop(drop_indices, inplace=True)

# Converting Month column to datetime
all_cpi_df["Month Int"] = pd.to_datetime(all_cpi_df["Month"], format="%b").dt.month
all_cpi_df["Time Period M"] = pd.PeriodIndex(year=all_cpi_df["Year"], month=all_cpi_df["Month Int"], freq="M")
# The demand dateset is limited to 2017-06 and 2020-09 inclusive, and we won't be imputing before or after
# Remove anything before or after these period
all_cpi_df = all_cpi_df[(all_cpi_df["Time Period M"] >= pd.Period("2017-06", freq="M")) &
                              (all_cpi_df["Time Period M"] <= pd.Period("2020-09", freq="M"))]
# Replacing the index with Time Period M
all_cpi_df_ts = all_cpi_df.reset_index(drop=True).set_index("Time Period M")
# Dropping year and month columns
all_cpi_df_ts.drop(columns=["Year", "Month Int", "Month"], inplace=True)
#print(all_cpi_df_ts)


'''
Alcohol CPI Dataset Characterization and Transformations
'''

# Summary of the Alc CPI dataset ()
# print(dataframe_size_summary_df[dataframe_size_summary_df["Df Name"]=="Alcohol CPI Dataset"])
# print(dataframe_column_summary_df_list["Alcohol CPI Dataset"])

# Dataset is in wide format, will need to use pd.melt() to change to long
alc_cpi_df = alc_cpi_df_raw.melt(id_vars="Year", var_name="Month", value_name="alcohol_cpi")
#print(alc_cpi_df)
# There are annual, and half values that need to be removed
# Getting indices
drop_indices = alc_cpi_df[(alc_cpi_df["Month"]=="Annual")|(alc_cpi_df["Month"]=="HALF1")|(alc_cpi_df["Month"]=="HALF2")].index
alc_cpi_df.drop(drop_indices, inplace=True)

# Converting Month column to datetime
alc_cpi_df["Month Int"] = pd.to_datetime(alc_cpi_df["Month"], format="%b").dt.month
alc_cpi_df["Time Period M"] = pd.PeriodIndex(year=alc_cpi_df["Year"], month=alc_cpi_df["Month Int"], freq="M")
# The demand dateset is limited to 2017-06 and 2020-09 inclusive, and we won't be imputing before or after
# Remove anything before or after these period
alc_cpi_df = alc_cpi_df[(alc_cpi_df["Time Period M"] >= pd.Period("2017-06", freq="M")) &
                              (alc_cpi_df["Time Period M"] <= pd.Period("2020-09", freq="M"))]
# Replacing the index with Time Period M
alc_cpi_df_ts = alc_cpi_df.reset_index(drop=True).set_index("Time Period M")
# Dropping year and month columns
alc_cpi_df_ts.drop(columns=["Year", "Month Int", "Month"], inplace=True)
#print(alc_cpi_df_ts)


'''
Weather Dataset Characterization and Transformations
'''

# Summary of the weather dataset ()
#print(dataframe_size_summary_df[dataframe_size_summary_df["Df Name"]=="Weather Dataset"])
#print(dataframe_column_summary_df_list["Weather Dataset"])

weather_df = weather_df_raw.copy()
# Renaming columns
weather_df.rename(columns={"Avg_Temp_f": "avg_temp_f", "Min_Temp_F": "min_temp_f",
                           "Max_Temp_F": "max_temp_f", "Precipitation (in.)": "precipitation_inches"}, inplace=True)

# Converting Month column to int
weather_df["Month Int"] = pd.to_datetime(weather_df["Month"], format="%B").dt.month
weather_df["Time Period M"] = pd.PeriodIndex(year=weather_df["Year"], month=weather_df["Month Int"], freq="M")
# The demand dateset is limited to 2017-06 and 2020-09 inclusive, and we won't be imputing before or after
# Remove anything before or after these period
weather_df = weather_df[(weather_df["Time Period M"] >= pd.Period("2017-06", freq="M")) &
                              (weather_df["Time Period M"] <= pd.Period("2020-09", freq="M"))]
# Replacing the index with Time Period M
weather_df_ts = weather_df.reset_index(drop=True).set_index("Time Period M")
# Dropping year and month columns
weather_df_ts.drop(columns=["Year", "Month Int", "Month", "Month_Year"], inplace=True)
#print(weather_df_ts)


'''
Merging the dataframes for the ML models
'''

merged_ml_ts = pd.concat([demand_df_ts,all_cpi_df_ts,alc_cpi_df_ts,employment_df_ts,
                          weather_df_ts], axis=1)
# Sorting chronologically by index
merged_ml_ts.sort_index(inplace=True)
# Exporting to an Excel file
#merged_ml_ts.to_excel("merged_ts_dataset.xlsx")


'''
Dataset Partitioning
PARTITIONS WILL BE CREATED AFTER MISSING DATA IS IMPUTED
MISSING DATA WILL BE IMPUTED IN EDA DUE TO DEPENDENCE ON SEASONALITY
'''

# Demand partitions (Classical models)
def get_classical_partitions(train_size, df):
    '''
    Parameters
    ----------
    train_size : Number of time periods to serve as the training period.
    df : Pandas dataframe, classical dataset (demand only)

    Returns
    -------
    train_classical_split : training partition of demand dataframe.
    test_classical_split : testing partition of demand dataframe.
    '''
    train_classical_split = df[:train_size].copy()
    test_classical_split = df[train_size:].copy()
    return train_classical_split, test_classical_split


# Merged partitions (ML models)
def get_ml_partitions(train_size, df):
    '''
    Parameters
    ----------
    train_size : Number of time periods to serve as the training period.
    df : Pandas dataframe, ML dataset (demand + supporting datasets "merged...")

    Returns
    -------
    train_ml_split : training partition of the merged dataframe.
    test_ml_split : testing partition of the merged dataframe.

    '''
    train_ml_split = merged_ml_ts[:train_size].copy()
    test_ml_split = merged_ml_ts[train_size:].copy()
    return train_ml_split, test_ml_split