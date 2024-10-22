# Import Packages
import os
import numpy as np
import pandas as pd
from smallscout.params import *
from google.cloud import bigquery

def quarters(month):
    if month <= 3:
        return 'Q1'
    if month <= 6:
        return 'Q2'
    if month <= 9:
        return 'Q3'
    return 'Q4'

def load_financials():
    #Load Financial Data
    balance_sheet = pd.read_csv(BS_PATH, index_col=0)
    income_statement = pd.read_csv(IS_PATH, index_col=0)
    cash_flow_statement = pd.read_csv(CF_PATH, index_col=0)

    sub = pd.read_csv(SUB_PATH, index_col=0)
    sub = sub[['adsh', 'sic', 'afs']]

    #Clean up financial statements
    balance_sheet.drop(['coreg', 'report', 'form', 'fye', 'qtrs'], axis=1, inplace=True)
    income_statement.drop(['coreg', 'report', 'form', 'fye', 'qtrs'], axis=1, inplace=True)
    cash_flow_statement.drop(['coreg', 'report', 'form', 'fye', 'qtrs'], axis=1, inplace=True)

    balance_sheet.drop_duplicates(inplace=True)
    income_statement.drop_duplicates(inplace=True)
    cash_flow_statement.drop_duplicates(inplace=True)

    balance_sheet['date'] = pd.to_datetime(balance_sheet['date'])
    income_statement['date'] = pd.to_datetime(income_statement['date'])
    cash_flow_statement['date'] = pd.to_datetime(cash_flow_statement['date'])

    # Remove 2009 and earlier
    balance_sheet = balance_sheet[(balance_sheet.date.dt.year>=2010)&(balance_sheet.date.dt.year<2025)]
    income_statement = income_statement[(income_statement.date.dt.year>=2010)&(income_statement.date.dt.year<2025)]
    cash_flow_statement = cash_flow_statement[(cash_flow_statement.date.dt.year>=2010)&(cash_flow_statement.date.dt.year<2025)]


    # Difference quarterly data for income statements
    income_statement_adj = income_statement.sort_values(['cik', 'date']).reset_index(drop=True)
    for col in income_statement_adj.columns[8:-1]:
        income_statement_adj[col] = (income_statement_adj.groupby(['cik','fy'])[col].diff().fillna(income_statement_adj[col]))


    # Merge Financial Data and Create New Variables
    df_financials = balance_sheet.merge(income_statement_adj, how='inner', on=['adsh', 'cik', 'name', 'fy', 'fp', 'date', 'filed', 'ddate'],
                                   suffixes=('_bs', '_is'))
    df_financials = df_financials.merge(cash_flow_statement, how='inner', on=['adsh', 'cik', 'name', 'fy', 'fp', 'date', 'filed', 'ddate'],
                                    suffixes=('', '_cf'))
    df_financials = df_financials.merge(sub, how='left', on='adsh') #273,309 observations

    df_financials['sic_2d'] = df_financials.sic.apply(lambda x: str(x)[:2]) #70 unique industry IDs
    df_financials['quarter'] = df_financials.apply(lambda x: str(x.date.year)+'-'+quarters(x.date.month), axis=1)
    df_financials['year'] = df_financials.date.dt.year

    # Keep only full year observations
    df_financials = df_financials[df_financials.n_year==4]
    drop_cols = ['adsh', 'name', 'fy', 'fp', 'filed', 'ddate', 'sic']
    df_financials.drop(columns=drop_cols, inplace=True)

    df_financials = df_financials.sort_values(['cik', 'date']).drop_duplicates().reset_index(drop=True)

    return df_financials

def get_unique_cik():
    df = load_market_cap()
    final_cik_list = df[['CIK', 'TICKER']].drop_duplicates().reset_index(drop=True)
    return final_cik_list

def load_economic_variables():
    fred = pd.read_csv(FRED_PATH)
    fred['date'] = pd.to_datetime(fred.Date) + pd.tseries.offsets.MonthEnd(0)
    fred.drop(['Date'], axis=1, inplace=True)
    fred = fred[fred.date.dt.year>=2010]
    return fred

def load_market_cap():
    mkt_cap = pd.read_excel(MC_PATH, usecols=[0:60])
    mkt_cap = mkt_cap.melt(id_vars=['CIK', 'TICKER'], value_vars=mkt_cap.columns[2:], var_name='date', value_name='market_cap')
    mkt_cap['date'] = pd.to_datetime(mkt_cap.date)
    mkt_cap = mkt_cap.dropna().sort_values(['CIK', 'date']).reset_index(drop=True)

    idx = mkt_cap.index
    mkt_cap.sort_values(['CIK', 'TICKER', 'date'], inplace=True)

    ## Create target variables of interest
    # One quarter ahead
    mkt_cap['mc_qtr_growth'] = mkt_cap['market_cap'].diff()
    mkt_cap['mc_qtr_growth_pct'] = mkt_cap['mc_qtr_growth'] / mkt_cap['market_cap'].shift(1)

    mask = mkt_cap.TICKER != mkt_cap.TICKER.shift(1)
    mkt_cap.loc[mkt_cap['mc_qtr_growth'][mask].index, 'mc_qtr_growth'] = np.nan
    mkt_cap.loc[mkt_cap['mc_qtr_growth_pct'][mask].index, 'mc_qtr_growth_pct'] = np.nan

    # One year ahead
    mkt_cap['mc_yr_growth'] = mkt_cap['market_cap'].diff(4)
    mkt_cap['mc_yr_growth_pct'] = mkt_cap['mc_yr_growth'] / mkt_cap['market_cap'].shift(4)

    mask = mkt_cap.TICKER != mkt_cap.TICKER.shift(4)
    mkt_cap.loc[mkt_cap['mc_yr_growth'][mask].index, 'mc_yr_growth'] = np.nan
    mkt_cap.loc[mkt_cap['mc_yr_growth_pct'][mask].index, 'mc_yr_growth_pct'] = np.nan

    # Two years ahead
    mkt_cap['mc_2yr_growth'] = mkt_cap['market_cap'].diff(8)
    mkt_cap['mc_2yr_growth_pct'] = mkt_cap['mc_2yr_growth'] / mkt_cap['market_cap'].shift(8)

    mask = mkt_cap.TICKER != mkt_cap.TICKER.shift(8)
    mkt_cap.loc[mkt_cap['mc_2yr_growth'][mask].index, 'mc_2yr_growth'] = np.nan
    mkt_cap.loc[mkt_cap['mc_2yr_growth_pct'][mask].index, 'mc_2yr_growth_pct'] = np.nan

    # Small Cap Flag
    mkt_cap['small_cap'] = mkt_cap.market_cap.apply(lambda x: 1 if x <= 2*10**3 else 0) # less than USD 2_000_000_000
    mkt_cap['micro_cap'] = mkt_cap.market_cap.apply(lambda x: 1 if x < 3*10**2 else 0) # less than USD 300_000_000
    mkt_cap['quarter'] = mkt_cap.date.apply(lambda x: str(x.year)+'-'+quarters(x.month))

    mkt_cap.reindex(idx)

    return mkt_cap

def load_stock_data():
    stocks = pd.read_csv(STOCK_PATH)
    stocks['date'] = pd.to_datetime(stocks.Month, format='%Y-%m') + pd.tseries.offsets.MonthEnd(0)
    stocks['quarter'] = stocks.apply(lambda x: str(x.date.year)+'-'+ quarters(x.date.month), axis=1)
    stocks.drop(columns=['Month', 'Monthly_Volume_Avg'], inplace=True)

    # Transform monthly data to quarterly
    stocks_qtr = stocks.groupby(['Ticker', 'quarter'], as_index=False).agg({'Monthly_Avg_Close': 'mean',
                                                                           'Monthly_Volume_Total': 'mean',
                                                                           'Monthly_Volatility': 'mean',
                                                                           'date': 'last'})
    stocks_qtr.drop('date', axis=1, inplace=True)
    return stocks_qtr

def merge_sets(stocks=False):
    df_financials = load_financials()
    fred = load_economic_variables()
    mkt_cap = load_market_cap()
    if stocks == True:
        stocks = load_stock_data()
        level2 = mkt_cap.merge(stocks, left_on=['TICKER', 'quarter'], right_on=['Ticker', 'quarter'], how='inner')
        level2.drop(columns=['date'], inplace=True)

    level1 = df_financials.merge(fred, how='left', on='date')
    level1.reset_index(drop=True, inplace=True)

    if stocks == True:
        merged_df = level1.merge(level2, left_on=['cik', 'quarter'], right_on=['CIK', 'quarter'], how='inner')
    else:
        merged_df = level1.merge(mkt_cap, left_on=['cik', 'quarter'], right_on=['CIK', 'quarter'], how='inner')

    merged_df['n_year'] = merged_df.groupby(['cik', 'year'])['year'].transform('count')
    merged_df = merged_df[merged_df.n_year==4]
    merged_df.drop('n_year', axis=1, inplace=True)

    merged_df['n_cik'] = merged_df.groupby(['cik'])['cik'].transform('count')
    merged_df = merged_df[merged_df.n_cik>=12]
    merged_df.drop('n_cik', axis=1, inplace=True)

    merged_df = merged_df.drop_duplicates().reset_index(drop=True)

    return merged_df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    #data.columns = ['pickup_datetime'] + [f"_{i}" for i in range(1,len(data.columns)-1)] + ['fare_amount']

    # Load data onto full_table_name
    client = bigquery.Client(gcp_project)
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else


    print(f"✅ Data saved to bigquery, with shape {data.shape}")

#load_data_to_bq(data=merge_sets(), gcp_project=GCP_PROJECT, bq_dataset=BQ_DATASET, table=BQ_REGION, truncate=False)
