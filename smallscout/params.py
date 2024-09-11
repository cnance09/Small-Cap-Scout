import os
import numpy as np

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")
GAR_REPO = os.environ.get("GAR_REPO")
DOCKER_IMAGE_NAME = os.environ.get("DOCKER_IMAGE_NAME")
SERVICE_URL = os.environ.get("SERVICE_URL")

# BigQuery
BQ_REGION = os.environ.get("BQ_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")

# Temporary Paths
BS_PATH = os.environ.get("BS_PATH")
IS_PATH = os.environ.get("IS_PATH")
CF_PATH = os.environ.get("CF_PATH")
SUB_PATH = os.environ.get("SUB_PATH")
FRED_PATH = os.environ.get("FRED_PATH")
MC_PATH = os.environ.get("MC_PATH")
STOCK_PATH = os.environ.get("STOCK_PATH")
QUERY_PATH= os.environ.get("QUERY_PATH")

DTYPES_RAW= {
    'name': 'object',
    'TICKER': 'object',
    'date': 'object',
    'quarter': 'object',
    'Assets': 'float64',
    'AssetsCurrent': 'float64',
    'Cash': 'float64',
    'AssetsNoncurrent': 'float64',
    'Liabilities': 'float64',
    'LiabilitiesCurrent': 'float64',
    'LiabilitiesNoncurrent': 'float64',
    'Equity': 'float64',
    'HolderEquity': 'float64',
    'RetainedEarnings': 'float64',
    'AdditionalPaidInCapital': 'float64',
    'TreasuryStockValue': 'float64',
    'TemporaryEquity': 'float64',
    'RedeemableEquity': 'float64',
    'LiabilitiesAndEquity': 'float64',
    'Revenues': 'float64',
    'CostOfRevenue': 'float64',
    'GrossProfit': 'float64',
    'OperatingExpenses': 'float64',
    'OperatingIncomeLoss': 'float64',
    'IncomeLossFromContinuingOperationsBeforeIncomeTaxExpenseBenefit': 'float64',
    'AllIncomeTaxExpenseBenefit': 'float64',
    'IncomeLossFromContinuingOperations': 'float64',
    'IncomeLossFromDiscontinuedOperationsNetOfTax': 'float64',
    'ProfitLoss': 'float64',
    'NetIncomeLossAttributableToNoncontrollingInterest': 'float64',
    'NetIncomeLoss': 'float64',
    'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations': 'float64',
    'NetCashProvidedByUsedInFinancingActivitiesContinuingOperations': 'float64',
    'NetCashProvidedByUsedInInvestingActivitiesContinuingOperations': 'float64',
    'NetCashProvidedByUsedInOperatingActivities': 'float64',
    'NetCashProvidedByUsedInFinancingActivities': 'float64',
    'NetCashProvidedByUsedInInvestingActivities': 'float64',
    'CashProvidedByUsedInOperatingActivitiesDiscontinuedOperations': 'float64',
    'CashProvidedByUsedInInvestingActivitiesDiscontinuedOperations': 'float64',
    'CashProvidedByUsedInFinancingActivitiesDiscontinuedOperations': 'float64',
    'EffectOfExchangeRateFinal': 'float64',
    'CashPeriodIncreaseDecreaseIncludingExRateEffectFinal': 'float64',
    'afs': 'object',
    'sic_2d': 'object',
    'GDP': 'float64',
    'interest_rate': 'float64',
    'unemployment_rate': 'float64',
    'median_cpi': 'float64',
    'market_cap': 'float64',
    'small_cap': 'Int64',
    'micro_cap': 'Int64',
    'qtr': 'object'
}
