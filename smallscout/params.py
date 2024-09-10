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

# Temporary Paths
BS_PATH = os.environ.get("BS_PATH")
IS_PATH = os.environ.get("IS_PATH")
CF_PATH = os.environ.get("CF_PATH")
SUB_PATH = os.environ.get("SUB_PATH")
FRED_PATH = os.environ.get("FRED_PATH")
MC_PATH = os.environ.get("MC_PATH")
STOCK_PATH = os.environ.get("STOCK_PATH")
QUERY_PATH= os.environ.get("QUERY_PATH")
MODEL_PATH = os.environ.get("MODEL_PATH")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH")
