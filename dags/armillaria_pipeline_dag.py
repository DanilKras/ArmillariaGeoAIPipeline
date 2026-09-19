import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_extraction import fetch_gbif_occurrences
from src.data_processing import process_and_thin_data
from src.feature_extraction import run_advanced_feature_extraction
from src.inference import generate_risk_map
from src.model_training import train_spatial_sdm
from src.negative_sampling import generate_ecological_pseudo_absences

default_args = {
    "owner": "danil",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
}


@dag(
    dag_id="armillaria_geoai_pipeline",
    default_args=default_args,
    description="End-to-End GeoAI pipeline for Armillaria ostoyae risk mapping",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["geoai", "species-distribution", "xgboost", "planetary-computer"],
)
def armillaria_geoai_dag():

    @task(task_id="extract_gbif_occurrences")
    def task_extract():
        fetch_gbif_occurrences()

    @task(task_id="spatial_rarefaction")
    def task_thinning():
        process_and_thin_data()

    @task(task_id="pseudo_absence_sampling")
    def task_sampling():
        generate_ecological_pseudo_absences()

    @task(task_id="cloud_feature_engineering")
    def task_features():
        run_advanced_feature_extraction()

    @task(task_id="train_spatial_xgboost")
    def task_train():
        train_spatial_sdm()

    @task(task_id="generate_spatial_inference")
    def task_infer():
        generate_risk_map()

    t1 = task_extract()
    t2 = task_thinning()
    t3 = task_sampling()
    t4 = task_features()
    t5 = task_train()
    t6 = task_infer()

    t1 >> t2 >> t3 >> t4 >> t5 >> t6


dag_instance = armillaria_geoai_dag()
