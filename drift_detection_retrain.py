import os
import json
import mlflow
import mlflow.spark
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from datetime import datetime
from evidently.report import Report
from evidently.metrics import *
from pyspark.ml.recommendation import ALS, ALSModel

# ✅ Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "drift_retrain.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ✅ Initialize Spark
spark = SparkSession.builder.appName("DriftDetectionRetrain").getOrCreate()

# ✅ Paths
REFERENCE_DATA_PATH = "data/processed/reference_data.parquet"
INCREMENTAL_DATA_DIR = "data/incremental"

# ✅ Load Reference Data (Historical Interactions)
def load_reference_data():
    """Loads historical reference data for drift detection"""
    if not os.path.exists(REFERENCE_DATA_PATH):
        logging.warning("⚠️ No reference data found! Drift detection may be inaccurate.")
        return None
    return spark.read.parquet(REFERENCE_DATA_PATH)

# ✅ Load Latest ALS Model from MLflow
def load_latest_model():
    """Loads the latest ALS model from MLflow"""
    model_uri = "models:/MusicRec_Best_Model/latest"
    return mlflow.spark.load_model(model_uri)

# ✅ Load New Data for Incremental Training
def load_new_data():
    """Load the most recent incremental data"""
    
    if not os.path.exists(INCREMENTAL_DATA_DIR) or not os.listdir(INCREMENTAL_DATA_DIR):
        logging.info("⚠️ No new data found for incremental training.")
        return None

    latest_date = sorted(os.listdir(INCREMENTAL_DATA_DIR))[-1]  # Get most recent date folder
    new_data_path = os.path.join(INCREMENTAL_DATA_DIR, latest_date, "new_interactions.parquet")

    if not os.path.exists(new_data_path):
        logging.info("⚠️ No new interactions found in latest batch.")
        return None

    logging.info(f"📥 Loading new data from: {new_data_path}")
    return spark.read.parquet(new_data_path)

# ✅ Detect Data Drift
def detect_drift(reference_data, new_data):
    """Compares reference data with new interactions for drift detection"""
    
    if reference_data is None or new_data is None:
        logging.warning("⚠️ Skipping drift detection due to missing data.")
        return False, {}

    # Convert to Pandas for EvidentlyAI
    ref_pd = reference_data.toPandas()
    new_pd = new_data.toPandas()

    # Create EvidentlyAI Report
    report = Report(metrics=[
        DataDriftPreset(),
        ColumnDriftMetric(column_name="plays"),
        ColumnDriftMetric(column_name="user_id"),
        ColumnDriftMetric(column_name="song_id"),
    ])
    
    report.run(reference_data=ref_pd, current_data=new_pd)
    drift_results = report.as_dict()

    # Extract Drift Scores
    drift_metrics = {
        "data_drift_score": drift_results["metrics"][0]["result"]["dataset_drift"],
        "plays_drift_score": drift_results["metrics"][1]["result"]["drift_score"],
        "user_drift_score": drift_results["metrics"][2]["result"]["drift_score"],
        "song_drift_score": drift_results["metrics"][3]["result"]["drift_score"],
    }

    logging.info(f"📊 Drift Scores: {drift_metrics}")
    mlflow.log_metrics(drift_metrics)

    # If dataset drift detected, retrain the model
    return drift_metrics["data_drift_score"] > 0.6, drift_metrics  # Threshold = 60% Drift

# ✅ Incrementally Update ALS Model
def update_model(existing_model, new_data):
    """Performs incremental update on the existing ALS model using new data"""

    # Convert data types
    new_data = new_data.withColumn("user_id", col("user_id").cast("integer"))
    new_data = new_data.withColumn("song_id", col("song_id").cast("integer"))
    new_data = new_data.withColumn("plays", col("plays").cast("double"))

    # Configure ALS with same hyperparameters as the existing model
    als = ALS(
        maxIter=1,  # Incremental update with 1 iteration
        rank=existing_model.rank,
        regParam=existing_model._java_obj.parent().getRegParam(),
        alpha=existing_model._java_obj.parent().getAlpha(),
        userCol="user_id",
        itemCol="song_id",
        ratingCol="plays",
        implicitPrefs=True,
        coldStartStrategy="drop"
    )

    logging.info("🔄 Performing Incremental Training on New Data...")
    updated_model = als.fit(new_data)

    return updated_model

# ✅ Save and Register Updated Model
def register_updated_model(model):
    """Save and register updated ALS model to MLflow"""
    with mlflow.start_run():
        mlflow.spark.log_model(model, "ALS_Best_Model")
        mlflow.register_model("runs:/latest/MusicRec_Best_Model", "MusicRec_Best_Model")
    
    logging.info("✅ Incrementally Updated Model Registered to MLflow.")

# ✅ Main Execution: Check for Drift & Retrain If Needed
def main():
    """Main function for detecting drift and performing incremental training"""
    
    logging.info("🚀 Starting Drift Detection & Incremental Training Pipeline...")

    # Load reference data
    reference_data = load_reference_data()

    # Load new data
    new_data = load_new_data()
    if new_data is None:
        logging.info("⏭ No new data available. Skipping retraining.")
        return

    # Detect drift
    drift_detected, drift_metrics = detect_drift(reference_data, new_data)

    if drift_detected:
        logging.warning("⚠️ Significant Data Drift Detected! Retraining Model...")
        
        # Load latest model
        existing_model = load_latest_model()

        # Perform incremental update
        updated_model = update_model(existing_model, new_data)

        # Register the updated model
        register_updated_model(updated_model)

        logging.info("🎯 Incremental Training Completed!")
    else:
        logging.info("✅ No significant drift detected. Skipping retraining.")

if __name__ == "__main__":
    main()
