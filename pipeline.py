import os
import time
import logging
import subprocess
import mlflow
from pyspark.sql import SparkSession
from monitoring.drift_detection_retrain import main as run_drift_retrain

# ✅ Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ✅ Initialize Spark
spark = SparkSession.builder.appName("MLOpsPipeline").getOrCreate()

# ✅ Paths
INCREMENTAL_DATA_DIR = "data/incremental"

# ✅ Start Kafka Consumer (Background Process)
def start_kafka_consumer():
    logging.info("🎧 Starting Kafka Consumer to ingest new data...")
    
    consumer_script = "kafka/consumer.py"
    
    subprocess.Popen(
        ["nohup", "python", consumer_script, "&"],
        stdout=open(LOG_FILE, "a"),
        stderr=open(LOG_FILE, "a"),
        preexec_fn=os.setpgrp  # Run in background
    )

# ✅ Check if New Data Exists
def new_data_available():
    """Checks if new data exists in data/incremental"""
    if not os.path.exists(INCREMENTAL_DATA_DIR) or not os.listdir(INCREMENTAL_DATA_DIR):
        logging.info("⏭ No new data found. Skipping drift detection & retraining.")
        return False
    return True

# ✅ Run Drift Detection & Retraining
def run_monitoring_and_training():
    """Runs the drift detection and retraining module"""
    logging.info("🔍 Checking for drift & retraining if needed...")
    run_drift_retrain()  # Calls drift detection & incremental training

# ✅ Clean Up Old Data
def cleanup_old_data():
    """Deletes old incremental data to optimize storage"""
    logging.info("🧹 Cleaning up old incremental data...")

    for folder in os.listdir(INCREMENTAL_DATA_DIR):
        folder_path = os.path.join(INCREMENTAL_DATA_DIR, folder)
        try:
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
                logging.info(f"🗑 Deleted old data: {folder}")
        except Exception as e:
            logging.error(f"❌ Failed to delete {folder}: {e}")

# ✅ Main Pipeline Loop
def main():
    logging.info("🚀 Starting Full MLOps Pipeline...")

    # 1️⃣ Start Kafka Consumer to collect new data
    start_kafka_consumer()

    while True:
        # 2️⃣ Wait for new data
        time.sleep(30)  # Poll every 30 sec
        if not new_data_available():
            continue

        # 3️⃣ Run drift detection & retraining
        run_monitoring_and_training()

        # 4️⃣ Clean up old incremental data
        cleanup_old_data()

        logging.info("⏳ Sleeping for 24 hours before the next pipeline run...")
        time.sleep(86400)  # Sleep for 24h before next run

if __name__ == "__main__":
    main()
