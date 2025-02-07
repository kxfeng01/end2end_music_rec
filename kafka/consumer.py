
import json
import os
import time
import logging
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from confluent_kafka import Consumer, KafkaError, KafkaException
import configparser

# ✅ Setup Logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "consumer.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ✅ Load Kafka Configuration from `client.properties`
def load_kafka_config(config_path="client.properties"):
    abs_path = os.path.abspath(config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"ERROR: {abs_path} not found!")

    config = configparser.ConfigParser()
    config.read(abs_path)

    section = "default"
    if section not in config:
        raise ValueError(f"ERROR: Missing section [{section}] in {config_path}")

    return {
        "bootstrap.servers": config.get(section, "bootstrap.servers"),
        "security.protocol": config.get(section, "security.protocol"),
        "sasl.mechanisms": config.get(section, "sasl.mechanisms"),
        "sasl.username": config.get(section, "sasl.username"),
        "sasl.password": config.get(section, "sasl.password"),
        "group.id": "music_rec_consumer",
        "auto.offset.reset": "earliest",  # ✅ Start from latest messages
    }

# ✅ Initialize Spark
spark = SparkSession.builder.appName("KafkaConsumer") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# ✅ Define Kafka Consumer
def create_consumer():
    kafka_config = load_kafka_config()
    return Consumer(kafka_config)

# ✅ Define Schema for Incoming Data

schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("song_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("plays", IntegerType(), True)  # ✅ CORRECT COLUMN NAME
])


# ✅ Function to Process Kafka Messages and Write Daily Parquet Files
def process_messages(messages, output_base_path="data/incremental"):
    if not messages:
        logging.info("No messages to process.")
        return

    parsed_data = []
    for msg in messages:
        try:
            record = json.loads(msg.value().decode("utf-8"))
            logging.info(f"Raw message: {record}")  # Log the raw message

            if record.get("event_type") == "play":
                logging.info(f"Processing 'play' event: {record}")  # Log the play event
                record["timestamp"] = datetime.fromisoformat(record["timestamp"])
                parsed_data.append(record)
            else:
                logging.info(f"Skipping non-'play' event: {record}")  # Log skipped events

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"❌ JSON decode error: {e}")
            logging.error(f"❌ Faulty message: {msg.value().decode('utf-8')}")  # Log the problematic message

    if not parsed_data:
        logging.info("No 'play' events to process.")
        return

    # ✅ Convert to Spark DataFrame
    df = spark.createDataFrame(parsed_data, schema=schema)

    # ✅ Convert timestamp field to Spark TimestampType
    df = df.withColumn("timestamp", to_timestamp(col("timestamp")))

    # ✅ Get today's date for folder structure
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_base_path, today_date)
    os.makedirs(output_path, exist_ok=True)

    # ✅ Append new data to daily Parquet folder
    parquet_path = os.path.join(output_path, "new_interactions.parquet")
    df.write.mode("append").parquet(parquet_path)

    logging.info(f"✅ {len(parsed_data)} new interactions saved to {parquet_path}")
    print(f"✅ {len(parsed_data)} new interactions saved to {parquet_path}")

# ✅ Kafka Consumer Loop (Batch Processing Every 30 seconds)
def consume_data():
    consumer = create_consumer()
    consumer.subscribe(["daily_interactions"])
    
    logging.info("🎧 Kafka Consumer started. Listening for new messages...")
    print("🎧 Kafka Consumer started. Listening for new messages...")

    try:
        while True:
            msg_batch = []
            batch_start_time = time.time()

            logging.info("🔄 Polling Kafka for new messages...")
            print("🔄 Polling Kafka for new messages...")

            while time.time() - batch_start_time < 30:  # Poll for 30 seconds per batch
                print("🔍 Attempting to poll Kafka...")
                msg = consumer.poll(timeout=1.0)

                if msg is None:
                    print("⏳ No messages received in this poll...")
                    continue

                if msg.error():
                    print(f"❌ Kafka error: {msg.error()}")
                    logging.error(f"❌ Kafka error: {msg.error()}")
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                # ✅ If we get here, we received a message
                message_value = msg.value().decode("utf-8")
                print(f"📩 Received message: {message_value}")
                logging.info(f"📩 Received message: {message_value}")
                
                msg_batch.append(msg)

                # ✅ Process in smaller chunks for debugging
                if len(msg_batch) >= 5:
                    process_messages(msg_batch)
                    msg_batch = []

            if msg_batch:
                process_messages(msg_batch)

            logging.info("⏳ Sleeping for 30 seconds before checking again...")
            print("⏳ Sleeping for 30 seconds before checking again...")
            time.sleep(30)
    except KeyboardInterrupt:
        logging.info("🛑 Consumer stopped manually.")
        print("🛑 Consumer stopped manually.")
    finally:
        consumer.close()

# ✅ Run Consumer
if __name__ == "__main__":
    consume_data()
