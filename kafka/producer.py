import json
import random
import datetime
import os
import logging
import time
from confluent_kafka import Producer, KafkaException, KafkaError
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list
from configparser import ConfigParser

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "producer.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def load_kafka_config(config_path):
    """Load Kafka configurations from client.properties."""
    abs_path = os.path.abspath(config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"ERROR: {abs_path} not found!")

    config = ConfigParser()
    config.read(abs_path)

    if "default" not in config:
        raise ValueError(f"ERROR: Missing section [default] in {config_path}")

    kafka_config = {
        "bootstrap.servers": config.get("default", "bootstrap.servers"),
        "security.protocol": config.get("default", "security.protocol"),
        "sasl.mechanisms": config.get("default", "sasl.mechanisms"),
        "sasl.username": config.get("default", "sasl.username"),
        "sasl.password": config.get("default", "sasl.password"),
        "client.id": config.get("default", "client.id"),
    }
    return kafka_config

def load_user_song_data(spark):
    """Load users and songs from processed dataset."""
    users_df = spark.read.parquet("data/processed/users/")
    items_df = spark.read.parquet("data/processed/items/")
    interactions_df = spark.read.parquet("data/processed/interactions_full/")

    users = users_df.select("new_userId").rdd.flatMap(lambda x: x).collect()
    songs = items_df.select("new_songId").rdd.flatMap(lambda x: x).collect()

    user_history = (
        interactions_df.groupBy("new_userId")
        .agg(collect_list("new_songId").alias("played_songs"))
        .rdd.map(lambda row: (row["new_userId"], set(row["played_songs"])))
        .collectAsMap()
    )

    return users, songs, user_history

def generate_simulated_data(users, songs, user_history, num_events=500, old_song_ratio=0.8):
    """Generate simulated play events with 80-20 old-new song distribution."""
    interactions = []
    for _ in range(num_events):
        user_id = random.choice(users)
        played_songs = user_history.get(user_id, set())

        if random.random() < old_song_ratio and played_songs:
            song_id = random.choice(list(played_songs))
        else:
            unplayed_songs = list(set(songs) - played_songs)
            song_id = random.choice(unplayed_songs) if unplayed_songs else random.choice(songs)

        timestamp = datetime.datetime.now().isoformat()

        interaction = {
            "user_id": user_id,
            "song_id": song_id,
            "timestamp": timestamp,
            "event_type": "play",
            "plays": random.randint(1, 5),
        }

        interactions.append(interaction)
        logging.info(json.dumps(interaction))  # Log to file

    return interactions

def delivery_report(err, msg):
    """Delivery report for Kafka producer."""
    if err:
        logging.error(f"Message delivery failed: {err}")
    else:
        logging.info(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def main():
    spark = SparkSession.builder.appName("KafkaProducer") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()

    # Load Kafka config
    kafka_config = load_kafka_config("client.properties")
    print("✅ Successfully loaded Kafka config.")

    # Load user and song data
    users, songs, user_history = load_user_song_data(spark)

    # Configure Kafka producer
    producer = Producer(kafka_config)

    while True:
        batch = generate_simulated_data(users, songs, user_history)
        for msg in batch:
            producer.produce(
                topic="daily_interactions",
                key=str(msg["user_id"]).encode("utf-8"),
                value=json.dumps(msg).encode("utf-8"),
                callback=delivery_report
            )

        producer.flush()
        print(f"✅ Sent {len(batch)} interactions to Kafka.")
        time.sleep(86400)  # Simulate daily data generation

if __name__ == "__main__":
    main()
