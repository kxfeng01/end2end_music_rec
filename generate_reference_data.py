import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
import datetime

# ✅ Initialize Spark
spark = SparkSession.builder.appName("GenerateReferenceData").getOrCreate()

# ✅ Paths
INTERACTIONS_PATH = "data/processed/interactions_full"
REFERENCE_PATH = "data/processed/reference_data.parquet"

# ✅ Load Full Interactions Data
print("🚀 Loading full interactions dataset...")
interactions = spark.read.parquet(INTERACTIONS_PATH)

# ✅ Choose Reference Data Strategy
USE_RECENT_DATA = True  # Set to False to use all historical data

if USE_RECENT_DATA:
    # 🔹 Use data from the last 30 days (assuming a timestamp column exists)
    print("📅 Filtering interactions from the last 30 days...")
    
    # Check if timestamp column exists
    if "timestamp" in interactions.columns:
        # Get 30 days ago date
        thirty_days_ago = datetime.datetime.now() - datetime.timedelta(days=30)
        
        # Filter for last 30 days
        reference_data = interactions.filter(col("timestamp") >= lit(thirty_days_ago.isoformat()))
    else:
        print("⚠️ No timestamp column found! Using random sample instead.")
        reference_data = interactions.sample(fraction=0.1, seed=42)

else:
    # 🔹 Use ALL historical interactions
    print("📜 Using full historical interactions as reference data...")
    reference_data = interactions

# ✅ Save Reference Data
print(f"💾 Saving reference data to {REFERENCE_PATH} ...")
reference_data.write.mode("overwrite").parquet(REFERENCE_PATH)

print("✅ Reference data generation complete! 🎯")
