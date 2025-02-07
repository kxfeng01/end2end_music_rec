
import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

def load_data(base_dir):
    """
    Load raw data from the specified directory.
    Returns:
      - raw_plays_df: User-song interactions (train_triplets.txt)
      - songs2tracks_df: Song-to-track mappings
      - metadata_df: Song metadata
    """
    spark = SparkSession.builder.appName("ProcessData").getOrCreate()

    # Define file paths
    triplets_path = os.path.join(base_dir, "train_triplets.txt")
    songs2tracks_path = os.path.join(base_dir, "taste_profile_song_to_tracks.txt")
    metadata_path = os.path.join(base_dir, "msd_subset.csv")

    # Define schemas
    raw_plays_df = spark.read.csv(triplets_path, sep='\t', inferSchema=True, header=False) \
                            .toDF("userId", "songId", "Plays")

    songs2tracks_df = spark.read.csv(songs2tracks_path, sep='\t', inferSchema=True, header=False) \
                                .toDF("songId", "trackId")

    metadata_df = spark.read.csv(metadata_path, sep=',', inferSchema=True, header=False) \
                           .toDF("artist_familiarity", "artist_hotttness", "artist_id", 
                                 "artist_name", "artist_terms", "artist_terms_freq", "release_id", 
                                 "duration", "songId", "year")

    return raw_plays_df, songs2tracks_df, metadata_df

def process_data(raw_dir, processed_dir):
    """
    Process raw data into formatted datasets with contiguous user & item IDs.
    
    Outputs:
      - interactions_full: all interactions with new_userId & new_songId.
      - training, validation, test: data splits (60/20/20).
      - users: userId -> new_userId mapping.
      - items: songId -> new_songId mapping.
      - songs2tracks: song-to-track mappings with new_songId.
      - metadata: metadata with new_songId.
    """
    spark = SparkSession.builder.appName("ProcessData").getOrCreate()
    
    # Load raw datasets
    raw_plays_df, songs2tracks_df, metadata_df = load_data(raw_dir)

    # Generate contiguous user & item IDs using row_number()
    user_window = Window.orderBy("userId")
    item_window = Window.orderBy("songId")

    users = raw_plays_df.select("userId").distinct() \
                        .withColumn("new_userId", row_number().over(user_window) - 1)

    items = raw_plays_df.select("songId").distinct() \
                        .withColumn("new_songId", row_number().over(item_window) - 1)

    # Map original IDs to new integer-based IDs
    interactions = raw_plays_df.join(users, "userId") \
                               .join(items, "songId") \
                               .select("new_userId", "new_songId", "Plays") \
                               .cache()

    # **Split interactions into Training (60%), Validation (20%), Test (20%)**
    train, validation, test = interactions.randomSplit([0.6, 0.2, 0.2], seed=42)

    # Process songs-to-tracks & metadata (retain all mappings)
    processed_songs2tracks = songs2tracks_df.join(items, "songId", how="left")
    processed_metadata = metadata_df.join(items, "songId", how="left")

    # Save processed datasets in Parquet format
    interactions.write.mode("overwrite").parquet(f"{processed_dir}/interactions_full")
    train.write.mode("overwrite").parquet(f"{processed_dir}/training")
    validation.write.mode("overwrite").parquet(f"{processed_dir}/validation")
    test.write.mode("overwrite").parquet(f"{processed_dir}/test")
    users.write.mode("overwrite").parquet(f"{processed_dir}/users")
    items.write.mode("overwrite").parquet(f"{processed_dir}/items")
    processed_songs2tracks.write.mode("overwrite").parquet(f"{processed_dir}/songs2tracks")
    processed_metadata.write.mode("overwrite").parquet(f"{processed_dir}/metadata")

    print(f"✅ Data processing complete. Files saved to {processed_dir}")
    
    spark.stop()

if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    process_data(raw_dir, processed_dir)
