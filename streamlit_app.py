# !pip install streamlit
import streamlit as st
import mlflow
import mlflow.spark
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.recommendation import ALSModel

# -----------------------------------------------------------
# 1) Spark + MLflow Setup
# -----------------------------------------------------------
spark = SparkSession.builder \
    .appName("StreamlitRecSystem") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

# Attempt to load the model
als_model = None
st.title("Music Recommendation Demo")

try:
    als_model = ALSModel.load("local_export/ALSModel_Standalone")
except Exception as e:
    st.error(f"❌ Failed to load model from MLflow: {e}")

# -----------------------------------------------------------
# 2) Recommendation UI
# -----------------------------------------------------------
st.header("Get Recommendations")

if als_model is None:
    st.error("❌ No valid ALSModel loaded. Cannot recommend songs.")
else:
    user_id_input = st.number_input("Enter a user_id for recommendations:", min_value=0, step=1, value=123)
    top_k = st.slider("Number of Recommendations (K)", min_value=1, max_value=20, value=5)

    if st.button("Recommend Songs"):
        # Create a Spark DataFrame for the user
        user_df = spark.createDataFrame([(user_id_input,)], ["new_userId"])

        # Generate recommendations
        try:
            recs = als_model.recommendForUserSubset(user_df, top_k)
            if recs.count() == 0:
                st.warning("No recommendations found for this user. Possibly user not in training data.")
            else:
                row = recs.collect()[0]
                song_recs = row["recommendations"]
                recommended_song_ids = [r["new_songId"] for r in song_recs]
                st.write(f"**Recommended songs for user {user_id_input}:**")
                st.json(recommended_song_ids)
        except Exception as e:
            st.error(f"❌ Internal error: {str(e)}")
