import os
import mlflow
import mlflow.spark
import mlflow.pyfunc
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.recommendation import ALSModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ✅ Initialize Spark
spark = SparkSession.builder \
    .appName("RecommendationService") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# ✅ FastAPI App
app = FastAPI(title="🎵 Music Recommendation API")

# Load Model
als_model = ALSModel.load("local_export/ALSModel_Standalone")

# ✅ Define Request Body
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10  # Default to 10 recommendations

# ✅ Recommendation Endpoint
@app.post("/recommend")
def recommend_songs(request: RecommendationRequest):
    if als_model is None:
        raise HTTPException(status_code=500, detail="❌ Model not available")

    # Convert user_id into Spark DataFrame
    user_df = spark.createDataFrame([(request.user_id,)], ["new_userId"])

    # Generate Top-K Recommendations
    try:
        recommendations = als_model.recommendForUserSubset(user_df, request.top_k)
        # If user doesn't exist, we might get an empty result
        if not recommendations.count():
            raise IndexError("No recommendations for this user")
        rec_list = recommendations.collect()[0]["recommendations"]
        song_ids = [rec["new_songId"] for rec in rec_list]

        return {"user_id": request.user_id, "recommended_songs": song_ids}
    
    except IndexError:
        raise HTTPException(status_code=404, detail="❌ User ID not found in training data or no recommendations available.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Internal Error: {str(e)}")

# ✅ Health Check Endpoint
@app.get("/")
def root():
    return {"message": "🎵 Recommendation API is running!"}

# ✅ Run API with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
