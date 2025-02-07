import os
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, expr
from pyspark.sql.types import DoubleType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml.recommendation import ALSModel
from pyspark.ml import PipelineModel

def load_data(processed_dir):
    """
    Load processed training and validation data.
    """
    spark = SparkSession.builder.appName("GridSearchALS") \
            .config("spark.driver.memory", "12g") \
            .config("spark.executor.memory", "12g") \
            .config("spark.sql.shuffle.partitions", "2000")  \
            .getOrCreate()
    
    training_df = spark.read.parquet(os.path.join(processed_dir, "training"))
    validation_df = spark.read.parquet(os.path.join(processed_dir, "validation"))
    
    return spark, training_df, validation_df

def train_als(training_df, rank, regParam, alpha, seed=42):
    """
    Train an ALS model with specified hyperparameters.
    """
    als = ALS(
        maxIter=5,
        regParam=regParam,
        rank=rank,
        alpha=alpha,
        seed=seed,
        userCol="new_userId",
        itemCol="new_songId",
        ratingCol="Plays",
        implicitPrefs=True,
        coldStartStrategy="drop"
    )
    
    return als.fit(training_df)

def evaluate_model(model, validation_df, k=10):
    """
    Evaluate the trained ALS model using MAP and Precision@K.
    """
    users = validation_df.select("new_userId").distinct()
    preds = model.recommendForUserSubset(users, k)
    
    # Prepare ground truth
    ground_truth = validation_df.groupBy("new_userId") \
        .agg(collect_list("new_songId").alias("true_items")) \
        .withColumn("true_items", expr("transform(true_items, x -> cast(x as double))"))
    
    # Join predictions with ground truth
    results = preds.join(ground_truth, "new_userId") \
        .withColumn("predicted_items", expr("transform(recommendations, x -> cast(x.new_songId as double))"))

    # Ranking evaluation
    map_eval = RankingEvaluator(metricName="meanAveragePrecision",
                                predictionCol="predicted_items",
                                labelCol="true_items")
    precision_eval = RankingEvaluator(metricName="precisionAtK",
                                      predictionCol="predicted_items",
                                      labelCol="true_items")

    return {
        "MAP": map_eval.evaluate(results),
        "Precision_at_K": precision_eval.evaluate(results)
    }

if __name__ == "__main__":
    # 1) Initialize Spark and load data
    processed_dir = "data/processed"
    spark, training_df, validation_df = load_data(processed_dir)

    # 2) Set MLflow experiment
    mlflow.set_experiment("ALS_Implicit_Training")

    # Define hyperparameter grid
    ranks = [32] 
    regParams = [0.01, 0.05]
    alphas = [10, 40, 100]

    best_map = float('-inf')
    best_model = None
    best_params = {}

    # 3) Start grid search
    for rank in ranks:
        for regParam in regParams:
            for alpha in alphas:
                with mlflow.start_run():
                    print(f"🔍 Training ALS: rank={rank}, regParam={regParam}, alpha={alpha}")
                    
                    # Train model
                    model = train_als(training_df, rank, regParam, alpha)

                    # Evaluate model
                    metrics = evaluate_model(model, validation_df, k=10)
                    map_score = metrics["MAP"]
                    precision_score = metrics["Precision_at_K"]

                    # Log hyperparameters and metrics
                    mlflow.log_params({"rank": rank, "regParam": regParam, "alpha": alpha})
                    mlflow.log_metrics(metrics)

                    print(f"✅ rank={rank}, regParam={regParam}, alpha={alpha} → MAP={map_score:.5f}, Precision@10={precision_score:.5f}")

                    # Track best model
                    if map_score > best_map:
                        best_map = map_score
                        best_model = model
                        best_params = {"rank": rank, "regParam": regParam, "alpha": alpha}

    # 4) Register the best model in MLflow
    if best_model:
        print(f"\n🏆 Best Model Found: Rank={best_params['rank']}, RegParam={best_params['regParam']}, Alpha={best_params['alpha']} → MAP={best_map:.5f}")

        with mlflow.start_run():
            # Log best model
            mlflow.spark.log_model(best_model, "ALS_Best_Model")
            mlflow.log_dict(best_params, "best_model_params.json")

            # Register in Model Registry
            model_uri = mlflow.get_artifact_uri("ALS_Best_Model")
            registered_model = mlflow.register_model(model_uri, "MusicRec_Best_Model")  
            print("🎯 Model Registered as 'MusicRec_Best_Model' in MLflow")

            # 5) Export the best model locally
            local_export_dir = "local_export"
            if not os.path.exists(local_export_dir):
                os.mkdir(local_export_dir)

            print(f"⏬ Downloading model artifacts to '{local_export_dir}/ALS_Best_Model' ...")
            downloaded_dir = download_artifacts(
                artifact_uri=model_uri,
                dst_path=local_export_dir
            )
            print(f"✅ Exported to '{downloaded_dir}'")

            # 6) Extract ALS model from PipelineModel (if applicable) and save it separately
            print("🔍 Checking if ALSModel needs extraction...")
            loaded_obj = mlflow.spark.load_model(model_uri)
            als_model_path = os.path.join(local_export_dir, "ALSModel_Standalone")

            if isinstance(loaded_obj, ALSModel):
                als_model = loaded_obj
                print("✅ ALSModel found and extracted!")
            elif isinstance(loaded_obj, PipelineModel):
                print("🔍 Found a PipelineModel. Attempting to extract ALSModel stage...")
                for stage in loaded_obj.stages:
                    if isinstance(stage, ALSModel):
                        als_model = stage
                        print("✅ ALSModel extracted from pipeline!")
                        break
                else:
                    print("❌ No ALSModel found in the pipeline!")
                    als_model = None
            else:
                print(f"❌ Loaded object is neither PipelineModel nor ALSModel: {type(loaded_obj)}")
                als_model = None

            # Save extracted ALS model separately
            if als_model:
                als_model.write().overwrite().save(als_model_path)
                print(f"✅ ALSModel saved separately at '{als_model_path}'")
            else:
                print("⚠️ No valid ALSModel to save.")

    else:
        print("⚠️ No best model found! Possibly no training or evaluation?")

    print("\n🚀 Grid Search Complete!")
    spark.stop()

