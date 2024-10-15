from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark Session
spark = SparkSession.builder.appName("MovieLensRecommendation").getOrCreate()

# Load Processed Data
processed_data_path = "s3://your-bucket/movielens/processed/data.parquet"
data_df = spark.read.parquet(processed_data_path)

# Split Data into Training and Test Sets
training_df, test_df = data_df.randomSplit([0.8, 0.2])

# Build ALS Recommendation Model
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol='userIndex',
    itemCol='movieIndex',
    ratingCol='rating',
    coldStartStrategy='drop'
)

# Hyperparameter Tuning with Cross-Validation
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 20, 30]) \
    .addGrid(als.regParam, [0.05, 0.1, 0.15]) \
    .build()

evaluator = RegressionEvaluator(
    metricName='rmse',
    labelCol='rating',
    predictionCol='prediction'
)

cross_validator = CrossValidator(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3
)

cv_model = cross_validator.fit(training_df)

# Evaluate the Model
predictions = cv_model.transform(test_df)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Save the Model\model_output_path = "s3://your-bucket/movielens/model/"
cv_model.bestModel.save(model_output_path)
