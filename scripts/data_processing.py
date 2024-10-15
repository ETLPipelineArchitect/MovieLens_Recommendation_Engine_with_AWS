import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer

# Initialize Spark Session
spark = SparkSession.builder.appName("MovieLensRecommendation").getOrCreate()

# Read Data from S3
target_s3_bucket = "s3://your-bucket/movielens/raw/ratings.csv"
target_s3_movies = "s3://your-bucket/movielens/raw/movies.csv"
ratings_df = spark.read.csv(target_s3_bucket, header=True, inferSchema=True)
movies_df = spark.read.csv(target_s3_movies, header=True, inferSchema=True)

# Handle Missing Values
ratings_df = ratings_df.dropna()
movies_df = movies_df.dropna()

# Extract Year from Title Using Regular Expressions
def extract_year(title):
    pattern = re.compile(r'\((\d{4})\)')
    match = pattern.search(title)
    if match:
        return int(match.group(1))
    else:
        return None

extract_year_udf = udf(extract_year, IntegerType())
movies_df = movies_df.withColumn('year', extract_year_udf(col('title')))

# Join Ratings and Movies DataFrames
data_df = ratings_df.join(movies_df, on='movieId', how='inner')

# Index userId and movieId
user_indexer = StringIndexer(inputCol='userId', outputCol='userIndex').fit(data_df)
movie_indexer = StringIndexer(inputCol='movieId', outputCol='movieIndex').fit(data_df)
data_df = user_indexer.transform(data_df)
data_df = movie_indexer.transform(data_df)

# Save Processed Data
processed_data_path = "s3://your-bucket/movielens/processed/data.parquet"
data_df.write.parquet(processed_data_path, mode='overwrite')
