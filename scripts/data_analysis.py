from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("MovieLensRecommendation").getOrCreate()

# Load Processed Data
processed_data_path = "s3://your-bucket/movielens/processed/data.parquet"
data_df = spark.read.parquet(processed_data_path)

data_df.createOrReplaceTempView("movie_data")

# Top 10 Rated Movies
top_movies = spark.sql("""
  SELECT title, AVG(rating) as avg_rating, COUNT(*) as num_ratings
  FROM movie_data
  GROUP BY title
  HAVING num_ratings > 1000
  ORDER BY avg_rating DESC
  LIMIT 10
""")
top_movies.show()

data_df.write.csv("s3://your-bucket/movielens/output/top_movies.csv", mode='overwrite')

# Top Genres by Rating
genre_ratings = spark.sql("""
  SELECT explode(split(genres, '\|')) as genre, AVG(rating) as avg_rating
  FROM movie_data
  GROUP BY genre
  ORDER BY avg_rating DESC
""")
genre_ratings.show()

genre_ratings.write.csv("s3://your-bucket/movielens/output/genre_ratings.csv", mode='overwrite')

# Rating Distribution Over Years
ratings_over_years = spark.sql("""
  SELECT year, AVG(rating) as avg_rating
  FROM movie_data
  WHERE year IS NOT NULL
  GROUP BY year
  ORDER BY year
""")
ratings_over_years.show()

ratings_over_years.write.csv("s3://your-bucket/movielens/output/ratings_over_years.csv", mode='overwrite')
