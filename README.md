# Building a Recommendation Engine using MovieLens Dataset with AWS and Apache Spark

## **Project Overview**

**Objective:** Develop an end-to-end recommendation engine that processes the MovieLens dataset to provide movie recommendations through collaborative filtering techniques. The project involves data ingestion, processing, model building, evaluation, and visualization of results.

**Technologies Used:**

- **AWS Services:** S3
- **Programming Languages:** Python
- **Big Data Technologies:** Apache Spark
- **Machine Learning Frameworks:** PyTorch, Spark MLlib
- **Visualization:** Matplotlib, Seaborn, Jupyter Notebooks

---

## **Project Architecture**

1. **Data Ingestion:**
   - Ingest the MovieLens dataset from S3, including user ratings and movie information.

2. **Data Processing:**
   - Clean and preprocess the data using PySpark, handling missing values and extracting relevant features.

3. **Model Building:**
   - Utilize the Alternating Least Squares (ALS) algorithm from Spark MLlib to build a recommendation model based on user interactions.

4. **Model Evaluation:**
   - Evaluate the model performance using RMSE and perform hyperparameter tuning using cross-validation.

5. **Data Analysis:**
   - Analyze the processed data to extract insights, such as top-rated movies and rating distributions over the years.

6. **Visualization:**
   - Generate visualizations to represent the results of data analyses and model performance metrics.

---

## **Step-by-Step Implementation Guide**

### **1. Setting Up AWS Resources**

- **Create an S3 Bucket:**
  - Store raw and processed datasets along with model outputs.

### **2. Data Processing with PySpark**

#### **a. Script to Process MovieLens Data**

- **Read Data from S3:**

  ```python
  from pyspark.sql import SparkSession
  
  # Initialize Spark Session
  spark = SparkSession.builder.appName("MovieLensRecommendation").getOrCreate()
  
  # Read movie ratings and movies data from S3
  data_df = spark.read.csv("s3://your-bucket/movielens/raw/ratings.csv", header=True, inferSchema=True)
  movies_df = spark.read.csv("s3://your-bucket/movielens/raw/movies.csv", header=True, inferSchema=True)
  ```

- **Data Cleaning and Preparation:**

  ```python
  # Handle missing values
  data_df = data_df.dropna()
  movies_df = movies_df.dropna()
  ```

- **Save Processed Data to S3:**

  ```python
  data_df.write.parquet("s3://your-bucket/movielens/processed/data.parquet", mode='overwrite')
  ```

### **3. Model Building with Spark MLlib**

- **Build and Train ALS Model:**

  ```python
  from pyspark.ml.recommendation import ALS
  
  # Initialize ALS model
  als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')
  
  # Train the model
  model = als.fit(training_df)
  ```

### **4. Model Evaluation**

- **Evaluate Model Performance:**

  ```python
  from pyspark.ml.evaluation import RegressionEvaluator
  
  predictions = cv_model.transform(test_df)
  evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
  rmse = evaluator.evaluate(predictions)
  print(f"Root-mean-square error = {rmse}")
  ```

### **5. Data Analysis with SparkSQL**

- **Write SparkSQL Queries:**

  ```python
  data_df.createOrReplaceTempView("movie_data")
  
  # Example Query for Top Rated Movies
  top_movies = spark.sql("""
    SELECT title, AVG(rating) as avg_rating, COUNT(*) as num_ratings
    FROM movie_data
    GROUP BY title
    ORDER BY avg_rating DESC
    LIMIT 10
  """)
  top_movies.show()
  ```

### **6. Visualization**

#### **a. Using Jupyter Notebooks for Visuals**

- **Visualize Results in Notebooks:**

  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Load data and create plots
  ratings_over_years = pd.read_csv('data/ratings_over_years.csv')
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=ratings_over_years, x='year', y='avg_rating')
  plt.title('Average Movie Ratings Over Years')
  plt.show()
  ```

---

## **Project Documentation**

- **README.md:**
  - **Project Title:** Building a Recommendation Engine using MovieLens Dataset with AWS and Apache Spark
  - **Description:** An end-to-end data engineering project that processes the MovieLens dataset and builds a recommendation system using collaborative filtering.
  - **Contents:**
    - **Introduction**
    - **Project Architecture**
    - **Technologies Used**
    - **Setup Instructions**
    - **Running the Project**
    - **Data Processing Steps**
    - **Model Building and Evaluation**
    - **Data Analysis and Results**
    - **Visualization**
    - **Conclusion**

- **Code Organization:**

  ```
  ├── README.md
  ├── data
  │   ├── sample_data.csv
  ├── notebooks
  │   └── visualization.ipynb
  ├── resources
  │   └── architecture_diagram.png
  └── scripts
      ├── data_analysis.py
      ├── data_processing.py
      ├── model_building.py
  ```

---

## **Best Practices**

- **Use Version Control:**
  - Initialize a Git repository and track changes regularly.

  ```bash
  git init
  git add .
  git commit -m "Initial commit with project structure and documentation"
  ```

- **Optimize Spark Jobs:**
  - Leverage Spark configurations for better performance and manage resources efficiently.

- **Security:**
  - Ensure AWS credentials are not hardcoded and use IAM roles for permission management.

---

## **Demonstrating Skills**

- **Data Processing and Engineering Concepts:**
  - Utilize Apache Spark for handling large datasets efficiently.
  
- **Machine Learning Proficiency:**
  - Implement collaborative filtering using Spark's MLlib.

- **Data Visualization:**
  - Generate insightful visualizations to communicate findings effectively.

---

## **Additional Enhancements**

- **Continuous Integration:**
  - Set up CI/CD pipelines for automated testing and deployment.

- **Containerization:**
  - Use Docker to containerize all scripts for easier deployment and execution.

- **Advanced Features:**
  - Consider integrating deep learning techniques for improved recommendation systems.
