from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    countDistinct,
    sum as spark_sum,
    avg,
    count,
    when
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# ============================================================
# TASK 4: CUSTOMER SEGMENTATION USING SPARK MLlib
# ============================================================

spark = (
    SparkSession.builder
    .appName("EcommerceCustomerSegmentation")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------
# 1. INPUT / OUTPUT PATHS
# ------------------------------------------------------------

input_path = "hdfs://namenode:9000/e-commerce_data/ecommerce_dataset.csv"

output_path = "hdfs://namenode:9000/e-commerce_data/task4_customer_segments"

summary_path = "hdfs://namenode:9000/e-commerce_data/task4_cluster_summary"


print("\n==============================================")
print("TASK 4: CUSTOMER SEGMENTATION")
print("==============================================")
print("Reading data from HDFS...")


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(input_path)
)

print("\n===== RAW DATA =====")
print("Number of records:", df.count())

print("\n===== DATA SCHEMA =====")
df.printSchema()


# ------------------------------------------------------------
# 3. DATA PREPROCESSING
# ------------------------------------------------------------

print("\n==============================================")
print("DATA PREPROCESSING")
print("==============================================")


# Keep only records with a valid customer ID
clean_df = df.filter(
    col("customer_id").isNotNull()
)


# Remove records where important numerical values are missing
clean_df = clean_df.filter(
    col("quantity").isNotNull()
    & col("total_price_usd").isNotNull()
    & col("profit_usd").isNotNull()
    & col("customer_loyalty_score").isNotNull()
    & col("age").isNotNull()
)


# Keep sensible values
clean_df = clean_df.filter(
    (col("quantity") >= 0)
    & (col("total_price_usd") >= 0)
    & (col("profit_usd").isNotNull())
    & (col("age") > 0)
)


print("Records after preprocessing:", clean_df.count())


# ------------------------------------------------------------
# 4. CUSTOMER-LEVEL FEATURE ENGINEERING
# ------------------------------------------------------------

print("\n==============================================")
print("CUSTOMER FEATURE ENGINEERING")
print("==============================================")


customer_features = (
    clean_df
    .groupBy("customer_id")
    .agg(
        countDistinct("order_id").alias("total_orders"),
        spark_sum("quantity").alias("total_quantity"),
        spark_sum("total_price_usd").alias("total_sales"),
        spark_sum("profit_usd").alias("total_profit"),
        avg("customer_loyalty_score").alias("loyalty_score"),
        avg("age").alias("average_age"),
        avg("rating").alias("average_rating"),
        avg("discount_percent").alias("average_discount")
    )
)


# Remove any remaining null values
customer_features = customer_features.dropna()


print(
    "Number of customers:",
    customer_features.count()
)

print("\n===== CUSTOMER FEATURES =====")

customer_features.show(10, truncate=False)


# ------------------------------------------------------------
# 5. SELECT FEATURES FOR MACHINE LEARNING
# ------------------------------------------------------------

print("\n==============================================")
print("FEATURE SELECTION")
print("==============================================")


feature_columns = [
    "total_orders",
    "total_quantity",
    "total_sales",
    "total_profit",
    "loyalty_score"
]


print("Features used for K-Means:")
for feature in feature_columns:
    print("-", feature)


# ------------------------------------------------------------
# 6. CREATE FEATURE VECTOR
# ------------------------------------------------------------

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="raw_features"
)

assembled = assembler.transform(customer_features)


# ------------------------------------------------------------
# 7. STANDARDISE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

scaler_model = scaler.fit(assembled)

scaled_data = scaler_model.transform(assembled)


# ------------------------------------------------------------
# 8. TRAIN K-MEANS MODEL
# ------------------------------------------------------------

print("\n==============================================")
print("K-MEANS MODEL TRAINING")
print("==============================================")


# Three customer segments:
# 0 = lower-value
# 1 = medium-value
# 2 = higher-value
#
# Cluster labels themselves are arbitrary.
# We interpret them later using cluster statistics.

kmeans = KMeans(
    k=3,
    seed=42,
    featuresCol="features",
    predictionCol="cluster"
)

model = kmeans.fit(scaled_data)


# ------------------------------------------------------------
# 9. MAKE PREDICTIONS
# ------------------------------------------------------------

predictions = model.transform(scaled_data)


print("\n===== SAMPLE CLUSTER PREDICTIONS =====")

predictions.select(
    "customer_id",
    "total_orders",
    "total_quantity",
    "total_sales",
    "total_profit",
    "loyalty_score",
    "cluster"
).show(15, truncate=False)


# ------------------------------------------------------------
# 10. MODEL EVALUATION
# ------------------------------------------------------------

print("\n==============================================")
print("MODEL EVALUATION")
print("==============================================")


evaluator = ClusteringEvaluator(
    predictionCol="cluster",
    featuresCol="features",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

silhouette = evaluator.evaluate(predictions)


print(
    "Silhouette Score:",
    round(silhouette, 4)
)


# ------------------------------------------------------------
# 11. CLUSTER SUMMARY
# ------------------------------------------------------------

print("\n==============================================")
print("CLUSTER ANALYSIS")
print("==============================================")


cluster_summary = (
    predictions
    .groupBy("cluster")
    .agg(
        count("*").alias("number_of_customers"),
        avg("total_orders").alias("avg_orders"),
        avg("total_quantity").alias("avg_quantity"),
        avg("total_sales").alias("avg_sales"),
        avg("total_profit").alias("avg_profit"),
        avg("loyalty_score").alias("avg_loyalty"),
        spark_sum("total_sales").alias("cluster_total_sales"),
        spark_sum("total_profit").alias("cluster_total_profit")
    )
    .orderBy("cluster")
)


print("\n===== CUSTOMER SEGMENT SUMMARY =====")

cluster_summary.show(
    truncate=False
)


# ------------------------------------------------------------
# 12. SAVE CUSTOMER SEGMENTS TO HDFS
# ------------------------------------------------------------

print("\n==============================================")
print("WRITING CUSTOMER SEGMENTS TO HDFS")
print("==============================================")


(
    predictions
    .select(
        "customer_id",
        "total_orders",
        "total_quantity",
        "total_sales",
        "total_profit",
        "loyalty_score",
        "average_age",
        "average_rating",
        "average_discount",
        "cluster"
    )
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(output_path)
)


# ------------------------------------------------------------
# 13. SAVE CLUSTER SUMMARY
# ------------------------------------------------------------

(
    cluster_summary
    .write
    .mode("overwrite")
    .option("header", True)
    .csv(summary_path)
)


print("\n==============================================")
print("TASK 4 COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nSilhouette Score:", round(silhouette, 4))

print("\nCustomer segment results:")
print(output_path)

print("\nCluster summary:")
print(summary_path)


spark.stop()