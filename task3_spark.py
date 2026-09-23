from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, round, when

spark = SparkSession.builder \
    .appName("EcommerceTimeAndAgeAnalysis") \
    .getOrCreate()

input_path = "hdfs://namenode:9000/e-commerce_data/ecommerce_dataset.csv"

time_output_path = "hdfs://namenode:9000/e-commerce_data/task3_order_time_results"
age_output_path = "hdfs://namenode:9000/e-commerce_data/task3_age_group_results"

# Read CSV from HDFS
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(input_path)


# ============================================================
# ANALYSIS 1: ORDER TIME ANALYSIS
# ============================================================

# Group e-commerce records by order hour
# and calculate number of orders and total sales

order_time_summary = df.groupBy("order_hour").agg(
    count("order_id").alias("number_of_orders"),
    round(sum("total_price_usd"), 2).alias("total_sales_usd")
)

# Sort results by order hour
order_time_summary = order_time_summary.orderBy(
    col("order_hour").asc()
)

# Action: display results
print("===== ORDER TIME ANALYSIS =====")
order_time_summary.show(50, truncate=False)

# Action: write results to HDFS
order_time_summary.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(time_output_path)

print("===== ORDER TIME RESULTS WRITTEN TO HDFS =====")
print(time_output_path)


# ============================================================
# ANALYSIS 2: CUSTOMER AGE GROUP ANALYSIS
# ============================================================

# Create customer age groups
age_group_df = df.withColumn(
    "age_group",
    when((col("age") >= 18) & (col("age") <= 25), "18-25")
    .when((col("age") >= 26) & (col("age") <= 35), "26-35")
    .when((col("age") >= 36) & (col("age") <= 45), "36-45")
    .when((col("age") >= 46) & (col("age") <= 55), "46-55")
    .when(col("age") >= 56, "56+")
    .otherwise("Unknown")
)

# Group records by age group and calculate
# number of orders and total sales

age_group_summary = age_group_df.groupBy("age_group").agg(
    count("order_id").alias("number_of_orders"),
    round(sum("total_price_usd"), 2).alias("total_sales_usd")
)

# Sort results by age group
age_group_summary = age_group_summary.orderBy(
    col("age_group").asc()
)

# Action: display results
print("===== CUSTOMER AGE GROUP ANALYSIS =====")
age_group_summary.show(50, truncate=False)

# Action: write results to HDFS
age_group_summary.write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(age_output_path)

print("===== AGE GROUP RESULTS WRITTEN TO HDFS =====")
print(age_output_path)

spark.stop()
