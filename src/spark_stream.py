from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)
from pyspark.ml import PipelineModel


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "telco-churn"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / "data" / "telco_churn_model")


def main():
    spark = (
        SparkSession.builder
        .appName("TelcoChurnKafkaSparkStreaming")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # Offline eğitilen pipeline modeli yükle
    model = PipelineModel.load(MODEL_PATH)

    # Kafka'dan gelecek Telco JSON için şema
    schema = StructType([
        StructField("customerID", StringType(), True),
        StructField("gender", StringType(), True),
        StructField("SeniorCitizen", IntegerType(), True),
        StructField("Partner", StringType(), True),
        StructField("Dependents", StringType(), True),
        StructField("tenure", IntegerType(), True),
        StructField("PhoneService", StringType(), True),
        StructField("MultipleLines", StringType(), True),
        StructField("InternetService", StringType(), True),
        StructField("OnlineSecurity", StringType(), True),
        StructField("OnlineBackup", StringType(), True),
        StructField("DeviceProtection", StringType(), True),
        StructField("TechSupport", StringType(), True),
        StructField("StreamingTV", StringType(), True),
        StructField("StreamingMovies", StringType(), True),
        StructField("Contract", StringType(), True),
        StructField("PaperlessBilling", StringType(), True),
        StructField("PaymentMethod", StringType(), True),
        StructField("MonthlyCharges", DoubleType(), True),
        StructField("TotalCharges", StringType(), True),
        StructField("Churn", StringType(), True),
    ])

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    value_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str")

    parsed_df = value_df.select(
        F.from_json(F.col("json_str"), schema).alias("data")
    ).select("data.*")

    # TotalCharges'u double'a çevir (offline eğitimle aynı mantık)
    parsed_df = parsed_df.withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges")) == "", None)
         .otherwise(F.col("TotalCharges"))
         .cast("double")
    )

    # Modelin beklediği numeric kolonlarda eksik olanları at
    feature_df = parsed_df.dropna(
        subset=["tenure", "MonthlyCharges", "TotalCharges"]
    )

    # Pipeline modelini uygula
    pred_df = model.transform(feature_df)

    extract_churn_prob = F.udf(
        lambda v: float(v[1]) if v is not None and len(v) > 1 else None,
        DoubleType()
    )

    pred_df = pred_df.withColumn(
        "churn_probability",
        extract_churn_prob(F.col("probability"))
    )

    result_df = pred_df.select(
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "InternetService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
        "prediction",
        "churn_probability",
    )

    query = (
        result_df.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
