from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = str(BASE_DIR / "data" / "telco_churn_model")


def main():
    spark = (
        SparkSession.builder
        .appName("TelcoChurnOfflineModelTraining")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # 1) Veri setini oku
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(DATA_PATH)
    )

    # TotalCharges bazen boş string, onları None yapıp double'a cast et
    df = df.withColumn(
        "TotalCharges",
        F.when(F.trim(F.col("TotalCharges")) == "", None)
         .otherwise(F.col("TotalCharges"))
         .cast("double")
    )

    cols_to_use = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    df = df.select(*cols_to_use)

    # Eksik kritik numeric veya label satırlarını at
    df = df.dropna(subset=["tenure", "MonthlyCharges", "TotalCharges", "Churn"])

    # *** ÖNEMLİ DÜZELTME ***
    # Churn sadece "Yes" ve "No" olan satırlar kalsın.
    # Böylece model tam ikili (binary) sınıflandırma yapar.
    df = df.filter(F.col("Churn").isin("Yes", "No"))

    # 2) Feature ve label tanımı
    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

    categorical_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    # Churn (Yes/No) -> label (0/1)
    label_indexer = StringIndexer(
        inputCol="Churn",
        outputCol="label",
        handleInvalid="error",  # Artık sadece Yes/No var, başka değer görürse bilerek patlasın
    ).fit(df)

    df_labeled = label_indexer.transform(df)

    print("Label mapping (index -> Churn value):")
    for idx, lab in enumerate(label_indexer.labels):
        print(f"  {idx} -> {lab}")
    # Burada sadece 2 değer görmelisin, örn: 0 -> 'No', 1 -> 'Yes'

    # Kategorik kolonlar için indexer
    cat_indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep",
        )
        for c in categorical_cols
    ]

    cat_idx_cols = [f"{c}_idx" for c in categorical_cols]
    cat_ohe_cols = [f"{c}_ohe" for c in categorical_cols]

    encoder = OneHotEncoder(
        inputCols=cat_idx_cols,
        outputCols=cat_ohe_cols,
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=numeric_cols + cat_ohe_cols,
        outputCol="features",
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0,
        family="binomial",  # Binary olarak sabitliyoruz
    )

    pipeline = Pipeline(
        stages=cat_indexers + [encoder, assembler, lr]
    )

    train_df, test_df = df_labeled.randomSplit([0.8, 0.2], seed=42)

    model = pipeline.fit(train_df)

    # 4) Değerlendirme
    predictions = model.transform(test_df)

    correct = predictions.filter(F.col("prediction") == F.col("label")).count()
    total = predictions.count()
    accuracy = correct / total if total > 0 else 0.0

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = evaluator.evaluate(predictions)

    print(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Test AUC:      {auc:.4f}")

    # 5) Modeli kaydet
    # Eski bir model klasörü varsa temizle
    model.write().overwrite().save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
