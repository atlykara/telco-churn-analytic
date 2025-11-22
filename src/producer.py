import time
import json
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "telco-churn"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

SLEEP_SECONDS = 0.3  # satırlar arası bekleme (stream hızı)


def create_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    return producer


def stream_csv_to_kafka():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    producer = create_producer()
    print(f"Streaming {len(df)} rows from {DATA_PATH} to topic '{KAFKA_TOPIC}' ...")

    try:
        for idx, row in df.iterrows():
            message = row.to_dict()
            producer.send(KAFKA_TOPIC, value=message)
            print(f"[PRODUCER] Sent row {idx}: {message.get('customerID')} (Churn={message.get('Churn')})")
            time.sleep(SLEEP_SECONDS)
    except KeyboardInterrupt:
        print("Streaming interrupted by user.")
    finally:
        producer.flush()
        producer.close()
        print("Producer closed.")


if __name__ == "__main__":
    stream_csv_to_kafka()
