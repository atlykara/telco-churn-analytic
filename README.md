## 1. Start Zookeeper: First Terminal
   bin/zookeeper-server-start.sh config/zookeeper.properties

## 2. Start Kafka broker: Second Terminal
   bin/kafka-server-start.sh config/server.properties

## 3. Create topic (only once):
   bin/kafka-topics.sh --create \
     --topic telco-churn \
     --bootstrap-server localhost:9092 \
     --partitions 1 \
     --replication-factor 1

## 4. Run model training:
   python src/train_model.py

## 5. Start streaming consumer: Third Terminal
   python src/spark_stream.py

## 6. Start producer: Fourth Terminal
   python src/producer.py
