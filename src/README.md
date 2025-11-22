1) Start Zookeeper
   bin/zookeeper-server-start.sh config/zookeeper.properties

2) Start Kafka broker
   bin/kafka-server-start.sh config/server.properties

3) Create topic (only once):
   bin/kafka-topics.sh --create \
     --topic telco-churn \
     --bootstrap-server localhost:9092 \
     --partitions 1 \
     --replication-factor 1

4) Run model training:
   python src/train_model.py

5) Start streaming consumer:
   python src/spark_stream.py

6) Start producer:
   python src/producer.py
