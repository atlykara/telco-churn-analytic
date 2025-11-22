## 1. Start Zookeeper: First Terminal
   ```bash
   cd ~/tools/kafka
   bin/zookeeper-server-start.sh config/zookeeper.properties
   ```

## 2. Start Kafka broker: Second Terminal
   ```bash
   cd ~/tools/kafka
   bin/kafka-server-start.sh config/server.properties
   ```

## 3. Create topic (only once):
   ```bash
   bin/kafka-topics.sh --create \
     --topic telco-churn \
     --bootstrap-server localhost:9092 \
     --partitions 1 \
     --replication-factor 1
   ```

## 4. Run model training:
```bash
   python src/train_model.py
```
## 5. Start streaming consumer: Third Terminal
```bash
   cd ~/projects/telco-churn-analytic
   source .venv/bin/activate
   cd src
   python spark_stream.py
```
## 6. Start producer: Fourth Terminal
```bash
   cd ~/projects/telco-churn-analytic
   source .venv/bin/activate
   cd src
   python producer.py
```

