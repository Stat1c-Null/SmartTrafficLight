docker build -t commentxd/smart_traffic_backend:latest .
docker run -d -p 8000:8000 --name SmartTraffic commentxd/smart_traffic_backend:latest