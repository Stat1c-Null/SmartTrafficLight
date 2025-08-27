from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_DATABASE_URL = os.getenv("MongoDB_URL")
client = MongoClient(MONGO_DATABASE_URL)
db = client["SmartTraffic"]