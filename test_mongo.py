from pymongo import MongoClient
import os

try:
    print("Attempting to connect to MongoDB...")
    client = MongoClient("mongodb://[::1]:27017", serverSelectionTimeoutMS=5000)
    print("Client created.")
    print(f"Server info: {client.server_info()}")
    print("Connection successful!")
    print(f"Databases: {client.list_database_names()}")
except Exception as e:
    print(f"Connection failed: {e}")
