# data_ingestion.py
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables

def load_movies_to_mongodb(csv_path="dataset/movies.csv"): # Assuming your CSV is in 'dataset' folder
    """
    Loads movie data from a CSV file into the MongoDB 'Movies' collection.
    Connects to MongoDB using MONGO_URI from .env, or defaults to localhost.
    """
    # Get MongoDB URI from environment variables
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/") 

    try:
        client = MongoClient(mongo_uri)
        client.admin.command('ping') 
        print(f"Connected to MongoDB for ingestion: {mongo_uri.split('@')[-1] if '@' in mongo_uri else 'localhost'}")
    except Exception as e:
        print(f"Error connecting to MongoDB for ingestion: {e}")
        print("Please ensure your MongoDB (Atlas or local) is running and accessible.")
        return

    db = client.Movies_DB
    collection = db.Movies

    # Clear existing data ONLY if you want to completely refresh
    # For initial load, you might want to skip this if you're appending
    # If you're sure you want to replace everything:
    # collection.delete_many({}) 
    # print("Cleared existing movie data in MongoDB.")

    try:
        movies_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}. Please check the path.")
        return

    movies_list = []
    for index, row in movies_df.iterrows():
        genres = row['genres'].split('|')
        movie_doc = {
            "movie_id": int(row['movieId']),
            "title": row['title'],
            "genres": genres,
            # Add other fields if your CSV/dataset has them (e.g., 'overview', 'cast')
        }
        movies_list.append(movie_doc)

    if movies_list:
        try:
            # Use ordered=False for faster inserts, but might stop on first error
            collection.insert_many(movies_list, ordered=False) 
            print(f"Successfully inserted {len(movies_list)} movies into MongoDB Atlas.")
        except Exception as e:
            print(f"Error inserting documents: {e}")
            print("Documents might already exist or there's a unique key violation if you're not clearing the collection.")
    else:
        print("No movies to insert.")

if __name__ == "__main__":
    # Ensure 'dataset/movies.csv' exists relative to this script
    load_movies_to_mongodb()