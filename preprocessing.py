import pandas as pd
from pymongo import MongoClient

def get_movies_dataframe():
    client = MongoClient('localhost', 27017)
    db = client.Movies_DB
    collection = db.Movies_dataset

    movies_data = list(collection.find({}))
    movies_df = pd.DataFrame(movies_data)

    # Drop the MongoDB _id column if not needed
    if '_id' in movies_df.columns:
        movies_df = movies_df.drop(columns=['_id'])

    # For MovieLens, you'll have 'genres' as a list. Join it back to a string for TF-IDF.
    # If using TMDB, you might have 'overview' or other text fields.
    movies_df['genres_str'] = movies_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

    print(f"Loaded {len(movies_df)} movies from MongoDB.")
    return movies_df

if __name__ == "__main__":
    df = get_movies_dataframe()
    print(df.head())