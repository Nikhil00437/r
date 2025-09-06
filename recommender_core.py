# recommender_core.py
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import os # Import os module
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Data Retrieval ---
def get_movies_dataframe():
    """
    Connects to MongoDB (local or Atlas based on MONGO_URI env var),
    retrieves movie data from the 'Movies' collection,
    and returns it as a pandas DataFrame.
    """
    # Get MongoDB URI from environment variables
    # Default to local if MONGO_URI is not set (for local development fallback)
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/") 

    try:
        client = MongoClient(mongo_uri)
        # Ping the server to check connection
        client.admin.command('ping') 
        print(f"Connected to MongoDB: {mongo_uri.split('@')[-1] if '@' in mongo_uri else 'localhost'}")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Please ensure your MongoDB (Atlas or local) is running and accessible.")
        return pd.DataFrame() # Return empty DataFrame on connection failure


    db = client.Movies_DB
    collection = db.Movies

    print("Fetching movies from MongoDB...")
    movies_data = list(collection.find({}))
    
    if not movies_data:
        print("No movies found in the collection. Please ensure Phase 1 data ingestion was successful.")
        return pd.DataFrame()

    movies_df = pd.DataFrame(movies_data)
    
    if '_id' in movies_df.columns:
        movies_df = movies_df.drop(columns=['_id'])
    
    print(f"Loaded {len(movies_df)} movies from MongoDB.")
    return movies_df


# --- 2. Data Preprocessing for TF-IDF ---
def preprocess_data(df):
    """
    Preprocesses the DataFrame for TF-IDF vectorization.
    For MovieLens, it joins genres into a single string.
    For TMDB, it would combine 'overview', 'genres', 'cast', 'crew'.
    """
    # Assuming MovieLens structure for now, where 'genres' is a list
    # If your TMDB data has 'overview', 'keywords', 'cast', 'crew', you'd combine those here.
    df['combined_features'] = df['genres'].apply(lambda x: ' '.join(x).replace(' ', '') if isinstance(x, list) else str(x).replace(' ', ''))
    
    # You might want to clean the title to match user input more easily
    df['clean_title'] = df['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)$', '', x).strip().lower() if isinstance(x, str) else x)
    
    print("Data preprocessing complete.")
    return df

# --- 3. Feature Extraction (TF-IDF) ---
def get_tfidf_matrix(df):
    """
    Applies TF-IDF vectorization to the combined features of the movies.
    """
    tfidf = TfidfVectorizer(stop_words='english') # 'stop_words' is useful if you have plot summaries
    print("Generating TF-IDF matrix...")
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, tfidf # Return tfidf object for potential later use if needed

# --- 4. Similarity Calculation (Cosine Similarity) ---
def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculates the cosine similarity between all movie feature vectors.
    """
    print("Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Cosine similarity matrix calculated.")
    return cosine_sim

# --- 5. Recommendation Function ---
def get_recommendations(movie_title, tfidf_matrix, df, top_n=10):
    """
    Generates movie recommendations for a given movie title by calculating
    similarity on-the-fly.

    Args:
        movie_title (str): The title of the movie to get recommendations for.
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix for all movies.
        df (pd.DataFrame): The DataFrame containing movie data.
        top_n (int): The number of recommendations to return.

    Returns:
        list: A list of recommended movie titles.
    """
    clean_input_title = movie_title.lower()
    
    indices = pd.Series(df.index, index=df['clean_title']).drop_duplicates()

    # --- Find the movie index ---
    movie_idx = -1
    original_title_used = ""

    if clean_input_title in indices:
        movie_idx = indices[clean_input_title]
        original_title_used = df.loc[movie_idx, 'title']
    else:
        # Try a partial match if exact match fails
        matching_titles = df[df['clean_title'].str.contains(clean_input_title, na=False, regex=False)]
        if not matching_titles.empty:
            matched_title = matching_titles['clean_title'].iloc[0]
            print(f"Did not find exact match for '{movie_title}'. Suggesting based on '{matching_titles['title'].iloc[0]}'.")
            movie_idx = indices[matched_title]
            original_title_used = matching_titles['title'].iloc[0]
        else:
            print(f"Movie '{movie_title}' not found in the database. Please check the spelling.")
            return []

    # --- On-the-fly Calculation ---
    # 1. Get the TF-IDF vector for the input movie
    movie_vector = tfidf_matrix[movie_idx]


    # # Reshape the movie_vector to a 2D array (1 row, N columns)
    # # This is the new line to add.
    # movie_vector = movie_vector.reshape(1, -1)


    # 2. Calculate cosine similarity between this single movie and ALL others
    # The result 'sim_scores' will have shape (1, num_movies)
    sim_scores = cosine_similarity(movie_vector, tfidf_matrix)

    # 3. Flatten the array and enumerate to get (index, score) pairs
    sim_scores_list = list(enumerate(sim_scores[0]))

    # 4. Sort the movies based on the similarity scores
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)

    # 5. Get the scores of the top_n most similar movies (excluding itself)
    sim_scores_list = sim_scores_list[1:top_n+1]

    # 6. Get the movie indices
    movie_indices = [i[0] for i in sim_scores_list]

    # 7. Return the original titles of the recommended movies
    recommended_titles = df['title'].iloc[movie_indices].tolist()
    print(f"\nRecommendations for '{original_title_used}':")
    return recommended_titles

# --- Main execution flow for testing (optional, can be moved to main.py) ---
if __name__ == "__main__":
    # 1. Get DataFrame
    movies_df = get_movies_dataframe()

    if not movies_df.empty:
        # 2. Preprocess data
        movies_df = preprocess_data(movies_df)

        # 3. Get TF-IDF matrix
        tfidf_matrix, _ = get_tfidf_matrix(movies_df)

        # 4. NOTE: We NO LONGER calculate the full cosine similarity matrix here!
        #    cosine_sim = calculate_cosine_similarity(tfidf_matrix)  <-- DELETE THIS LINE

        # 5. Get Recommendations (example usage)
        print("\n--- Testing Recommendation Function ---")
        
        # Test with a known movie
        example_movie_title_1 = "Toy Story" 
        # Pass tfidf_matrix directly
        recommendations_1 = get_recommendations(example_movie_title_1, tfidf_matrix, movies_df)
        for i, movie in enumerate(recommendations_1):
            print(f"{i+1}. {movie}")

        # ... (your other tests would work the same way) ...
        print("\n-------------------------------------")
        example_movie_title_4 = "lion king"
        recommendations_4 = get_recommendations(example_movie_title_4, tfidf_matrix, movies_df)
        for i, movie in enumerate(recommendations_4):
            print(f"{i+1}. {movie}")
    else:
        print("Cannot run recommendation core without movie data. Please load data first.")