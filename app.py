# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import sys
import threading

# Import functions from your recommender_core.py
from recommender_core import get_movies_dataframe, preprocess_data, get_tfidf_matrix, get_recommendations

app = Flask(__name__)

# Global variables to store our pre-calculated data
movies_df = None
tfidf_matrix = None # Change this from cosine_sim_matrix to tfidf_matrix
_recommender_initialized = False
_recommender_lock = threading.Lock()

# --- Application Initialization ---
@app.before_request
def initialize_recommender():
    global movies_df, tfidf_matrix, _recommender_initialized # Change variable here

    with _recommender_lock:
        if not _recommender_initialized:
            print("🎬 Initializing Flask Movie Recommender Engine... 🍿")

            # Step 1: Get DataFrame
            movies_df = get_movies_dataframe()
            if movies_df.empty:
                print("ERROR: No movie data available. Please ensure your MongoDB has data and is running.")
                sys.exit(1)

            # MODIFICATION: Reduce the dataset size to a manageable number of movies.
            movies_df = movies_df.head(10000)
            print(f"Using a reduced dataset of the first {len(movies_df)} movies to prevent MemoryError.")

            # Step 2: Preprocess data
            movies_df = preprocess_data(movies_df)

            # Step 3: Get TF-IDF matrix
            tfidf_matrix, _ = get_tfidf_matrix(movies_df)

            # We DO NOT need to calculate the full cosine similarity matrix anymore.
            # The 'recommender_core.py' script now calculates it on-the-fly.
            
            _recommender_initialized = True
            print("✅ Movie Recommender Engine initialized successfully!")

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        movie_title = request.form['movie_title']
        
        if not movie_title:
            return jsonify({'error': 'Please enter a movie title.'}), 400

        # Check if the needed global variables are set.
        if movies_df is None or tfidf_matrix is None:
            return jsonify({'error': 'Recommender engine not initialized. Please restart the server.'}), 500

        # Pass the tfidf_matrix, not the old cosine_sim_matrix.
        recommendations = get_recommendations(movie_title, tfidf_matrix, movies_df)

        if recommendations:
            return jsonify({'recommendations': recommendations})
        else:
            return jsonify({'message': f"Could not find recommendations for '{movie_title}'. Please try another title or check your spelling."})

if __name__ == '__main__':
    app.run(debug=True)