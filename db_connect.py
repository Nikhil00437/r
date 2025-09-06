from pymongo import MongoClient


def get_movie_collection():
    client = MongoClient('localhost', 27017)
    db = client.Movies_DB
    collection = db.Movies_dataset
    print("Connected to MongoDB!")
    return collection

if __name__ == "__main__":
    movies_collection = get_movie_collection()
    # You can test by inserting a dummy document
    # movies_collection.insert_one({"title": "Test Movie", "genre": ["Action"]})
    # print(movies_collection.find_one({"title": "Test Movie"}))
    # movies_collection.delete_one({"title": "Test Movie"}) # Clean up