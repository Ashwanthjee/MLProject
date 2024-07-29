import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data for training the model
emotion_genre_data = {
    "happy": "Comedy",
    "sad": "Drama",
    "excited": "Action",
    "fear": "Horror",
    "love": "Romance",
    "curious": "Mystery",
    "inspired": "Biography",
    "adventurous": "Adventure",
    "tense": "Thriller",
    "intrigued": "Crime",
    "Drama": 'Drama',
    "Action": 'Action',
    "Thriller": 'Thriller',
    "Adventure": 'Adventure',
    "Comedy": 'Comedy',
    "Biography": 'Biography',
    "Mystery": 'Mystery',
    "Romance": 'Romance',
    "Crime": 'Crime',
    "Horror": 'Horror',
}

# Prepare training data
emotions = list(emotion_genre_data.keys())
genres = list(emotion_genre_data.values())

# Vectorize emotions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emotions)

# Encode genres
genre_encoder = {genre: i for i, genre in enumerate(set(genres))}
y = [genre_encoder[genre] for genre in genres]

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
with open('emotion_genre_model.pkl', 'wb') as file:
    pickle.dump((vectorizer, model, genre_encoder), file)

def predict_genre(emotion):
    """Predict the genre based on the given emotion using the trained model."""
    with open('emotion_genre_model.pkl', 'rb') as file:
        vectorizer, model, genre_encoder = pickle.load(file)
    
    emotion_vector = vectorizer.transform([emotion])
    predicted_genre_index = model.predict(emotion_vector)[0]
    
    genre_decoder = {i: genre for genre, i in genre_encoder.items()}
    return genre_decoder.get(predicted_genre_index)
