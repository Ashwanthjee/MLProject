import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

# Train Support Vector Machine model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

# Save models and vectorizer
with open('models.pkl', 'wb') as file:
    pickle.dump((vectorizer, logistic_model, svm_model, genre_encoder), file)

print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
