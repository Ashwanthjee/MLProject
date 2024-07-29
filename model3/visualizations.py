import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load movie data from CSV
def load_movies_from_csv(file_path):
    """Load movie data from a CSV file and return a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

# Load movies data
movies_df = load_movies_from_csv('movies.csv')

# Ensure the 'Genre' and a text column are present in the DataFrame
text_column = 'Genre'  # Replace with the actual name of your text column

if 'Genre' in movies_df.columns and text_column in movies_df.columns:
    # Prepare training data
    genres = movies_df['Genre']
    text_data = movies_df[text_column]

    # Vectorize movie text data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)

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

    # Print accuracy
    print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")
    print(f"SVM Accuracy: {svm_accuracy:.2f}")

    # Visualize genre distribution in movies
    plt.figure(figsize=(12, 6))
    genre_counts = movies_df['Genre'].value_counts()
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='viridis')
    plt.title('Distribution of Genres in Movies Dataset')
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    # Add data labels
    for index, value in enumerate(genre_counts.values):
        plt.text(index, value + 0.5, str(value), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('static/genre_distribution.png')
    plt.close()

    # Visualize movie ratings distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(movies_df['Rating'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')

    # Add data labels
    for patch in plt.gca().patches:
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height + 0.5, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('static/ratings_distribution.png')
    plt.close()

    # Visualize model accuracy comparison
    model_names = ['Logistic Regression', 'SVM']
    accuracies = [logistic_accuracy, svm_accuracy]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=model_names, y=accuracies, palette='coolwarm')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.5)

    # Add data labels
    for index, value in enumerate(accuracies):
        plt.text(index, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('static/model_accuracy_comparison.png')
    plt.close()

else:
    print("The required columns are not present in the CSV file.")
