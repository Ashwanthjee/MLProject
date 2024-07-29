from flask import Flask, render_template, request, redirect, url_for
from model import predict_genre
from utils import load_movies_from_csv, filter_movies_by_genre

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/results', methods=['GET','POST'])
def results():
    emotion = request.form['emotion'].strip()
    genre = predict_genre(emotion)
    
    if not genre:
        return render_template('results.html', error="Invalid emotion or no matching genre found.", movies=[])
    
    movies = load_movies_from_csv('movies.csv')
    filtered_movies = filter_movies_by_genre(movies, genre)
    filtered_movies.sort(key=lambda x: float(x['Rating']), reverse=True)
    
    return render_template('results.html', movies=filtered_movies[:70])

@app.route('/visualizations', methods=['GET'])
def visualizations():
    return render_template('visualizations.html')

if __name__ == '__main__':
    app.run(debug=True)
