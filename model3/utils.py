import csv
import textwrap

def load_movies_from_csv(file_path):
    """Load movie data from a CSV file and return a list of dictionaries."""
    movies = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                movies.append(row)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return movies

def filter_movies_by_genre(movies, genre):
    """Filter movies by the specified genre."""
    return [movie for movie in movies if genre in movie['Genre'].split(', ')]

def wrap_text(text, width=20):
    """Wrap text to a specified width."""
    return "\n".join(textwrap.wrap(text, width))
