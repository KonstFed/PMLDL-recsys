import os

import pandas as pd


def _read_recs(path: str):
    data = pd.read_csv(path, delimiter='\t', header=None)
    data.columns = ["user_id", "item_id", "rating", "timestamp"]
    return data

def _read_users(path: str):
    users = pd.read_csv(path, delimiter='|', header=None)
    users.columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    return users

def _read_movies(path: str):
    movies = pd.read_csv(path, delimiter='|', header=None, encoding='latin-1')
    movies.columns = ["movie_id", "title", "release_date", "video_release_date", "IMBD_url"] + [f"genre_{x}" for x in range(19)]
    movies = movies.drop("video_release_date", axis=1)
    movies = movies.fillna(method='ffill')
    movies.release_date = pd.to_datetime(movies.release_date)
    return movies

def _read_genre(path: str):
    genres = pd.read_csv(path, delimiter='|', header=None)
    genres.columns = ["genre", "id"]
    return genres

def _read_occupation(path):
    occupation = pd.read_csv(path, delimiter='|', header=None)
    return occupation

def load_part(dataset_path: str, part: int):
    train_path = os.path.join(dataset_path, f"u{part}.base")
    test_path = os.path.join(dataset_path, f"u{part}.test")

    train_part = _read_recs(train_path)
    test_part = _read_recs(test_path)
    return train_part, test_part
    
def load_datasets(dataset_path: str):
    users = _read_users(os.path.join(dataset_path, "u.user"))
    items = _read_movies(os.path.join(dataset_path, "u.item"))
    parts = []
    for part in range(1, 6):
        parts.append(load_part(dataset_path, part))
    return users, items, parts