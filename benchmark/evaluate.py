import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k

from dataset import load_datasets

users, items, parts = load_datasets("data/raw/ml-100k")


categories_occupation = [
    ['administrator', 'artist', 'doctor', 'educator', 'engineer',
     'entertainment', 'executive', 'healthcare', 'homemaker', 'lawyer',
     'librarian', 'marketing', 'none', 'other', 'programmer', 'retired',
     'salesman', 'scientist', 'student', 'technician', 'writer']
]
encoder = OneHotEncoder(categories=categories_occupation, sparse=False)  # Specify sparse=False to get a dense array


def get_user_item_sparse(interactions: pd.DataFrame, threshold=0):
    data = interactions[interactions['rating'] > threshold]

    rows = data['user_id'] - 1
    cols = data['item_id'] - 1
    data = data['rating'] / 5
    
    return csr_matrix((data, (rows, cols)), shape=(943, 1682))

def get_user_features_sparse(users: pd.DataFrame):
    normalised = users.drop('zip_code', axis=1)
    # Normalize 'age'
    normalised['age'] = users['age'] / 73
    
    # Drop the original 'occupation' column and concatenate the encoded one-hot columns
    normalised = pd.get_dummies(normalised, columns=['occupation'])
    normalised.gender = normalised.gender == 'M'
    normalised.user_id = normalised.user_id - 1
    normalised.set_index('user_id', inplace=True)
    normalised = normalised.astype('float32')
    return csr_matrix(normalised, shape=(943, 23))

def get_item_features_sparse(items: pd.DataFrame):
    normalised = items.drop(["release_date", "title", "IMBD_url"], axis=1)
    normalised.movie_id = normalised.movie_id - 1
    normalised.set_index('movie_id', inplace=True)
    return csr_matrix(normalised, shape=(1682, 19))
    
user_features = get_user_features_sparse(users)
item_features = get_item_features_sparse(items)

all_precisions = []
all_recals = []

for part in tqdm(parts):
    train_set = get_user_item_sparse(part[0])
    test_set = get_user_item_sparse(part[1], threshold=2.5)
    # model = LightFM(no_components=160, loss='warp', item_alpha=1e-7, learning_rate=0.02, max_sampled=50)
    model = LightFM(no_components=10, loss='warp', item_alpha=1e-4, learning_rate=0.01, max_sampled=40)
    model.fit(train_set, user_features=user_features, item_features=item_features, epochs=40, num_threads=4)
    precision = precision_at_k(model, test_set, user_features=user_features, item_features=item_features).mean()
    recal = recall_at_k(model, test_set, user_features=user_features, item_features=item_features).mean()

    all_precisions.append(precision)
    all_recals.append(recal)

print(all_precisions)
print(f"MAP@10: {sum(all_precisions) / len(all_precisions):5f}")
print(all_recals)
print(f"MAR@10: {sum(all_recals) / len(all_recals):5f}")