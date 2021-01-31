import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score

# Built-in python packages
import json

# Variables
DATA_FILEPATH = './reviews_train.csv'

def load_data(path: str, delimiter=',', quotechar='"') -> pd.DataFrame:
    return pd.read_csv(path, delimiter=delimiter, quotechar=quotechar)

def preprocess(data: pd.DataFrame, column_name: str = 'helpful'):
    if column_name in data.columns:
        data[column_name] = data[column_name].apply(lambda x: tuple(json.loads(x)))
    if 'score' in data.columns:
        data['score'] = data['score'].astype(str)
    return data.dropna()

def split_train_test(data: pd.DataFrame, train_size: float = 0.9, class_name: str = 'score', groupby: str = 'asin'):
    group_split = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    for train_idx, test_idx in group_split.split(data, groups=data['asin']):
        data_train = data.iloc[train_idx].reset_index(drop=True)
        data_test = data.iloc[test_idx].reset_index(drop=True)
        
        return (
            data_train.drop(columns=[class_name]), data_train[class_name],
            data_test.drop(columns=[class_name]), data_test[class_name])

def score_metric(y_true, y_pred) -> float:
    return balanced_accuracy_score(y_true, y_pred)

def get_all_data():
    return split_train_test(preprocess(load_data(DATA_FILEPATH)))

def normalize_review_weight(weight):
    normalized_weight = 1 - (min(weight) + 1 )/(max(weight) + 1)
    if weight[1] > weight[0]:
        normalized_weight *= -1
    return (normalized_weight + 1) / 2