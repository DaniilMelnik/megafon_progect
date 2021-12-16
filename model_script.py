import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import dask.dataframe as dd

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


if __name__ == '__main__':
    df_features = dd.read_csv('features.csv', sep='\t')
    df = dd.read_csv('data_test.csv')
    df = df.merge(df_features, on='id').compute()
    df['target'] = model.predict(df)
    df = df.rename(columns={'buy_time_x': 'buy_time'})
    columns = ['buy_time', 'id', 'vas_id', 'target']
    df[columns].to_csv('answers_test.csv')
