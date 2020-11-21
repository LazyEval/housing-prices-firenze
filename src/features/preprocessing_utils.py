import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = dict()

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X.loc[X[col].notna(), col])
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))

            # Set unknown to new value so transform on test set handles unknown values
            max_value = max(le_dict.values())
            le_dict['_unk'] = max_value + 1

            self.encoders[col] = le_dict
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            le_dict = self.encoders[col]
            X.loc[X[col].notna(), col] = X.loc[X[col].notna(), col].apply(
                lambda x: le_dict.get(x, le_dict['_unk'])).values
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class ColumnSelector:
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        cpy_df = X[self.columns].copy()
        return cpy_df


def create_sqm_bins(data):
    """Create discretization of the square meters feature."""
    data['Superficie_bins'] = pd.cut(data['Superficie_m2'], bins=[0, 60, 80, 100, 120, 160, 200, 20000])
    return data


def imputation_with_bins(impute_col, bins_col):
    """Impute missing values in number of bathrooms and number of rooms using the discretized square meters feature."""

    def imputer(data):
        for sqm in data[bins_col].unique():
            mask = data[bins_col] == sqm
            data.loc[mask & (data[impute_col] == 0), impute_col] = data.loc[mask, impute_col].value_counts().index[0]
        return data

    return imputer
