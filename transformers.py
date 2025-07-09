
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import LeaveOneOutEncoder


"""
    impute mean for features with missing value
    that have at least NON-NULL in each unique countries
"""
class CountryLevelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='Country', columns_to_impute=None):
        self.country_col = country_col
        self.columns_to_impute = columns_to_impute
        self.means_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        self.means_ = {}

        for col in self.columns_to_impute:
            valid = X[X[col].notnull()].groupby(self.country_col)[col].mean()
            self.means_[col] = valid.to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns_to_impute:
            X[col] = X.apply(
                lambda row: self._get_country_value(row, col),
                axis=1
            )
        return X
    
    def _get_country_value(self, row, col):
        if pd.notnull(row[col]):
            return row[col]

        country = row[self.country_col]
        mean = self.means_.get(col, {})
        value = mean.get(country, np.nan)

        return value if pd.notnull(value) else row[col]


"""
    impute median for features that are ALL NULL
    in each unique countries based on
    Status (Developing/Developed)
"""
class StatusLevelImputer(BaseEstimator, TransformerMixin):
    def __init__(self, status_col='Status', columns_to_impute=None):
        self.status_col = status_col
        self.columns_to_impute = columns_to_impute
        self.medians_ = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        self.medians_ = (
            X.groupby(self.status_col)[self.columns_to_impute]
              .median()
              .to_dict(orient='index')
        )
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns_to_impute:
            X[col] = X.apply(
                lambda row: self._impute_from_status(row, col),
                axis=1
            )
        return X
    
    def _impute_from_status(self, row, col):
        if pd.notnull(row[col]):
            return row[col]
        return self.medians_[row[self.status_col]][col]
    

"""
For features needing winsorization
"""
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.bounds = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.bounds[col] = (lower, upper)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in self.columns:
            lower, upper = self.bounds[col]
            X[col] = np.clip(X[col], lower, upper)
        return X
    
    # need to be added for custom classes since ColumnTransformers has `set_output` enabled
    def set_output(self, transform=None):
        return self
    

# LeaveOneOutEncoding for column `Country`
class LOOEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder = LeaveOneOutEncoder(cols=columns)

    def fit(self, X, y):
        self.encoder.fit(X[self.columns], y)
        return self
    
    def transform(self, X):
        encoded = self.encoder.transform(X[self.columns])
        X = X.copy()
        for col in self.columns:
            X[col + '_encoded'] = encoded[col]
            X.drop(columns=col, inplace=True)
        return X
    
    # need to be added for custom classes since ColumnTransformers has `set_output` enabled
    def set_output(self, transform=None):
        return self
    

# remove prefixes between columntransforms
class RemovePrefix(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = [col.split('__')[-1] for col in X.columns]
        else:
            self.columns_ = [f"col_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X.values, columns=self.columns_, index=X.index)
        else:
            return pd.DataFrame(X, columns=self.columns_)

"""class RemovePrefix(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = [col.split('__')[-1] for col in X.columns]
        #else:
            #self.columns_ = [f'col_{i}' for i in range(X.shape[1])]
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns_, index=X.index)"""


# reflect + log function
def reflect_log_transformer(x):
    """x = np.array(x)
    max_val = np.nanmax(x, axis=0) + 1
    return np.log(max_val - x)"""

    x = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x
    max_val = x.max(skipna=True) + 1
    return np.log(max_val - x)


def square_transformer(x):
    if isinstance(x, pd.DataFrame):
        return  x ** 2
    else:
        return pd.DataFrame(x) ** 2
