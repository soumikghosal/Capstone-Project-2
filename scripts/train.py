import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import cloudpickle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

df = pd.read_csv("data/diabetes_prediction_dataset.csv")

def standardize_column_names(col_lst):
    return [col.lower().replace(' ', '_') for col in col_lst]

df.columns = standardize_column_names(df.columns)

num_col = [col for col in df.columns if is_numeric_dtype(df[col])]
cat_col = [col for col in df.columns if col not in num_col]

num_col.remove("id")
df.drop(columns="id", inplace=True)


# target column
target_col = "diabetes"
# removing the target column
num_col.remove(target_col)

seed_value = 42

df_full_train, df_test = train_test_split(df, test_size=0.15, random_state=seed_value, stratify=df[target_col])
df_train, df_val = train_test_split(df_full_train, test_size=df_test.shape[0], random_state=seed_value, stratify=df_full_train[target_col])
len(df_train), len(df_val), len(df_test)

X_full_train, y_full_train = df_full_train.drop(target_col, axis=1), df_full_train[target_col]
X_train, y_train =  df_train.drop(target_col, axis=1), df_train[target_col]
X_val, y_val =  df_val.drop(target_col, axis=1), df_val[target_col]
X_test, y_test =  df_test.drop(target_col, axis=1), df_test[target_col]

class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.ss = StandardScaler().set_output(transform="pandas")
        self.numeric_cols = numeric_cols
        return

    def fit(self, X):
        self.imputer.fit(X[self.numeric_cols])
        self.ss.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        X[self.numeric_cols] = self.imputer.transform(X[self.numeric_cols])
        X[self.numeric_cols] = self.ss.transform(X[self.numeric_cols])
        return X.to_dict("records")

numeric_cols = X_train.select_dtypes(exclude=["object"]).columns
pipe = Pipeline([('ss', MyStandardScaler(numeric_cols=numeric_cols)), ('dv', DictVectorizer(sparse=False).set_output(transform="pandas"))])

dict_X_train = pipe.fit_transform(X_train)
dict_X_full_train = pipe.transform(X_full_train)
dict_X_val = pipe.transform(X_val)
dict_X_test = pipe.transform(X_test)


class_full_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_full_train), y=y_full_train)
class_full_weight = dict(zip(np.unique(y_full_train), class_full_weight))

class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(zip(np.unique(y_train), class_weight))

sample_full_weights = compute_sample_weight(
    class_weight=class_full_weight,
    y=y_full_train
)

### LOGISTIC REGRESSION 
lr = LogisticRegression(C= 20, class_weight= 'balanced', max_iter = 200, solver= 'lbfgs')

if __name__ == "__main__":
    lr.fit(dict_X_full_train, y_full_train)#, sample_weight=sample_full_weights, eval_set=(X_test, y_test))

    with open('models/diabetes_pred_model.bin', 'wb') as f_out:
        cloudpickle.dump((pipe, lr), f_out)
        print("Model saved")



