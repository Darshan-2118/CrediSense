import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        # Default features; the preprocessor will adapt to whichever are present
        self.numeric_features = ['income', 'loan_amount', 'debt_to_income_ratio', 'cibil_score', 'age', 'dependents', 'previous_loans', 'missed_emis']
        self.categorical_features = ['employment_type', 'property_area']

        # Transformers are created on demand in preprocess based on available columns
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # placeholder; actual ColumnTransformer will be built in preprocess
        self.preprocessor = None

    def preprocess(self, data):
        # Accept dict-like input for tests and convert to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        # If a DataFrame is provided, build the ColumnTransformer for available cols
        if isinstance(data, pd.DataFrame):
            present_num = [c for c in self.numeric_features if c in data.columns]
            present_cat = [c for c in self.categorical_features if c in data.columns]

            transformers = []
            if present_num:
                transformers.append(('num', self.numeric_transformer, present_num))
            if present_cat:
                transformers.append(('cat', self.categorical_transformer, present_cat))

            if not transformers:
                # Nothing to transform; return raw values
                return data.values

            self.preprocessor = ColumnTransformer(transformers=transformers)
            return self.preprocessor.fit_transform(data)

        # If input is array-like, assume preprocessor has been fit and transform
        return self.preprocessor.transform(data)