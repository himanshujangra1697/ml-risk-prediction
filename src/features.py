import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_feature_pipeline(numeric_features, categorical_features):
    """
    Creates a production-grade transformation pipeline.
    """
    # 1. Numeric Pipeline: Impute missing with median, then Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline: Impute missing with 'missing' label, then One-Hot Encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

if __name__ == "__main__":
    # Load your raw data
    df = pd.read_csv('data/raw/train.csv')

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    
    # print("Columns after renaming:", df.columns.tolist())
    # delete row where Churn is missing
    df = df.dropna(subset=['churn'])
    
    # Define your columns based on the dataset
    # (Update these names based on the actual CSV headers)
    target = 'churn'
    X = df.drop(columns=[target, 'customerid']) # Drop ID as it's not a feature
    y = df[target]

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and fit pipeline
    pipeline = build_feature_pipeline(num_cols, cat_cols)
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    # Save the pipeline (This is the "brain" for your API)
    joblib.dump(pipeline, 'docker/preprocessor.joblib')
    print("Feature engineering pipeline saved to docker/preprocessor.joblib")

    # Save processed data for train.py
    pd.DataFrame(X_train_processed).to_csv('data/processed/train_x.csv', index=False)
    y_train.to_csv('data/processed/train_y.csv', index=False)