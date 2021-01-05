from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def preprocessing_pipeline(cat_features, num_features):
    """Create pre-processing pipeline to concatenate with the final estimator."""
    cat_transformer = Pipeline([
        ('imputing', SimpleImputer(strategy='most_frequent')),
        ('oh_encoding', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_transformer = Pipeline([
        ('transforming', PowerTransformer()),
        ('imputing', SimpleImputer(strategy='mean'))
    ])

    pipeline = ColumnTransformer([
        ('categoricals', cat_transformer, cat_features),
        ('numericals', num_transformer, num_features)
    ],
        remainder='passthrough'
    )
    return pipeline
