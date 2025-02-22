import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("Housing.csv")

# Define Target (y) and Features (X)
y = df["price"]  # Target variable
X = df.drop(columns=["price"])  # All other columns are features

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Apply One-Hot Encoding
if categorical_cols:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Drop original categorical columns and concatenate encoded data
    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, X_encoded_df], axis=1)

    # Save encoder
    joblib.dump(encoder, 'encoder.pkl')
else:
    encoder = None
    joblib.dump(encoder, 'encoder.pkl')

# Save feature names
joblib.dump(X.columns, 'model_features.pkl')

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, 'model.pkl')

print("Model trained and saved successfully!")
