import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the pharmacovigilance dataset
DATA_PATH = BASE_DIR / "data" / "cases.csv"

df = pd.read_csv(DATA_PATH)


# Input: adverse event description
X = df["event"]

# Target: seriousness classification
y = df["seriousness"]


# Create the NLP + Machine Learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Train the model
model.fit(X, y)


# Example new pharmacovigilance case
new_case = ["Severe allergic reaction with difficulty breathing"]

prediction = model.predict(new_case)

print("New Case:")
print(new_case[0])

print("\nPredicted Seriousness:")
print(prediction[0])
