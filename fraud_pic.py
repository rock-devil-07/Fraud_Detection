import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('new_farud_no_label.csv')

# Replace null values with an empty string
df = df.where(pd.notnull(df), '')

# Add a 'Label' column based on the 'Amount' column
df['Label'] = df['Amount'].apply(lambda x: 1 if x > 5000 else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['Label'], test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and Logistic Regression
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('model', LogisticRegression())
])

# Train the pipeline
clf.fit(X_train, y_train)

# Evaluate the model
print("Model Accuracy on Test Data:", clf.score(X_test, y_test))

# Save the pipeline to a pickle file
with open('fraud_detection_pipeline.pkl', 'wb') as file:
    pickle.dump(clf, file)

print("Pipeline has been saved as 'fraud_detection_pipeline.pkl'")
