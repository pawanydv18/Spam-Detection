import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("spam.csv",encoding='latin-1')[['v1','v2']]
df.columns = ['label','message']
# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Custom test
msg = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."]
msg_vec = vectorizer.transform(msg)
print("Prediction:", "Spam" if model.predict(msg_vec)[0] else "Ham")
