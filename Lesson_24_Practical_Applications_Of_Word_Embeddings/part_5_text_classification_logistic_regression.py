from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample labeled data
data = [
    ("I love this product", 1),
    ("This is the worst experience I've had", 0),
    ("Absolutely fantastic!", 1),
    ("I'm never coming back", 0),
    ("It's okay, not great", 0)
]

# Prepare data inputs and labels
texts, labels = zip(*data)
texts = [word_tokenize(text.lower()) for text in texts]

# Create a feature matrix
X = []
for text in texts:
    vector = np.mean([model.wv[word] for word in text if word in model.wv], axis=0)
    X.append(vector)

X = np.array(X)
y = np.array(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Evaluate the model
accuracy = classifier.score(X_test, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')
