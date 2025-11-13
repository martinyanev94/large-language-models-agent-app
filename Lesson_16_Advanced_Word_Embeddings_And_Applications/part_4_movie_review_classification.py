# Sample movie reviews and labels
reviews = [
    "I loved the movie. It was fantastic!",
    "This film was okay, but not great.",
    "I hated every single moment.",
    "Absolutely amazing! Five stars!",
    "Worst movie ever. I want my money back."
]
labels = [1, 1, 0, 1, 0]  # 1 for positive, 0 for negative
import numpy as np

def create_embeddings(reviews):
    embeddings = []
    for review in reviews:
        words = review.lower().split()  # Tokenizing by space
        vectors = [fasttext_model.wv[word] for word in words if word in fasttext_model.wv]  # Getting vectors for known words
        average_vector = np.mean(vectors, axis=0) if vectors else np.zeros(fasttext_model.vector_size)
        embeddings.append(average_vector)
    return np.array(embeddings)

# Create embeddings for the reviews
embeddings = create_embeddings(reviews)
pip install scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Initializing the classifier
classifier = LogisticRegression()

# Training the model
classifier.fit(X_train, y_train)

# Making predictions
predictions = classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the model is:", accuracy)
