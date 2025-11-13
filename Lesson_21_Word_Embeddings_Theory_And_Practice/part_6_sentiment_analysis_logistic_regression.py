from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample labeled data for sentiment analysis
X = [
    "I love the new phone",
    "This movie is terrible",
    "What a fantastic restaurant!",
    "The product broke after one use",
]
y = [1, 0, 1, 0]  # 1: positive, 0: negative

# Tokenization again
tokenized_X = [nltk.word_tokenize(sentence.lower()) for sentence in X]

# Create embeddings for each sentence by averaging embeddings for individual words
def get_sentence_vector(sentence):
    vector = []
    for word in sentence:
        if word in model.wv:
            vector.append(model.wv[word])
    return np.mean(vector, axis=0)

sentence_vectors = [get_sentence_vector(sentence) for sentence in tokenized_X]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, y, test_size=0.25)

# Training a logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Making predictions on the test set
predictions = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of the sentiment analysis model:", accuracy)
