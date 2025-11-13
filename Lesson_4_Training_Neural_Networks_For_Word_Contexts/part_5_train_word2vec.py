def train_word2vec(sentences):
    for sentence in sentences:
        for word in sentence:
            context_words = get_context(sentence, word)
            predicted = model.predict(word)
            loss = calculate_loss(predicted, context_words)
            model.update_weights(loss)

def get_context(sentence, word):
    return [w for w in sentence if w != word]

def calculate_loss(predicted, actual):
    return some_loss_function(predicted, actual)

def update_weights(loss):
    # Perform backpropagation
    pass
