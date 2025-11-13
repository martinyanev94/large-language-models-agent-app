import numpy as np

# Sample document
document = """
Artificial intelligence is the simulation of human intelligence processes by machines,
especially computer systems. These processes include learning (the acquisition of information
and rules for using it), reasoning (using the rules to reach approximate or definite conclusions),
and self-correction. Specific applications of AI include expert systems, natural language processing, and speech recognition.
"""

# Split document into sentences
sentences = document.split('. ')
tfidf_sentence_matrix = vectorizer.fit_transform(sentences)

# Summarizing based on the maximum TF-IDF score for each sentence
sentence_scores = np.array(tfidf_sentence_matrix.sum(axis=1)).flatten()
ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[::-1]]

# Selecting top n sentences for summary
n = 2
summary = ' '.join(ranked_sentences[:n])
print("Generated Summary:")
print(summary)
