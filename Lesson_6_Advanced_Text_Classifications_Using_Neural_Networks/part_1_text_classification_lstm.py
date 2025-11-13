import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample dataset creation
data = {
    'text': [
        'Economic growth is crucial for overall development.',
        'New policies are impacting the political landscape.',
        'Climate change affects everyone, regardless of borders.',
        'Technological advances are shaping the future of work.',
        'Elections are crucial for a healthy democracy.'
    ],
    'labels': [
        [1, 0, 1, 0], # Economy, Politics, Environment, Technology
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ]
}
df = pd.DataFrame(data)

# Preparing the data
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

# Padding the sequences
X = pad_sequences(sequences)
y = np.array(df['labels'].tolist())
