!pip install tensorflow
!pip install numpy
!pip install pandas
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Loading the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Exploring the shape of the data
print(f'Train samples: {len(x_train)}, Test samples: {len(x_test)}')
