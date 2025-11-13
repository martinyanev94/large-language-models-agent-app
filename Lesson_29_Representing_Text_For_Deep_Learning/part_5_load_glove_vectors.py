import numpy as np

# Load GloVe vectors (ensure you have the file in the same directory)
def load_glove_model(glove_file):
    glove_model = {}
    with open(glove_file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    return glove_model

glove_model = load_glove_model('glove.6B.100d.txt')  # Make sure you have the .txt file accessible
print(glove_model['python'])  # Prints the GloVe vector for the word 'python'
