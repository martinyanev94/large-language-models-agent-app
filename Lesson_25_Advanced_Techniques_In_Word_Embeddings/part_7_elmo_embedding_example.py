from allennlp.modules.elmo import Elmo

# Define parameters for ELMo
elmo = Elmo(options_file='path_to_elmo_options.json', weight_file='path_to_elmo_weights.hdf5', cuda=False)

# Example sentence
sentence = ["The cat sat on the mat."]
embeddings = elmo(sentence)

# You can get the sentence embeddings
sentence_embedding = embeddings['elmo_representations'][0]
