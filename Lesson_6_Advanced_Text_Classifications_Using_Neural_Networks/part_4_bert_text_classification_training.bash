pip install transformers
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Set up the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Tokenizing the input data
inputs = tokenizer(df['text'].tolist(), max_length=128, padding=True, truncation=True, return_tensors='tf')
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history_bert = model.fit(inputs['input_ids'], y_train, validation_data=(inputs['input_ids'], y_test), epochs=3, batch_size=2)
plt.plot(history_bert.history['accuracy'], label='Training Accuracy')
plt.plot(history_bert.history['val_accuracy'], label='Validation Accuracy')
plt.title('BERT Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
