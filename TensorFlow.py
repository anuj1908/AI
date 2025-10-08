import tensorflow as tf
import numpy as np     
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
emails = [
 "Buy cheap watches! Free shipping!",
 "Meeting for lunch today?",
 "Claim your prize! You've won $1,000,000!",
 "Important meeting at 3 PM.",
]
labels = [1, 0, 1, 0]
max_words = 1000
max_len = 50
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(emails)
sequences = tokenizer.texts_to_sequences(emails)
X_padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
model = tf.keras.Sequential([
 tf.keras.layers.Embedding(input_dim=max_words, output_dim=16,
input_length=max_len),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(16, activation='relu'),
 tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
training_data = np.array(X_padded)
training_labels = np.array(labels)
model.fit(training_data, training_labels, epochs=10) 
file_path = "Spam.txt"
with open(file_path, "r", encoding="utf-8") as file:
 sample_email_text = file.read()
sequences_sample = tokenizer.texts_to_sequences([sample_email_text])
sample_email_padded = pad_sequences(sequences_sample, maxlen=max_len,
padding="post", truncating="post")
prediction = model.predict(sample_email_padded)
threshold = 0.5
if prediction > threshold:
 print(f"Sample Email ('{file_path}'): SPAM")
else:
 print(f"Sample Email ('{file_path}'): NOT SPAM")
