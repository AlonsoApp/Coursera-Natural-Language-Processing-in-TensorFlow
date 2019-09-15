import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(l.numpy())


training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

# Preparamos el tokenizer

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

# Transformamos las frases de su version codificada con el tokenizer de vuelta a lenguaje normal con solo las palabras
# del word_index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(padded[1]))
print(training_sentences[1])

# Creamos la red con una primera capa de Embedding que trasnforma las palabras en vectores de tamanyo embedding_dim
# y max_length de tamaño de frase (16 x 120)
# luego usamos Flatten para colocar esa matriz 2D 16 x 120 en una 1D 1920 para meterla de input a la Dense
# en vez de Flatten podemos usar una capa GlobalAveragePooling1D() que nos daria una capa de 16 (la media de los
# embeddings de la frase)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

# Los pesos de la ultima capa van a ser los valores de los vectores de 16 para las 10000 palabras que tiene vocab_size
# Ahora ya tenemos los embeddings de nuestro vocabulario
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)

# Ahora cogemos y por cada palabra, sacamos los 16 pesos asociados al index de cada palabra y los exportamos

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')  # vectores embedding de cada palabra del diccionario
out_m = io.open('meta.tsv', 'w', encoding='utf-8')  # las palabras del diccionario ordenadas igual que los vectores
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# Para visualizar los datos de meta.tsv y vecs.tsv ir a https://projector.tensorflow.org cargar los dos archivos y
# pulsar Sphereize data Usa la funcion Search a la derecha y flipa
