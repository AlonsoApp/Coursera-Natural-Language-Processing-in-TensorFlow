# La ideqa detras de esto es que normalmente los datasets viene ya tokenizados por otros datascientists

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder

# Explore the encoder (this is a tfds.features.text.SubwordTextEncoder)

sample_string = 'TensorFlow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))

# See the tokens themselves

for ts in tokenized_string:
    print('{} ----> {}'.format((ts, tokenizer.decode([ts]))))

# Let's train de model with this tokenizer

embedding_dim = 64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30

history = model.fit(train_data, epochs=num_epochs,
                    validation_data=test_data)

# Plot the results

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


plot_graphs(history, "acc")
plot_graphs(history, "loss")

# Esto no da buenos resultados porque: Well, the keys in the fact that we're using sub-words and not for-words,
# sub-word meanings are often nonsensical and it's only when we put them together in sequences that they have
# meaningful semantics. Thus, some way from learning from sequences would be a great way forward, and that's exactly
# what you're going to do next week with Recurrent Neural Networks

