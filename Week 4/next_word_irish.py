import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

tokenizer = Tokenizer()

data = "In the town of Athy one Jeremy Lanigan \n battered away till he hadn't a pound \n his father he died and made him a man again \n left a farm with ten acres of ground \n he gave a grand party for friends a relations \n who did not forget him when come to the will \n and if you'll but listen I'll make you're eyes glisten \n of rows and ructions at Lanigan's Ball \n \n six long months I spent in Dub-i-lin \n six long months doing nothin' at all \n six long months I spent in Dub-i-lin \n learning to dance for Lanigan's Ball \n I stepped out I stepped in again \n I stepped out I stepped in again \n I stepped out I stepped in again \n learning to dance for Lanigan's Ball \n Myself to be sure got free invitaions \n for all the nice boys and girls I did ask \n in less than 10 minutes the friends and relations \n were dancing as merry as bee 'round a cask \n There was lashing of punch and wine for the ladies \n potatoes and cakes there was bacon a tay \n there were the O'Shaughnessys, Murphys, Walshes, O'Gradys \n courtin' the girls and dancing away \n they were doing all kinds of nonsensical polkas \n all 'round the room in a whirly gig \n but Julia and I soon banished their nonsense \n and tipped them a twist of a real Irish jig \n Oh how that girl got mad on me \n and danced till you'd think the ceilings would fall \n for I spent three weeks at Brook's academy \n learning to dance for Lanigan's Ball \n The boys were all merry the girls were all hearty \n dancing away in couples and groups \n till an accident happened young Terrance McCarthy \n put his right leg through Miss Finerty's hoops \n The creature she fainted and cried 'melia murder' \n cried for her brothers and gathered them all \n Carmody swore that he'd go no further \n till he'd have satisfaction at Lanigan's Ball \n In the midst of the row Miss Kerrigan fainted \n her cheeks at the same time as red as a rose \n some of the boys decreed she was painted \n she took a wee drop too much I suppose \n Her sweetheart Ned Morgan all powerful and able \n when he saw his fair colleen stretched out by the wall \n he tore the left leg from under the table \n and smashed all the dishes at Lanigan's Ball \n Boy oh Boys tis then there was ructions \n myself got a kick from big Phelam McHugh \n but soon I replied to this kind introduction \n and kicked up a terrible hullaballoo \n old Casey the piper was near being strangled \n they squeezed up his pipes bellows chanters and all \n the girls in their ribbons they all got entangled \n and that put an end to Lanigan's Ball"

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print(tokenizer.word_index['in'])
print(tokenizer.word_index['the'])
print(tokenizer.word_index['town'])
print(tokenizer.word_index['of'])
print(tokenizer.word_index['athy'])
print(tokenizer.word_index['one'])
print(tokenizer.word_index['jeremy'])
print(tokenizer.word_index['lanigan'])

print(xs[6])

print(xs[5])
print(ys[5])

print(tokenizer.word_index)

model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # adam suele ser bueno para categorical
history = model.fit(xs, ys, epochs=500, verbose=1)


import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


plot_graphs(history, 'accuracy')

seed_text = "Laurence went to dublin"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
