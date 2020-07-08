import numpy as np 
import tensorflow as tf
import pandas
import sys
import json
import h5py

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pdb

target_name_dict = {'stat.AP' : 0,
                    'stat.CO' : 1,
                    'stat.ME' : 2,
                    'stat.ML' : 3,
                    'stat.OT' : 4
                    }

label2target = { v:k for k,v in target_name_dict.items()}

files = ["../data/2015ml.h5",
         "../data/2016ml.h5",
         "../data/2017ml.h5",
         "../data/2018ml.h5",
         "../data/2019ml.h5",
        ]


abstracts = []
labels = []
for f in files:

    store = pandas.HDFStore(f)
    #import pdb
    #pdb.set_trace()
    df = store['/df']
    store.close()

    abstracts += list(df['abstract'])
    labels = np.hstack([labels,np.array(df['categories'])])


labels = np.asarray([item[0] for item in labels.tolist()])


selected_labels = ['stat.AP', 'stat.CO', 'stat.ME', 'stat.ML', 'stat.OT']


labels_selected = np.asarray([item for item in labels.tolist() if item in selected_labels])


jj = 0 
abstracts_selected = []



for item in labels.tolist(): 
    if item in selected_labels:
        abstracts_selected.append(abstracts[jj])
        jj = jj + 1
        
abstracts_selected = np.asarray(abstracts_selected)


print (np.unique(labels_selected))
print("---------")


labels = labels_selected



for i in range(2):
    print(abstracts[i])
    print(target_name_dict[labels[i]])
    print("---------")


num_words = 10000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(abstracts)
sequences = tokenizer.texts_to_sequences(abstracts)
seq = pad_sequences(sequences, padding='post', value=0, maxlen=100)



# Tokenizers come with a convenient list of words and IDs
dictionary = tokenizer.word_index
# Let's save this out so we can use it later
with open('dictionary_ML.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)



np.random.seed(1234)
ind = np.random.randint(0, len(labels), len(labels))
print(ind.shape)
labels = labels[ind]
seq = seq[ind,:]




split_1 = int(0.8 * len(labels))
split_2 = int(0.9 * len(labels))
train_labels = labels[:split_1]
dev_labels = labels[split_1:split_2]
test_labels = labels[split_2:]

train_seq = seq[:split_1, :]
dev_seq = seq[split_1:split_2, :]
test_seq = seq[split_2:, :]


#%%
vocab_size = 10000


#%%
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 64))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(5, activation=tf.nn.sigmoid))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


y_train_num = np.asarray([target_name_dict[x] for x in train_labels.tolist()])
y_test_num = np.asarray([target_name_dict[x] for x in test_labels.tolist()])

train_labels_onehot = to_categorical(y_train_num)
test_labels_onehot = to_categorical(y_test_num)

history = model.fit(train_seq, train_labels_onehot, epochs=10, steps_per_epoch=32, validation_split=0.3, validation_steps=32)


#%%
ev  = model.evaluate(test_seq, test_labels_onehot, steps=10)
print(ev)


#%%
text = ["we present high dispersion spectroscopic data of the compact planetary nebula vy 1 2 where high expansion velocities up to 100 km s are found in the ha n ii and o iii emission lines hst images reveal a bipolar structure vy 1 2 displays a bright ring like structure with a size of 2 4 2 and two faint bipolar lobes in the west east direction a faint pair of knots is also found located almost on opposite sides of the nebula at pa degrees furthermore deep low dispersion spectra are also presented and several emission lines are detected for the first time in this nebula such as the doublet cl iii a k iv a c ii 6461 a the doublet c iv 5801 5812 a by comparison with the solar abundances we find enhanced n depleted c and solar o the central star must have experienced the hot bottom burning cn cycle during the 2nd dredge up phase implying a progenitor star of higher than 3 solar masses the ver"]


#%%
seq_1 = tokenizer.texts_to_sequences(text)


#%%
seq_2 = pad_sequences(seq_1, padding='post', value=0, maxlen=350)


#%%
prob = model.predict(seq_2)
prob /= prob.sum()
print(prob)
ii = np.argmax(prob)
print(label2target[ii])



# serialize model to JSON
model_json = model.to_json()
with open("model_ML.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save('model_ML.h5')

print("Saved model to disk")






