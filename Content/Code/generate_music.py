
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import plot_model

import functions as fxn
from config import *


# convert fully trained weights to midi file
def weights_to_midi(note_file, sequence_length, weight_file, temperature, offset_adj):
    with open(note_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    network_input, network_output, n_patterns, n_vocab, pitchnames = fxn.prepare_sequences(notes, sequence_length)
    normalized_input = fxn.reshape_for_creation(network_input, n_patterns, sequence_length, n_vocab)
    model = fxn.create_network(normalized_input, n_vocab, weight_file)
    prediction_output= fxn.generate_notes(model, network_input, pitchnames,sequence_length, notes_generated, n_vocab, temperature)
    output_notes, midi_file = fxn.create_midi(prediction_output, output_tag, offset_adj)
    return output_notes, model, midi_file

output_notes, model, midi_file = weights_to_midi(note_file, sequence_length, weight_file, temperature, offset_adj)

def myprint(s):
    with open(subfolder+'modelsummary.txt','w+') as f:
        print(s, file=f)

model.summary(print_fn=myprint)
plot_model(model, to_file=subfolder+'model.png')
history = pd.read_pickle(history_file)

# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig(subfolder+"plot.png")