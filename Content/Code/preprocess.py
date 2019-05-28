import util
import numpy as np
from config import *
from keras.utils import np_utils
import os

def train_preprocess():

    # Convert MIDI to notes
    if os.path.exists(notes_data):
        notes = np.load(notes_data)
    else:
        notes = util.convert_midis_to_notes()

    # Create sequences
    train_x, train_y, _, n_vocab, _ = util.create_sequences(notes)

    # convert network input/output from lists to numpy arrays
    # reshape input to (notes less sequence length, sequence length)
    # reshape output to (notes less sequence length, unique notes/chords)
    # Data reshape
    train_x = np.reshape(train_x, (len(train_x), sequence_length, 1))
    train_y = np_utils.to_categorical(train_y)

    return train_x, train_y, n_vocab

def test_preprocess():

    train_x, _, _, n_vocab, pitchnames = util.load_sequences()
    # the network input variables below are unshaped (pre-reshaped)
    # Data reshape
    train_x = np.reshape(train_x, (len(train_x), sequence_length, 1)) / float(n_vocab)
    return train_x, n_vocab, pitchnames