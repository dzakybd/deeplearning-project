import os
import numpy as np
from music21 import converter, note, chord, stream, instrument
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Dropout, CuDNNLSTM, Activation
from keras.callbacks import CSVLogger
from config import *
import matplotlib.pyplot as plt

# Preprocessing #

# Convert midi file dataset to notes
def convert_midis_to_notes():
    print("Convert MIDIs to notes")
    # list of notes and chords
    notes = []
    note_count = 0

    # loading midi filepaths
    for file in os.listdir(midi_data):
        print(file)
        try:
            # midi type music21.stream.Score
            midi = converter.parse(midi_data+file)
            parts = instrument.partitionByInstrument(midi)

            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            # notes_to_parse type music21.stream.iterator.RecursiveIterator
            for e in notes_to_parse:
                if isinstance(e, note.Note):
                    notes.append(str(e.pitch))
                elif isinstance(e, chord.Chord):
                    to_append = '.'.join(str(n) for n in e.normalOrder)
                    notes.append(to_append)
            note_count += 1
        except Exception as e:
            print(e)
            pass
    assert note_count > 0
    n_vocab = len(set(notes))
    print("Loaded {} midi files {} notes and {} unique notes".format(note_count, len(notes), n_vocab))
    np.save(notes_data, notes)
    print("Input notes/chords stored as {} then pickled at {}".format(type(notes), notes_data))
    print("First 20 notes/chords: {}".format(notes[:20]))

    return notes

def load_sequences():
    information = np.load(preprocess_data)
    train_x = information[0]
    train_y = information[1]
    n_patterns = information[2]
    n_vocab = information[3]
    pitchnames = information[4]
    return train_x, train_y, n_patterns, n_vocab, pitchnames

def create_sequences(notes):
    print("\n**Preparing sequences for training**")
    pitchnames = sorted(set(i for i in notes)) # list of unique chords and notes
    n_vocab = len(pitchnames)
    print("Pitchnames (unique notes/chords from 'notes') at length {}: {}".format(len(pitchnames), pitchnames))
    # enumerate pitchnames into dictionary embedding
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print("Note to integer embedding created at length {}".format(len(note_to_int)))

    train_x = []
    train_y = []

    # i equals total notes less declared sequence length of LSTM (ie 5000 - 100)
    # sequence input for each i is list of notes i to end of sequence length (ie 0-100 for i = 0)
    # sequence output for each i is single note at i + sequence length (ie 100 for i = 0)
    for i in range(0, len(notes) - sequence_length,1):
        sequence_in = notes[i:i + sequence_length] # 100
        sequence_out = notes[i + sequence_length] # 1

        # enumerate notes and chord sequences with note_to_int enumerated encoding
        # network input/output is a list of encoded notes and chords based on note_to_int encoding
        # if 100 unique notes/chords, the encoding will be between 0-100
        input_add = [note_to_int[char] for char in sequence_in]
        train_x.append(input_add) # sequence length
        output_add = note_to_int[sequence_out]
        train_y.append(output_add) # single note

    print("Network input and output created with (pre-transform) lengths {} and {}".format(len(train_x),len(train_y)))
    # print("Network input and output first list items: {} and {}".format(train_x[0],train_y[0]))
    # print("Network input list item length: {}".format(len(train_x[0])))
    n_patterns = len(train_x) # notes less sequence length
    print("Lengths. N Vocab: {} N Patterns: {} Pitchnames: {}".format(n_vocab,n_patterns, len(pitchnames)))

    information = [train_x, train_y, n_patterns, n_vocab, pitchnames]
    np.save(preprocess_data, information)

    return train_x, train_y, n_patterns, n_vocab, pitchnames


def get_model(train_x, n_vocab):
    print("\n**LSTM model initializing**")
    # this is a complete model file

    # network input shape (notes - sequence length, sequence_length, 1)
    timesteps = train_x.shape[1] # sequence length
    data_dim = train_x.shape[2] # 1

    print("Input nodes: {} Dropout: {}".format(first_layer, drop))
    print("Input shape (timesteps, data_dim): ({},{})".format(timesteps, data_dim))
    # for LSTM models, return_sequences sb True for all but the last LSTM layer
    # this will input the full sequence rather than a single value
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(first_layer), input_shape=(timesteps, data_dim)))
    model.add(Dense(first_layer))
    model.add(Dropout(drop))
    model.add(Dense(n_vocab)) # based on number of unique notes
    model.add(Dropout(drop))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# CREATE MIDI
# sample function from Keras Nietsche example
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def create_midi(prediction_output):
    print("\n**Creating midi**")
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # prepares chords (if) and notes (else)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Xylophone()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Xylophone()

            output_notes.append(new_note)
        offset += offset_adj #0.5

    return output_notes



def callback_builder():
    callbacks_list = []
    callbacks_list.append(CSVLogger(loss_log, separator=','))
    return callbacks_list

# fungsi untuk membuat plot akurasi dan loss tiap epoch
def create_plot(hist):
    xc = range(epochs)
    a = hist.history["loss"]
    b = hist.history['val_loss']
    plt.figure()
    plt.plot(xc, a)
    plt.plot(xc, b)
    plt.xlabel('epoch')
    plt.ylabel("loss")
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'test'])
    plt.savefig(loss_graph)