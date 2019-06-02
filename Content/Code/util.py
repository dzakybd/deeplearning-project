import os
import numpy as np
from music21 import *
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Dropout, CuDNNLSTM, Activation
from keras.callbacks import CSVLogger
from config import *
import matplotlib.pyplot as plt
from keras.utils import np_utils
from fractions import Fraction

# Preprocessing #
def train_preprocess():
    # Convert MIDI to notes
    information = {}
    instrus = {}
    if len(os.listdir(notes_path)) == 0:
        instrus = convert_midis_to_notes()
    else:
        for file in os.listdir(notes_path):
            temp = np.load(notes_path+file)
            instrus[file.split('.')[0]] = temp

    for i in instrus:

        # Create sequences
        train_x, train_y, n_patterns, n_vocab, pitchnames = create_sequences(instrus[i])

        # Data reshape
        train_x = np.reshape(train_x, (len(train_x), sequence_length, 1))
        train_y = np_utils.to_categorical(train_y)

        print("train_x", np.shape(train_x))
        print("train_y", np.shape(train_y))

        information[i] = [train_x, train_y, n_patterns, n_vocab, pitchnames]
        np.save(preprocess_data(i), information[i])

    return information

# Convert midi file dataset to notes
def convert_midis_to_notes():
    print("Convert MIDIs to notes")
    instrus = {}
    for file in os.listdir(midi_path):
        print(file)
        midi = converter.parse(midi_path + file)
        parts = instrument.partitionByInstrument(midi)
        for i in parts.parts:
            name = (str(i).split(' ')[-1])[:-1]
            notes_to_parse = i.recurse()
            length = len(notes_to_parse)
            if name[:2] == '0x':
                # name = 'unknown'
                continue
            elif length >= sequence_length:
                notes = []
                seqs = i.recurse()
                for element in seqs:
                    if isinstance(element, note.Note):
                        temp = str(element.pitch.nameWithOctave)+"|"+str(element.quarterLength)
                        notes.append(temp)
                    elif isinstance(element, chord.Chord):
                        pitches = []
                        for p1 in element.pitches:
                            p2 = str(p1.nameWithOctave)
                            pitches.append(p2)
                        pitch_names = '.'.join(n for n in set(pitches))
                        temp = pitch_names+"|"+str(element.quarterLength)
                        notes.append(temp)
                    elif isinstance(element, note.Rest):
                        temp = element.name + "|" + str(element.quarterLength)
                        notes.append(temp)

                if not len(notes) == 0:
                    if name in instrus.keys():
                        temp = instrus.get(name)
                        temp.extend(notes)
                    else:
                        temp = notes
                    instrus[name] = temp

    new_intrus = {}
    top_k = 5
    for i in sorted(instrus, key=lambda k: len(instrus[k]), reverse=True):
        top_k -= 1
        new_intrus[i] = instrus[i]

        print("Instrument {}, total {} notes with {} unique notes".format(i, len(instrus[i]), len(set(instrus[i]))))

        # Save notes
        np.save(notes_data(i), instrus[i])

        if top_k == 0:
            break

    names = set(new_intrus.keys())
    print("Total instrument {}. They are {}".format(len(names), names))

    return new_intrus


def create_sequences(notes):
    print("\n**Preparing sequences for training**")
    # list of unique chords and notes
    # pitchnames = sorted(set(i for i in notes))
    pitchnames = sorted(set(i for i in notes))

    n_vocab = len(pitchnames)

    print('n_vocab', n_vocab)
    print("Pitchnames (unique notes/chords from 'notes') at length {}".format(len(pitchnames)))
    # enumerate pitchnames into dictionary embedding
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

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

    print("Network input and output created with (pre-transform) lengths {}".format(len(train_x)))
    # print("Network input and output first list items: {} and {}".format(train_x[0],train_y[0]))
    # print("Network input list item length: {}".format(len(train_x[0])))
    n_patterns = len(train_x) # notes less sequence length
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

def generate_notes(model, train_x, n_vocab, pitchnames):
    print("\n**Generating notes**")
    # convert integers back to notes/chords
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    # randomly instantiate with single number from 0 to length of network input
    start = np.random.randint(0, len(train_x) - 1)
    pattern = train_x[start]
    pattern = np.squeeze(pattern).tolist()
    prediction_output = []
    # generated notes (ie 500)
    for note_index in range(sequence_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)[0]
        index = sample(prediction, temperature)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(i, prediction_output):
    count = 0
    print("\n**Creating midi**")
    output_notes = []
    offset = 0
    for pattern in prediction_output:
        # prepares chords (if) and notes (else)
        output_notes.append(instrument.fromString(i))
        notes = pattern.split("|")[0]
        duration = pattern.split("|")[-1]
        if '/' in duration:
            duration = round(float(Fraction(duration)), 2)
        else:
            duration = round(float(duration), 2)
        print(notes, duration)
        if '.' in notes:
            notes_in_chord = notes.split('.')
            n = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.fromString(i)
                n.append(new_note)
            new_chord = chord.Chord(n, quarterLength=duration)
            # new_chord.offset = offset
            output_notes.append(new_chord)
        elif notes == 'rest':
            new_note = note.Rest(quarterLength=duration)
            new_note.storedInstrument = instrument.fromString(i)
            # new_note.offset = offset
            output_notes.append(new_note)
        else:
            new_note = note.Note(notes, quarterLength=duration)
            new_note.storedInstrument = instrument.fromString(i)
            # new_note.offset = offset
            output_notes.append(new_note)
        # offset +=offset_adj

        # Save midi & notes
        midi_stream = stream.Stream(output_notes)
        # midi_stream.write('midi', fp=generated_midi(i))
        mid = midi.translate.streamToMidiFile(midi_stream)
        mid.open('out'+str(count)+'.mid', 'wb')
        mid.write()
        mid.close()
        count+=1

    return output_notes


def callback_builder(i):
    callbacks_list = []
    callbacks_list.append(CSVLogger(train_instrument_path(i, 4), separator=','))
    return callbacks_list


# fungsi untuk membuat plot akurasi dan loss tiap epoch
def create_plot(i, hist):
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
    plt.savefig(train_instrument_path(i, 3))