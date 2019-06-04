import os
import numpy as np
from music21 import *
from keras.models import Model
from keras.layers import Dense, Dropout, CuDNNLSTM, Activation, CuDNNGRU, Bidirectional, Input, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from config import *
import matplotlib.pyplot as plt
from keras.utils import np_utils


# Preprocessing #
def train_preprocess():
    information = {}
    print('Convert MIDI to notes')
    all_instrument_notes = convert_midis_to_notes()

    for i in all_instrument_notes:

        print('Create sequences')
        # Create sequences
        train_x, train_y, pitchnames = create_sequences(all_instrument_notes[i])

        print('Data reshape')
        # Data reshape
        train_x = np.reshape(train_x, (len(train_x), sequence_length, 1))
        train_y = np_utils.to_categorical(train_y, len(pitchnames))

        information[i] = [train_x, train_y, pitchnames]
        np.save(preprocess_data(i), information[i])

        # Write preprocess summary
        f = open(preprocess_summary(i), 'w')
        info = ""
        info += 'Instrument: ' + i
        info += '\nData size: ' + str(len(train_x))
        info += '\nSequence out data: ' + str(len(train_y))
        info += '\nSequence length: ' + str(train_x.shape[-2])
        info += '\nNumber of unique notes: ' + str(len(pitchnames))
        info += '\nSample unique note: ' + pitchnames[0]
        print(info)
        f.write(info)
        f.close()

    return information

# Convert midi file dataset to notes
def convert_midis_to_notes():
    all_instrument_notes = {}
    used_inst = [
        'Celesta',
        'Bass',
        'Flute',
        'Sitar',
        'Marimba'
    ]
    for file in os.listdir(midi_path):
        print(file)
        midi = converter.parse(midi_path + file)
        parts = instrument.partitionByInstrument(midi)
        for i in parts.parts:
            name = (str(i).split(' ')[-1])[:-1]
            if name not in used_inst:
                continue
            else:
                notes = []
                seqs = i.recurse()
                prev_offset = 0
                offset_dif = 0

                for element in seqs:
                    temp = ""
                    duration = round(float(element.quarterLength), 2)
                    if prev_offset != element.offset:
                        offset_dif = round(float(element.offset - prev_offset), 2)
                    if element.quarterLength == 0.0:
                        continue
                    elif isinstance(element, note.Note):
                        temp = str(element.pitch.nameWithOctave)
                        temp += "|"+str(duration)
                        temp += "|"+str(offset_dif)
                        notes.append(temp)
                    elif isinstance(element, chord.Chord):
                        pitches = []
                        for p1 in element.pitches:
                            p2 = str(p1.nameWithOctave)
                            pitches.append(p2)
                        pitches = set(pitches)
                        pitch_names = '.'.join(str(n) for n in pitches)
                        temp = pitch_names
                        temp += "|" + str(duration)
                        temp += "|" + str(offset_dif)
                        notes.append(temp)
                    elif isinstance(element, note.Rest):
                        temp = element.name
                        temp += "|" + str(duration)
                        temp += "|" + str(offset_dif)
                        notes.append(temp)
                    prev_offset = element.offset
                    # print(temp)
                if not len(notes) == 0:
                    if name in all_instrument_notes.keys():
                        temp = all_instrument_notes.get(name)
                        temp.extend(notes)
                    else:
                        temp = notes
                    all_instrument_notes[name] = temp

    for i in all_instrument_notes.keys():
        print("Instrument {} have {} unique notes".format(i, len(all_instrument_notes[i])))

    return all_instrument_notes


def create_sequences(notes):
    # list of unique chords and notes
    pitchnames = sorted(set(i for i in notes))

    # enumerate pitchnames into dictionary embedding
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    train_x = []
    train_y = []

    # i equals total notes less declared sequence length of LSTM (ie 5000 - 100)
    # sequence input for each i is list of notes i to end of sequence length (ie 0-100 for i = 0)
    # sequence output for each i is single note at i + sequence length (ie 100 for i = 0)
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        input_add = [note_to_int[char] for char in sequence_in]
        train_x.append(input_add)
        output_add = note_to_int[sequence_out]
        train_y.append(output_add)

    return train_x, train_y, pitchnames


def get_model(n_vocab):

    if rrn_type == 'LSTM':
        RNNLayer = CuDNNLSTM
    else:
        RNNLayer = CuDNNGRU

    model_input = Input(shape=(sequence_length, 1))

    for i in range(depth):

        if use_regularizer:
            regularizer = {'kernel_regularizer':l2(decay),
                           'recurrent_regularizer':l2(decay),
                           'bias_regularizer':l2(decay)}
        else:
            regularizer = {}

        if use_bidirectional:
            x = Bidirectional(RNNLayer(unit_size, return_sequences=False if i == depth-1 else True,
                                       kernel_initializer='he_normal', **regularizer))(model_input if i==0 else x)
        else:
            x = RNNLayer(unit_size, return_sequences=False if i == depth - 1 else True,
                         kernel_initializer='he_normal', **regularizer)(model_input if i == 0 else x)

        if use_regularizer:
            x = BatchNormalization()(x)
            x = Dropout(drop)(x)

    x = Dense(n_vocab)(x)
    model_output = Activation('softmax')(x)

    model = Model(model_input, model_output)
    optimizer = SGD(lr=lr_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def generate_notes(model, train_x, pitchnames):
    # convert integers back to notes/chords
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    n_vocab = len(pitchnames)
    # randomly instantiate with single number from 0 to length of network input
    start = np.random.randint(0, len(train_x) - 1)
    pattern = train_x[start]
    pattern = np.squeeze(pattern).tolist()
    prediction_output = []
    for note_index in range(sequence_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)[0]

        index = np.argmax(prediction)
        # index = sample(prediction, 1)

        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


# Convert notes to MIDI
def create_midi(i, prediction_output):
    output_notes = stream.Part()
    offset = 0
    output_notes.append(instrument.fromString(i))
    for pattern in prediction_output:
        notes = pattern.split("|")[0]
        # note_duration = round(float(pattern.split("|")[1]), 2)
        note_offset = round(float(pattern.split("|")[2]), 2)
        offset += note_offset
        # print(notes, note_duration, note_offset)
        if '.' in notes:
            notes_in_chord = notes.split('.')
            n = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.fromString(i)
                n.append(new_note)
            new_chord = chord.Chord(n)
            # new_chord.quarterLength = note_duration
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif notes == 'rest':
            new_note = note.Rest()
            # new_note.quarterLength = note_duration
            new_note.storedInstrument = instrument.fromString(i)
            new_note.offset = offset
            output_notes.append(new_note)
        else:
            new_note = note.Note(notes)
            new_note.storedInstrument = instrument.fromString(i)
            # new_note.quarterLength = note_duration
            new_note.offset = offset
            output_notes.append(new_note)

    return output_notes


# Callback for loss logging
def callback_builder(i):
    callbacks_list = []
    callbacks_list.append(CSVLogger(train_instrument_path(i, 4), separator=','))
    return callbacks_list


# Temperature-based event disribution
def sample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Create plot
def create_plot(i, hist, title):
    xc = range(epochs)
    a = hist.history[title]
    plt.figure()
    plt.plot(xc, a)
    plt.xlabel('epoch')
    plt.ylabel(title)
    plt.title('train_'+title)
    plt.grid(True)
    plt.savefig(train_instrument_path(i, 3))

