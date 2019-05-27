import sys
import pickle
import functions as fxn
from config import *
import matplotlib.pyplot as plt
from keras.utils import plot_model


def full_execution(midi_files, output_tag, total_epochs, batch_size, sequence_length, temperature, offset_adj):

    weight_file = None
    note_file = fxn.convert_midis_to_notes(midi_files, output_tag)
    epochs = total_epochs
    with open(note_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    network_input, network_output, n_patterns, n_vocab, pitchnames = fxn.prepare_sequences(notes, sequence_length)
    network_input_r, network_output_r = fxn.reshape_for_training(network_input, network_output,sequence_length)

    model = fxn.create_network(network_input_r, n_vocab, weight_file)
    model, weight_file, history = fxn.train_model(model, network_input_r, network_output_r, epochs, batch_size, output_tag, sequence_length)
    normalized_input = fxn.reshape_for_creation(network_input, n_patterns, sequence_length, n_vocab)
    model = fxn.create_network(normalized_input, n_vocab, weight_file)
    prediction_output= fxn.generate_notes(model, network_input, pitchnames,sequence_length, notes_generated, n_vocab, temperature)
    output_notes, midi_file = fxn.create_midi(prediction_output, output_tag, offset_adj)
    return model, history, weight_file



full_execution(midi_files, output_tag, total_epochs, batch_size, sequence_length, temperature, offset_adj)

