from keras.models import load_model
from util import *
from config import *
from music21 import *

# Load preprocessed data
information = {}
for file in os.listdir(preprocess_path):
    if file.split('.')[-1] == 'npy':
        temp = np.load(preprocess_path + file)
        information[file.split('.')[0]] = temp

all_instrument_notes = []

for i in information:
    print("Start generating {} instrument".format(i))

    # Load model
    model = load_model(train_instrument_path(i, 1))

    # Predict output notes
    train_x = information[i][0]
    pitchnames = information[i][2]
    prediction_output = generate_notes(model, train_x, pitchnames)

    # Convert to MIDI
    output_notes = create_midi(i, prediction_output)
    all_instrument_notes.append(output_notes)

    # Write MIDI for each instrument
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=generated_midi(i))

    print("{} instrument created".format(i))

# Write MIDI for combined instrument
midi_stream = stream.Stream(all_instrument_notes)
midi_stream.write('midi', fp=generated_midi("Combined"))



