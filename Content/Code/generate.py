from keras.models import load_model
from util import *
from config import *
from music21 import *

information = {}
for file in os.listdir(preprocess_path):
    if file.split('.')[-1] == 'npy':
        temp = np.load(preprocess_path + file)
        information[file.split('.')[0]] = temp

ins = []
for i in information:
    print("Start generating {} instrument".format(i))

    train_x = information[i][0]
    pitchnames = information[i][2]

    # Load model
    model = load_model(train_instrument_path(i, 1))

    prediction_output = generate_notes(model, train_x, pitchnames)

    # Convert to MIDI
    output_notes = create_midi(i, prediction_output)
    ins.append(output_notes)

    print("{} instrument created".format(i))
# Save midi & notes
midi_stream = stream.Stream(ins)
midi_stream.write('midi', fp=generated_midi(i))


