from keras.models import load_model
from util import *
from config import *
from music21 import stream
import glob

information = {}
for file in os.listdir(preprocess_path):
    temp = np.load(preprocess_path + file)
    information[file.split('.')[0]] = temp

for i in information:
    print("Generate on {}".format(i))

    train_x = information[i][0]
    n_vocab = information[i][3]
    pitchnames = information[i][4]
    # train_x = train_x / float(n_vocab)

    # Load model
    model = load_model(train_instrument_path(i, 1))

    prediction_output = generate_notes(model, train_x, n_vocab, pitchnames)
    # print("prediction_output", prediction_output)

    # Convert to MIDI
    print("Generated Note Length: {}\nFirst 10: {}".format(len(prediction_output), prediction_output[:10]))
    output_notes = create_midi(i, prediction_output)
    # print("output_notes", output_notes)

    # Save midi & notes
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=generated_midi(i))
    np.save(generated_notes(i), output_notes)
    print("Midi saved at: {}\nOutput notes/chords at {}".format(generated_midi(i), generated_notes(i)))