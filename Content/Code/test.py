from preprocess import test_preprocess
from keras.models import load_model
from util import *
from config import *

train_x, n_vocab, pitchnames = test_preprocess()
model = load_model(model_saved)

print("\n**Generating notes**")
# convert integers back to notes/chords
print("Length Pitchnames: {}".format(len(pitchnames)))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
print("Integer to note conversion at length: {}".format(len(int_to_note)))
# randomly instantiate with single number from 0 to length of network input
print("Network input length: {}".format(len(train_x)))
start = np.random.randint(0, len(train_x) - 1)
pattern = train_x[start]
pattern = np.squeeze(pattern).tolist()
prediction_output = []
# # for each note in notes generated declared as hyperparameter above (ie 500)
for note_index in range(sequence_length):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)[0]
    index = sample(prediction, temperature)
    result = int_to_note[index]

    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("Pattern ends with length {} and type {}".format(len(pattern), type(pattern)))
print("Generated Note Length: {}\nFirst 100: {}".format(len(prediction_output), prediction_output[:100]))

output_notes = create_midi(prediction_output)
print("Generating {} notes stored as {}".format(len(output_notes), type(output_notes)))
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp=generated_midi)
print("Midi saved at: {}".format(generated_midi))
np.save(generated_notes, output_notes)
print("Output notes/chords stored as {} then pickled at {}".format(type(output_notes), generated_notes))
