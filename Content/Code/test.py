from keras.models import load_model
from util import *
from config import *
from music21 import stream

# Load model
model = load_model(model_saved)

# Generate notes
train_x, _, _, n_vocab, pitchnames = load_sequences()
# train_x = train_x / float(n_vocab)
prediction_output = generate_notes(model, train_x, n_vocab, pitchnames)

# Convert to MIDI
print("Generated Note Length: {}\nFirst 10: {}".format(len(prediction_output), prediction_output[:10]))
output_notes = create_midi(prediction_output)

# Save midi & notes
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp=generated_midi)
np.save(generated_notes, output_notes)
print("Midi saved at: {}\nOutput notes/chords at {}".format(generated_midi, generated_notes))