import re
from datetime import datetime


dataset_base = "../Dataset/"
gamelan_dataset = dataset_base + "gamelan/"

result_base = "../Result/"
training_result = result_base + "training/"
model_saved = training_result + 'model_saved.h5'
note_file = training_result + 'gamelan-input_notes'
history_file = training_result + 'gamelan-history.pkl'
output_file = training_result + 'gamelan-output_notes'

generated_music = result_base + "generated_music"

timestamp = re.sub(r'[-: ]','',str(datetime.now()).split('.')[0])[:-2]
output_name = midi_files.split('/')[-2]
total_epochs = 10
batch_size = 128 # 128 for local; 512 for AWS
sequence_length = 200 # the LSTM RNN will consider this many notes
notes_generated = 500
temperature = 1.0
offset_adj = 0.5

