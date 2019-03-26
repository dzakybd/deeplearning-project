import re
from datetime import datetime

genre = "gamelan"
midi_files = 'music/{}/*.mid'.format(genre)

timestamp = re.sub(r'[-: ]','',str(datetime.now()).split('.')[0])[:-2]
output_name = midi_files.split('/')[-2]
total_epochs = 10
batch_size = 128 # 128 for local; 512 for AWS
sequence_length = 200 # the LSTM RNN will consider this many notes
notes_generated = 500
temperature = 1.0
offset_adj = 0.5

subfolder = 'output/'
output_tag = '{}{}-{}-'.format(subfolder, timestamp, output_name)

weight_file = subfolder + '201903261638-gamelan--last_weights.hdf5'
note_file = subfolder + '201903261638-gamelan-input_notes'
history_file = subfolder + '201903261638-gamelan-history.pkl'
output_file = subfolder + '201903261638-gamelan-output_notes'