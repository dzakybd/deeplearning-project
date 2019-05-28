# Configurable parameter #
scenario = 1

if scenario == 1:
    rrn_type = 1
elif scenario == 2:
    rrn_type = 1
elif scenario == 3:
    rrn_type = 1
elif scenario == 4:
    rrn_type = 2
elif scenario == 5:
    rrn_type = 2


epochs = 10
# 128 for local; 512 for AWS
batch_size = 32
# the LSTM RNN will consider this many notes
sequence_length = 200
temperature = 1.0
offset_adj = 0.5
first_layer = 512
drop = 0.5


# Fixed parameter #
dataset_base = "../Dataset/"
midi_data = dataset_base + "midi/"
notes_data = dataset_base + 'notes.npy'


result_base = "../Result/"
scenario_path = result_base + "scenario-{}/".format(scenario)
preprocess_data = scenario_path + 'train/preprocess_data.npy'
model_saved = scenario_path + 'train/model_saved.h5'
model_plot = scenario_path + 'train/model_plot.png'
loss_graph = scenario_path + 'train/loss_graph.png'
loss_log = scenario_path + 'train/loss_log.csv'
summary = scenario_path + 'train/summary.txt'

generated_midi = scenario_path + "test/generated_music.mid"
generated_notes = scenario_path + "test/generated_music.npy"

