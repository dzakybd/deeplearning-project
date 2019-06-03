# Default parameter #

reset_preprocess = False
use_bidirectional = False
rrn_type = 'LSTM'
depth = 3
unit_size = 256
sequence_length = 100

use_regularizer = True
drop = 0.2
decay = 1e-4

epochs = 50
batch_size = 20
lr_rate = 0.01
offset_adj = 0.5

# Experiment parameter #

scenario = 1

if scenario == 1:
    pass
elif scenario == 2:
    depth = 2
elif scenario == 3:
    depth = 1
elif scenario == 4:
    rrn_type = 'GRU'
elif scenario == 5:
    rrn_type = 'GRU'
    depth = 2
elif scenario == 6:
    rrn_type = 'GRU'
    depth = 1
elif scenario == 7:
    reset_preprocess = True
    sequence_length = 200
elif scenario == 8:
    reset_preprocess = True
    sequence_length = 300
elif scenario == 9:
    rrn_type = 'GRU'
    reset_preprocess = True
    sequence_length = 200
elif scenario == 10:
    rrn_type = 'GRU'
    reset_preprocess = True
    sequence_length = 300

# Fixed parameter #
dataset_base = "../Dataset/"
midi_path = dataset_base + "midi/"

result_base = "../Result/"
preprocess_path = result_base + 'preprocess/'
scenario_path = result_base + "scenario-{}/".format(scenario)
train_path = scenario_path + 'train/'
test_path = scenario_path + 'test/'


def preprocess_data(instrument):
    return preprocess_path + '{}.npy'.format(instrument)


def preprocess_summary(instrument):
    return preprocess_path + '{}_summary.txt'.format(instrument)


def train_instrument_path(instrument, usage=0):
    path = train_path + '{}/'.format(instrument)
    if usage == 1:
        path = path + 'model_saved.h5'
    elif usage == 2:
        path = path + 'model_plot.png'
    elif usage == 3:
        path = path + 'loss_graph.png'
    elif usage == 4:
        path = path + 'train_log.csv'
    elif usage == 5:
        path = path + 'summary.txt'
    else:
        return path
    return path


def generated_midi(instrument):
    return test_path + "{}.mid".format(instrument)

