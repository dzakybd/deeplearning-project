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
batch_size = 32
sequence_length = 200
temperature = 1.0
offset_adj = 0.5
first_layer = 512
drop = 0.5

# Fixed parameter #
dataset_base = "../Dataset/"
midi_path = dataset_base + "midi/"
notes_path = dataset_base + "notes/"


def notes_data(instrument):
    return notes_path + '{}.npy'.format(instrument)

result_base = "../Result/"
scenario_path = result_base + "scenario-{}/".format(scenario)
preprocess_path = scenario_path + 'preprocess/'
train_path = scenario_path + 'train/'


def preprocess_data(instrument):
    return preprocess_path + '{}.npy'.format(instrument)


def train_instrument_path(instrument, usage=0):
    path = train_path + '{}/'.format(instrument)
    if usage == 1:
        path = path + 'model_saved.h5'
    elif usage == 2:
        path = path + 'model_plot.png'
    elif usage == 3:
        path = path + 'loss_graph.png'
    elif usage == 4:
        path = path + 'loss_log.csv'
    elif usage == 5:
        path = path + 'summary.txt'
    else:
        return path
    return path


def generated_midi(instrument):
    return scenario_path + "test/{}.mid".format(instrument)


def generated_notes(instrument):
    return scenario_path + "test/{}.npy".format(instrument)

