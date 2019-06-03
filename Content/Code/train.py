from util import *
from config import *
from keras.utils import plot_model
from keras import backend as K
import time

information = {}
if len(os.listdir(preprocess_path)) == 0:
    print("Start preprocessing")
    information = train_preprocess()
else:
    print("Load preprocessed data")
    for file in os.listdir(preprocess_path):
        if file.split('.')[-1] == 'npy':
            temp = np.load(preprocess_path + file)
            information[file.split('.')[0]] = temp

for i in information:
    print('Start training on {} instrument'.format(i))
    train_x = information[i][0]
    train_y = information[i][1]
    pitchnames = information[i][2]

    # Train sequences set normalize
    train_x = train_x / float(len(pitchnames))

    if not os.path.exists(train_instrument_path(i)):
        os.makedirs(train_instrument_path(i))

    # Training
    model = get_model(len(pitchnames))
    model.summary()
    start = time.time()
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, callbacks=callback_builder(i))
    end = time.time()

    # Save model
    plot_model(model, to_file=train_instrument_path(i, 2), show_shapes=True, show_layer_names=True)
    create_plot(i, history, 'loss')
    model.save(train_instrument_path(i, 1))

    # Clear model
    del model
    K.clear_session()

    # Write training summary
    f = open(train_instrument_path(i, 5), 'w')
    info = ""
    info += 'Instrument: ' + i
    info += '\nScenario: ' + str(scenario)
    info += '\nRNN type: ' + rrn_type
    info += '\nDepth: ' + str(depth)
    if use_nietsche_sample:
        info += '\nUse nietsche sampling: ' + 'Yes / Temperature: ' + str(temperature)
    else:
        info += '\nUse nietsche sampling: No'
    if use_regularizer:
        info += '\nUse regularizer: ' + 'Yes / Drouput: ' + str(drop) +'/ Weight decay: ' + str(decay)
    else:
        info += '\nUse regularizer: No'

    info += '\n\nEpochs: ' + str(epochs)
    info += '\nBatch size: ' + str(batch_size)
    info += '\nOptimizer: SGD + momentum + lr ' + str(lr_rate)
    info += '\nSequence length ' + str(sequence_length)
    info += '\nRNN units: ' + str(unit_size)

    info += '\n\nTime interval: ' + str(end-start)

    print(info)
    f.write(info)
    f.close()

    print('End training on {} instrument'.format(i))