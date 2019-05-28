from util import *
from config import *
from keras.utils import plot_model
from keras import backend as K


information = {}
if len(os.listdir(preprocess_path)) == 0:
    information = train_preprocess()
else:
    for file in os.listdir(preprocess_path):
        temp = np.load(preprocess_path + file)
        information[file.split('.')[0]] = temp


for i in information:
    train_x = information[i][0]
    train_y = information[i][1]
    n_vocab = information[i][3]

    print("Training on instrument {}".format(i))
    print("Dataset shape {} with unique note {}".format(np.shape(train_x), n_vocab))

    if not os.path.exists(train_instrument_path(i)):
        os.makedirs(train_instrument_path(i))

    # Training
    model = get_model(train_x, n_vocab)
    model.summary()
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callback_builder(i))

    # Save model
    plot_model(model, to_file=train_instrument_path(i, 2), show_shapes=True, show_layer_names=True)
    create_plot(i, history)
    model.save(train_instrument_path(i, 1))

    del model
    K.clear_session()

# f = open(summary, 'w')
# info = ""
# info += '\naccuracy '
# print(info)
# f.write(info)