from util import *
from config import *
from keras.utils import plot_model

# Preprocessing
train_x, train_y, n_vocab = train_preprocess()


# Training
model = get_model(train_x, n_vocab)
model.summary()
history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Save model
plot_model(model, to_file=model_plot, show_shapes=True, show_layer_names=True)
create_plot(history)
model.save(model_saved)

# f = open(summary, 'w')
# info = ""
# info += '\naccuracy '
# print(info)
# f.write(info)