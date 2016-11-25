import os
import json
import matplotlib.pyplot as plt
from keras.models import model_from_json


if __name__ == '__main__':
    model_file = os.path.join('result', 'model.json')
    weight_file = os.path.join('result', 'model.h5')

    with open(model_file, 'r') as fp:
        model = model_from_json(fp.read())
    model.load_weights(weight_file)
    model.summary()

    W = model.layers[0].get_weights()[0]
    W = W.transpose(3, 2, 0, 1)
    print(W.shape)

    nb_filter, nb_channel, nb_row, nb_col = W.shape

    plt.figure()
    for i in range(nb_filter):
        im = W[i]

        plt.subplot(4, 8, i + 1)
        plt.axis('off')
        plt.imshow(im)
    plt.show()
