import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from keras.models import model_from_json
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


# util function to normalize
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


# util function to convert a tensor into a valid image
def deprocess_image(x):

    x -= x.mean()
    x /= np.amax(x)
    # x /= (x.std() + K.epsilon())
    x *= 0.1
    # x = np.clip(x, -1, 1)

    return x


# dimensions of the generated pictures for each filter.
def visualize(model_name, layer_name, num_filters_show):

    json_file = open('Models\\{}.json'.format(model_name), 'r')
    model = model_from_json(json_file.read())
    json_file.close()
    model.load_weights("Models\\{}.h5".format(model_name))
    model.summary()

    # this is the placeholder for the input images
    input_img = model.input
    img_width = int(input_img.shape[1])

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    num_filters = int(layer_dict[layer_name].output.shape[2])
    kept_filters = []

    for filter_index in range(num_filters):

        print('Processing filter %d' % filter_index)

        layer_output = layer_dict[layer_name].output

        # we build a loss function that maximizes the activation of the nth filter of the layer considered
        loss = K.mean(layer_output[ :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.0

        # we start from a gray image with some random noise
        input_img_data = np.random.random((1, img_width, 1))
        input_img_data = input_img_data - 0.5

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            # print('Current loss value:', loss_value)
            #if loss_value <= 0.:
                # print("stuck")
                # some filters get stuck to 0, we can skip them
                # break

        # decode the resulting input image
        print("Loss: {}".format(loss_value))
        if loss_value > -1000:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))

    # the filters that have the highest loss are assumed to be better-looking. we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)

    # fill the picture with our saved filters
    for i in range(num_filters_show):
        signal, loss = kept_filters[i]
        # print(loss)

        wavfile.write('Filters\\{}_filters_{}_{}.wav'.format(model_name, layer_name, i), 22050, signal)

        # t = np.arange(0, signal.shape[0], 1)
        # plt.plot(t, signal)
        # plt.show()