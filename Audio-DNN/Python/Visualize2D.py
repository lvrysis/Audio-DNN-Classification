import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from keras.models import model_from_json
from keras import backend as K
# from scipy.misc import imsave, toimage
from PIL import Image
import numpy as np


# util function to normalize
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.4

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
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
    img_height = int(input_img.shape[2])

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    num_filters = int(layer_dict[layer_name].output.shape[3])
    kept_filters = []
    for filter_index in range(num_filters):

        print('Processing filter %d' % filter_index)

        # we build a loss function that maximizes the activation of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])
        loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = np.random.random((1, img_width, img_height, 1))

        # input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            # print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))

    # we will stich the best n*n filters on a n x n grid.
    n = int(np.sqrt(num_filters_show))

    # the filters that have the highest loss are assumed to be better-looking. we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for our n x n filters of size w x h, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img

    # save the result to disk
    # toimage(stitched_filters.transpose(1,0,2)).show()
    img = Image.fromarray(stitched_filters.transpose(1,0,2).astype('uint8'), 'RGB');
    img = img.resize((img.size[0]*2,img.size[1]*2), Image.ANTIALIAS)
    img.save('Filters\\{}_filters_{}.png'.format(model_name,layer_name))
    img.show()

    # layer_dict = dict([(layer.name, layer) for layer in model.layers])
    # layer_name = 'conv2d_1'
    # filter_index = 10
    # slice = layer_dict[layer_name].get_weights()[0][:,:,0,filter_index]
    # img = Image.fromarray(np.uint8(slice*255), 'L')
    # img.show()