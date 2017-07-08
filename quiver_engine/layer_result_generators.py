from keras.models import Model


def get_outputs_generator(model, layer_name):

    def outputs_generator(image):
        layer_outputs = Model(
            input=model.input,
            output=model.get_layer(layer_name).output
        ).predict(image)

        if K.backend() == 'theano':
            #correct for channel location difference betwen TF and Theano
            layer_outputs = np.rollaxis(layer_outputs, 0, 3)

        return layer_outputs

    return outputs_generator
