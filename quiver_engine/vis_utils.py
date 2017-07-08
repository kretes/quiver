import numpy as np
import keras.backend as K
from quiver_engine.util import get_evaluation_context
from quiver_engine.file_utils import save_layer_img
from quiver_engine.layer_result_generators import get_outputs_generator

def save_layer_outputs(input_img, model, layer_name, temp_folder, input_path):

    with get_evaluation_context():

        layer = model.get_layer(layer_name)
        if ("GlobalAveragePooling2D" in str(type(layer))):
            next_layer = model.layers[model.layers.index(layer)+1]
            if ("Dense" in str(type(next_layer))):
                previous_layer = model.layers[model.layers.index(layer)-1]
                prev_layer_outputs = get_outputs_generator(model, previous_layer.name)(input_img)[0]
                next_layer_weights = next_layer.get_weights()[0]
                prev_layer_channels = next_layer_weights.shape[0]
                num_classes = next_layer_weights.shape[1]
                classes_outputs = np.zeros((prev_layer_outputs.shape[0],prev_layer_outputs.shape[1],len(classes)))
                for cl in range(0,num_classes):
                    for channel in range(0, prev_layer_channels):
                        classes_outputs[:, :, cl] = classes_outputs[:, :, cl] + prev_layer_outputs[:,:,channel] * next_layer_weights[channel,cl]

                classes_outputs = deprocess_image(classes_outputs)
                return [save_layer_img(
                    classes_outputs[:,:,cl],
                    layer_name,
                    cl,
                    temp_folder,
                    input_path,
                    False
                ) for cl in range(0,num_classes)]
            elif ("Activation" in str(type(next_layer))):
                previous_layer = model.layers[model.layers.index(layer)-1]
                prev_layer_outputs = get_outputs_generator(model, previous_layer.name)(input_img)[0]
                prev_layer_outputs = deprocess_image(prev_layer_outputs)

                return [
                    save_layer_img(
                        prev_layer_outputs[:, :, channel],
                        layer_name,
                        channel,
                        temp_folder,
                        input_path,
                        False
                    )
                    for channel in range(0, prev_layer_outputs.shape[2])
                ]

        layer_outputs = get_outputs_generator(model, layer_name)(input_img)[0]

        return [
            save_layer_img(
                layer_outputs[:, :, channel],
                layer_name,
                channel,
                temp_folder,
                input_path
            )
            for channel in range(0, layer_outputs.shape[2])
        ]
