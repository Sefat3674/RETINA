import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm

def generate_gradcam(model, img_array, class_index, layer_name, original_size):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), original_size)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.jet(heatmap / 255.0)[:, :, :3]
    overlay = 0.5 * jet + 0.5 * img_array[0]
    return np.uint8(overlay * 255)