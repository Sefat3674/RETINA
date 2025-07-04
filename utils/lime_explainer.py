from lime import lime_image
import numpy as np
from skimage.segmentation import mark_boundaries

def explain_with_lime(model, img_np, class_index):
    explainer = lime_image.LimeImageExplainer()
    def predict_fn(images):
        return model.predict(np.array(images))

    explanation = explainer.explain_instance(
        img_np, predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(
        label=class_index, positive_only=True, num_features=10, hide_rest=False)
    return mark_boundaries(temp, mask)
