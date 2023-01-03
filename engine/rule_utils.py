# Created by shaji on 03-Jan-23

import numpy as np
from engine import config
from engine.SpatialObject import generate_spatial_obj, calc_srv, spatial_obj, attrDiff, calc_property_matrix

def get_continual_spatial_objs(predictions, images, vertices, objects, log_manager):
    facts = []
    spatialObjMatrix = []
    for i in range(len(images)):
        image = images[i]
        vertex = vertices[i]
        prediction = predictions[i]

        prediction["boxes"] = prediction["boxes"][prediction["scores"] > log_manager.args.conf_threshold]
        prediction["labels"] = prediction["labels"][prediction["scores"] > log_manager.args.conf_threshold]
        prediction["masks"] = prediction["masks"][prediction["scores"] > log_manager.args.conf_threshold]
        prediction["scores"] = prediction["scores"][prediction["scores"] > log_manager.args.conf_threshold]

        img_labels = prediction["labels"].to("cpu").numpy()
        categories = config.categories
        # print(f"{len(img_labels)} objects has been detected.")
        labels_with_prob = zip(img_labels, prediction["scores"].detach().to("cpu").numpy())
        img_annot_labels = []
        for label, prob in labels_with_prob:
            print(f"categories: {categories}, label: {label}, prob: {prob:.2f}")
            img_annot_labels.append(f"{categories[label]}: {prob:.2f}")

        # create SpatialObjects to save object vectors
        spatialObjs = []
        for j in range(len(prediction["labels"])):
            spatialObj = generate_spatial_obj(id=j,
                                              vertex=vertex,
                                              img=image,
                                              label=prediction["labels"][j],
                                              mask=prediction["masks"][j],
                                              categories=categories)
            spatialObjs.append(spatialObj)

        spatialObjMatrix.append(spatialObjs)
    return spatialObjMatrix


def get_discrete_spatial_objs(continual_spatial_objs):
    facts = []
    for i in range(len(continual_spatial_objs)):
        property_matrix = calc_property_matrix(continual_spatial_objs[i], config.propertyNames)
        facts.append(property_matrix)
    facts = np.array(facts)
    return facts


def get_random_continual_spatial_objs(continual_spatial_objs):
    for img in continual_spatial_objs:
        for obj in img:
            obj["pos"] = np.random.rand(3)

    return continual_spatial_objs