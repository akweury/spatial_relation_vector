# Created by shaji on 02.12.2022

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import config
from engine.SpatialObject import generate_spatial_obj, calc_srv, spatial_obj, attrDiff


def mask_rcnn(img_tensor_int, weights=None):
    if weights is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    if not torch.is_tensor(img_tensor_int):
        img_tensor_int = pil_to_tensor(Image.open(img_tensor_int)).unsqueeze(dim=0)

    img_tensor_float = img_tensor_int / 255.0
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    img_preds = model(img_tensor_float)
    img_preds[0]["boxes"] = img_preds[0]["boxes"][img_preds[0]["scores"] > 0.8]
    img_preds[0]["labels"] = img_preds[0]["labels"][img_preds[0]["scores"] > 0.8]
    img_preds[0]["masks"] = img_preds[0]["masks"][img_preds[0]["scores"] > 0.8]
    img_preds[0]["scores"] = img_preds[0]["scores"][img_preds[0]["scores"] > 0.8]

    categories = weights.meta["categories"]
    img_labels = img_preds[0]["labels"].numpy()
    img_annot_labels = [f"{categories[label]}: {prob:.2f}" for label, prob in
                        zip(img_labels, img_preds[0]["scores"].detach().numpy())]
    img_output_tensor = draw_bounding_boxes(image=img_tensor_int[0],
                                            boxes=img_preds[0]["boxes"],
                                            labels=img_annot_labels,
                                            colors=["red" if categories[label] == "person" else "green" for label in
                                                    img_labels],
                                            width=2)

    img_masks_float = img_preds[0]["masks"].squeeze(1)
    img_masks_float[img_masks_float < 0.8] = 0
    img_masks_bool = img_masks_float.bool()
    img_output_tensor = draw_segmentation_masks(img_output_tensor, masks=img_masks_bool, alpha=0.8)
    img_output = to_pil_image(img_output_tensor)

    return img_output


def get_model_instance_segmentation(num_classes, weights=None):
    if weights is None:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def model_fe(predictions, images, vertices, objects, log_manager, entity_num):
    facts = []
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
            spatialObj = generate_spatial_obj(vertex=vertex,
                                              img=image,
                                              label=prediction["labels"][j],
                                              mask=prediction["masks"][j],
                                              categories=categories)
            spatialObjs.append(spatialObj)

        obj_num = len(spatialObjs)

        image_srv = []
        for obj_i in range(obj_num):
            for obj_j in range(obj_num):
                srv = calc_srv(spatialObjs[obj_j], spatialObjs[obj_j], entity_num)
                image_srv.append(srv)
        facts.append(image_srv)

    facts = np.array(facts)

    # DF = pd.DataFrame(facts)
    # DF.to_csv("data1.csv")
    return facts


def load_rules(data, entity_num):
    # create SpatialObjects to save object vectors
    target_obj = {}
    target_obj["position"] = np.array([data["target"]["x"], data["target"]["y"], data["target"]["z"]])
    target_obj["size"] = 0.2
    target_obj["shape"] = data["target"]["shape"]
    targetSpatialObj = spatial_obj(shape=target_obj["shape"], pos=target_obj["position"], size=target_obj["size"])

    ruleSpatialObjs = []
    ruleSpatialObjs.append(spatial_obj(shape=data["Objs"][0]["shape"],
                                       pos=np.array([data["Objs"][0]["x"],
                                                     data["Objs"][0]["y"],
                                                     data["Objs"][0]["z"]]),
                                       size=0.4949747))
    ruleSpatialObjs.append(spatial_obj(shape=data["Objs"][1]["shape"],
                                       pos=np.array([data["Objs"][1]["x"],
                                                     data["Objs"][1]["y"],
                                                     data["Objs"][1]["z"]]),
                                       size=0.35))

    obj_num = len(ruleSpatialObjs)
    srvs = np.zeros(shape=(obj_num, 6))
    for i in range(obj_num):
        srv = calc_srv(targetSpatialObj, ruleSpatialObjs[i], entity_num)
        srvs[i, :] = srv
    return srvs


def calc_rrv(facts):
    relative_relation_vectors = []
    np.set_printoptions(precision=2)
    facts = [facts[0]]
    for relation_vectors in facts:
        for i in range(relation_vectors.shape[0]):
            for j in range(relation_vectors.shape[1]):
                relation_vector = relation_vectors[i, j]
                print(f"relation vector: {relation_vector}")

    return None

# def similarity(vectorA, vectorB):



def learn_common_rv(data):
    batch, = data.shape(0)
    rv_learned = []
    rv_candidate = []
    for image_idx in range(data.shape(0)):
        for rv_idx in range(data.shape(1)):
            if len(rv_candidate) == 0:
                rv_candidate.append(data[image_idx, rv_idx])
                continue
            rv = data[image_idx, rv_idx]


    return None