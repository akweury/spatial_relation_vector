# Created by shaji on 02.12.2022
import json
import numpy as np
import torch
from PIL import Image
import cv2 as cv
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import config
from engine.SpatialObject import calc_srv, spatial_obj
from engine import rule_utils


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


def model_cd(color_categories, img, mask):
    # img_np = img.numpy()

    max_s_value = 1.0
    min_s_value = 0.78

    max_v_value = 1.0
    min_v_value = 0.2

    img_24bit = img.detach().to('cpu')
    img_np = img_24bit.numpy().astype(np.float32)

    img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

    img_obj = np.zeros(shape=img_24bit.numpy().shape)
    img_obj[mask != 0] = img_24bit[mask != 0]
    img_obj = img_obj.astype(np.float32)
    img_obj_rgb = cv.cvtColor(img_obj, cv.COLOR_BGR2RGB)
    # ret will return a true value if the frame exists otherwise False
    img_obj_hsv = cv.cvtColor(img_obj_rgb, cv.COLOR_RGB2HSV)
    img_obj_rgb_cb = cv.cvtColor(img_obj_hsv, cv.COLOR_HSV2RGB)

    red_l = np.array([230, min_s_value, min_v_value])  # setting the red lower limit
    red_h = np.array([240, max_s_value, max_v_value])  # setting the red upper limit

    green_l = np.array([110, min_s_value, min_v_value])  # setting the green lower limit
    green_h = np.array([125, max_s_value, max_v_value])  # setting the green upper limit

    blue_l = np.array([0, min_s_value, min_v_value])  # setting the blue lower limit
    blue_h = np.array([15, max_s_value, max_v_value])  # setting the blue upper limit

    r_mask = cv.inRange(img_obj_hsv, red_l, red_h)
    g_mask = cv.inRange(img_obj_hsv, green_l, green_h)
    b_mask = cv.inRange(img_obj_hsv, blue_l, blue_h)

    red_score = np.sum(r_mask)
    green_score = np.sum(g_mask)
    blue_score = np.sum(b_mask)
    color_label = np.argmax([0, red_score, green_score, blue_score])

    color = color_categories[color_label]
    return color


def load_rules(data, entity_num):
    # create SpatialObjects to save object vectors
    target_obj = {}
    target_obj["position"] = np.array([data["target"]["x"], data["target"]["y"], data["target"]["z"]])
    target_obj["size"] = 0.2
    target_obj["shape"] = data["target"]["shape"]
    targetSpatialObj = spatial_obj(id=target_obj["id"], shape=target_obj["shape"], pos=target_obj["position"],
                                   size=target_obj["size"])

    ruleSpatialObjs = []
    ruleSpatialObjs.append(spatial_obj(id=target_obj["id"], shape=data["Objs"][0]["shape"],
                                       pos=np.array([data["Objs"][0]["x"],
                                                     data["Objs"][0]["y"],
                                                     data["Objs"][0]["z"]]),
                                       size=0.4949747))
    ruleSpatialObjs.append(spatial_obj(id=target_obj["id"], shape=data["Objs"][1]["shape"],
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


def common_exist(property, property_matrices):
    for property_matrix in property_matrices:
        exist = False
        for obj in property_matrix:
            for obj_property in obj:
                if obj_property["value"] == property["value"]:
                    exist = True
        if not exist:
            return False
    return True


# def isSubList(sub_list, full_list):
#     list_a = []
#     list_b = []
#     for element in sub_list:
#         new_element = {"value": element["value"],
#                        "propertyType": element["propertyType"],
#                        }
#         list_a.append(new_element)
#     for element in full_list:
#         new_element = {"value": element["value"],
#                        "propertyType": element["propertyType"],
#                        }
#         list_b.append(new_element)
#
#     res = False
#     for idx in range(len(list_b) - len(list_a) + 1):
#         if list_b[idx: idx + len(list_a)] == list_a:
#             res = True
#             break
#     return res


def isSubList(sub_list, full_list):
    list_a = sub_list
    list_b = list(full_list)

    res = False
    for idx in range(len(list_b) - len(list_a) + 1):
        if list_b[idx: idx + len(list_a)] == list_a:
            res = True
            break
    return res


def common_exist_check(subset, property_matrices):
    if len(subset) == 0:
        return False
    property_matrix = property_matrices[0]
    common_matrix = []
    for obj in property_matrix:
        candidate_obj = []
        for property in obj:
            if property in subset:
                candidate_obj.append(property)
        if len(candidate_obj) > 0:
            common_matrix.append(candidate_obj)

    obj_available = np.ones(shape=property_matrices.shape[:2])
    for i in range(1, len(property_matrices)):
        property_matrix = property_matrices[i]
        for candidate_obj in common_matrix:
            exist = False
            for j in range(property_matrix.shape[0]):
                if obj_available[i, j] == 1:
                    if isSubList(candidate_obj, property_matrix[j]):
                        exist = True
                        obj_available[i, j] = 0
                        break
                        # property_matrix.remove(property_matrix[j])  # each object can only be matched once
            if not exist:
                return False
    return True


def rule_exist_search(property_matrices, learned_rules):
    common_exist_set = []
    common_for_all_set = []
    common_exist_rules = learned_rules
    common_for_all_rules = []
    image_num = len(property_matrices)
    property_num = len(property_matrices[0][0])
    for property_matrix in property_matrices:
        obj_num = len(property_matrix)
        for obj in property_matrix:
            for property in obj:
                if common_exist(property, property_matrices):
                    property["commonExist"] = True
                    common_exist_set.append(property)
        sub_common_sets = rule_utils.calc_subset(common_exist_set)

        for subset in sub_common_sets:
            if common_exist_check(subset, property_matrices):
                if subset not in common_exist_rules:
                    common_exist_rules.append(subset)

    return common_exist_rules


def rule_check(property_matrices, learned_rules):
    # delete repeated rules
    no_repeat_rules = []
    for rule in learned_rules:
        premise = rule["premise"]
        conclusion = rule["conclusion"]
        is_repeat_rule = False
        for no_repeat_rule in no_repeat_rules:
            premise_no_repeat = no_repeat_rule["premise"]
            conclusion_no_repeat = no_repeat_rule["conclusion"]
            if premise == premise_no_repeat and conclusion == conclusion_no_repeat:
                is_repeat_rule = True
                break
        if not is_repeat_rule:
            no_repeat_rules.append(rule)

    # check if any rules have been satisfied
    satisfied_rules = []
    unsatisfied_rules = []
    for rule in no_repeat_rules:
        if rule_utils.common_pair(rule, property_matrices):
            satisfied_rules.append(rule)
        else:
            unsatisfied_rules.append(rule)

    return satisfied_rules, unsatisfied_rules


def rule_search(property_matrices, learned_rules):
    # search for all the possible candidate rules
    candidate_rules = rule_utils.candidate_rule_search(property_matrices)

    # check if candidate rules exist in each image.
    learned_rules_batch = rule_utils.exist_rule_search(property_matrices, candidate_rules)

    # update rule list
    learned_rules = rule_utils.add_new_rules(learned_rules, learned_rules_batch)

    return learned_rules, learned_rules_batch


def obj2propertyList(obj):
    lists = []
    for property in obj:
        lists.append({
            "name": property.name,
            "value": property.value,
            "commonExist": property.commonExist,
            "parentId": property.parent
        })
    return lists


def save_rules(rules, file_name):
    rules_json = []
    for rule in rules:
        rule_json = {}
        premise_dict = {}

        premise_dict["ref"] = obj2propertyList(rule["premise"]["ref"])
        premise_dict["obj"] = obj2propertyList(rule["premise"]["obj"])
        rule_json["premise"] = premise_dict
        rule_json["conclusion"] = rule["conclusion"]
        rule_json["freq"] = rule['freq']
        rules_json.append(rule_json)

    with open(file_name, "w") as f:
        json.dump(rules_json, f)
