# Created by shaji on 03-Jan-23
import json
import numpy as np
from engine import config
from engine.SpatialObject import Property, generate_spatial_obj
from engine import mechanics, models
import torch


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


def propertyList2Obj(propertyList):
    obj = []
    for property in propertyList:
        obj.append(Property(property["value"], property["name"], property["parentId"]))
    return obj


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


def load_rules(file_name):
    # load the json file
    with open(file_name) as f:
        data = json.load(f)

    rules = []
    for rule_json in data:
        rule = {"premise": {"ref": propertyList2Obj(rule_json["premise"]["ref"]),
                            "obj": propertyList2Obj(rule_json["premise"]["obj"])},
                "conclusion": rule_json["conclusion"],
                "freq": rule_json["freq"]}
        rules.append(rule)
    return rules


def get_continual_spatial_objs(prefix, od_pred, images, vertices, objects, log_manager):
    """
    return a list of spatialObjs.
    Each spatial obj contains all the property information in continual space.
    """
    facts = []
    spatialObjMatrix = []
    for i in range(len(images)):
        image = images[i]
        vertex = vertices[i]
        od_prediction = od_pred[i]

        od_prediction["boxes"] = od_prediction["boxes"][od_prediction["scores"] > log_manager.args.conf_threshold]
        od_prediction["labels"] = od_prediction["labels"][od_prediction["scores"] > log_manager.args.conf_threshold]
        od_prediction["masks"] = od_prediction["masks"][od_prediction["scores"] > log_manager.args.conf_threshold]
        od_prediction["scores"] = od_prediction["scores"][od_prediction["scores"] > log_manager.args.conf_threshold]

        labels = od_prediction["labels"].to("cpu").numpy()
        categories = config.categories
        color_categories = config.color_categories
        scores = od_prediction["scores"].detach().to("cpu").numpy()
        boxes = od_prediction["boxes"].detach().to("cpu").numpy()
        masks = od_prediction["masks"].detach().to("cpu").numpy()
        pred_res = [{"score": scores[ind],
                     "label": labels[ind],
                     "box": boxes[ind],
                     "mask": masks[ind]} for ind in range(len(labels))]
        print(f"{len(pred_res)} objects have been detected.")
        if len(pred_res) >= log_manager.args.e:
            pred_res = sorted(pred_res, key=lambda x: x["score"], reverse=True)
            pred_res = pred_res[:log_manager.args.e]
        else:
            continue
        for pred in pred_res:
            print(f"\tcategories: {categories}, label: {pred['label']}, prob: {pred['score']:.2f}")
        from engine import plot_utils
        img_uint8 = (image * 255).to(torch.uint8)
        img_show = plot_utils.maskRCNNVisualization(img_uint8, pred_res, log_manager.args.conf_threshold, categories)
        img_show.save(str(config.storage / 'output' / f"{log_manager.args.subexp}" / f"{prefix}.png"))
        # create SpatialObjects to save object vectors
        spatialObjs = []
        for j in range(len(pred_res)):
            spatialObj = generate_spatial_obj(id=j,
                                              vertex=vertex,
                                              img=image,
                                              label=pred_res[j]["label"],
                                              mask=pred_res[j]["mask"],
                                              categories=categories,
                                              color_categories=color_categories,
                                              box=pred_res[j]["box"],
                                              pred=pred_res[j]["score"])
            spatialObjs.append(spatialObj)

        spatialObjMatrix.append(spatialObjs)
    return spatialObjMatrix


def get_discrete_spatial_objs(continual_spatial_objs):
    scene_predictions = []
    for img_objs in continual_spatial_objs:
        property_matrix = calc_property_matrix(img_objs, config.propertyNames)
        scene_predictions.append(property_matrix)
    scene_predictions = np.array(scene_predictions)
    return scene_predictions


def get_random_continual_spatial_objs(continual_spatial_objs, vertex_max, vertex_min):
    for img in continual_spatial_objs:
        for obj in img:
            obj.position = np.random.rand(3) * (vertex_max.numpy() - vertex_min.numpy()) + vertex_min.numpy()
    return continual_spatial_objs


def rule_combination(learned_rules):
    combined_rules = []
    for learned_rule in learned_rules:
        hasConflict = False
        for combined_rule in combined_rules:
            if combined_rule['premise'] == learned_rule['premise']:
                if combined_rule['conclusion'].keys() == learned_rule['conclusion'].keys():
                    hasConflict = True
                    if 'size' in combined_rule['conclusion'].keys():
                        if combined_rule['conclusion']['size'] != learned_rule['conclusion']['size']:
                            combined_rule['conclusion']['size'] += (config.relation_dict['size'])
                            combined_rule['freq'] = combined_rule['freq'] + learned_rule['freq']
                    if 'dir' in combined_rule['conclusion'].keys():
                        if combined_rule['conclusion']['dir'] != learned_rule['conclusion']['dir']:
                            combined_rule['conclusion']['dir'] += (learned_rule['conclusion']['dir'])
                            combined_rule['freq'] = combined_rule['freq'] + learned_rule['freq']
        if not hasConflict:
            combined_rules.append(learned_rule)
        print('break')
    return combined_rules


def isSubList(test_list, sublist):
    res = False
    for idx in range(len(test_list) - len(sublist) + 1):
        if test_list[idx: idx + len(sublist)] == sublist:
            res = True
            break
    return res


def equivalent_conclusions(conclusion, fact, key):
    if isinstance(conclusion[key], list):
        return isSubList(conclusion[key], fact[key])
    elif conclusion[key] == fact[key]:
        return True
    return False


def objs2scene(random_continual_spatial_objs, vertex_max, vertex_min):
    scene = []
    for pair in random_continual_spatial_objs:
        for obj in pair:
            scene.append(obj.__dict__)
    for obj in scene:
        for key in obj.keys():
            if isinstance(obj[key], np.ndarray):
                if key == "position":
                    obj[key][2] = 1 - obj[key][2]  # unity has inverse z axis
                    obj[key] = obj[key] * (vertex_max - vertex_min) + vertex_min
                obj[key] = obj[key].tolist()

    return scene


def get_obj_vector(obj, propertyNames):
    obj_vector = []
    for j in range(len(propertyNames)):
        mapped_property = mechanics.property_mapping(obj.__dict__[propertyNames[j]], propertyNames[j])
        propertyObj = Property(mapped_property, propertyNames[j], obj.id)
        obj_vector.append(propertyObj)
    return obj_vector


def calc_property_matrix(objs, propertyNames):
    '''
    return a list of relationship dictionaries in an image.
    One of the objects is considered as reference, then calculate the relationship between reference object and others.
    The relationships are described by discrete symbols.
    '''

    # find the reference obj
    obj_ref = mechanics.find_ref_obj(objs)
    ref_obj_vector = get_obj_vector(obj_ref, propertyNames)

    obj_relation_matrix = []
    for obj in objs:
        if obj != obj_ref:
            # relationship
            ref_dir = mechanics.dir_mapping(obj_ref.position, obj.position)
            ref_size = mechanics.size_mapping(obj_ref.size, obj.size)
            # obj vector
            obj_vector = get_obj_vector(obj, propertyNames)
            obj_relation_matrix.append({
                "ref": ref_obj_vector,
                "dir": [ref_dir],
                "size": [ref_size],
                "obj": obj_vector,
            })
    return obj_relation_matrix


def calc_subset(full_set):
    full_set = list(full_set)
    # lists = [[]]  # empty set has been included
    lists = []  # no empty set has been included
    for i in range(len(full_set) + 1):
        for j in range(i):
            subset = full_set[j:i]
            if subset not in lists:
                lists.append(subset)
    return lists


def calc_subset_2objs(ref_obj, obj):
    lists = []
    ref_obj_subset = calc_subset(ref_obj)
    obj_subset = calc_subset(obj)
    for ref_obj_property in ref_obj_subset:
        for obj_property in obj_subset:
            lists.append({"ref": ref_obj_property, "obj": obj_property})

    return lists


def candidate_rule_search(property_matrices):
    premise_conclusion_pairs = []
    for img_relations in property_matrices:
        for relation in img_relations:
            premise = calc_subset_2objs(relation["ref"], relation["obj"])
            conclusion = [{"dir": relation["dir"]},
                          {"size": relation["size"]},
                          {"dir": relation["dir"], "size": relation["size"]}
                          ]
            premise_conclusion_pairs.append({"premise": premise, "conclusion": conclusion, "freq": 1})

    return premise_conclusion_pairs


def add_new_rules(learned_rules, new_rules):
    is_new_rule = True
    for new_rule in new_rules:
        for old_rule in learned_rules:
            if old_rule["premise"] == new_rule["premise"] and \
                    old_rule["conclusion"] == new_rule["conclusion"]:
                old_rule["freq"] += new_rule["freq"]
                is_new_rule = False
                break
        if is_new_rule:
            learned_rules.append(new_rule)
    return learned_rules


def exist_rule_search(property_matrices, candidate_rules):
    learned_rules_batch = []
    for candidate_rule in candidate_rules:
        premises = candidate_rule["premise"]
        conclusions = candidate_rule["conclusion"]
        for premise in premises:
            for conclusion in conclusions:
                new_batch_rule = {"premise": premise, "conclusion": conclusion, "freq": 1}
                if common_pair(new_batch_rule, property_matrices):
                    is_new_batch_rule = True

                    for each_rule in learned_rules_batch:
                        if each_rule["premise"] == new_batch_rule["premise"] and \
                                each_rule["conclusion"] == new_batch_rule["conclusion"]:
                            each_rule["freq"] += 1
                            is_new_batch_rule = False
                            break
                    if is_new_batch_rule:
                        learned_rules_batch.append(new_batch_rule)
    return learned_rules_batch


def common_pair(given_pair, property_matrices):
    for img_relations in property_matrices:
        is_sub_pair = False
        for relation_pair in img_relations:
            # check if given pair is a subset of each relation pair in the image
            is_sub_pair = check_sub_pair(given_pair, relation_pair)
            if is_sub_pair:
                break
        if not is_sub_pair:
            return False
    return True


def check_sub_pair(sub_pair, pair):
    if isSubObj(sub_pair["premise"]["ref"], pair["ref"]) and isSubObj(sub_pair["premise"]["obj"], pair["obj"]):
        if len(sub_pair["conclusion"]) == 2:
            if equivalent_conclusions(
                    sub_pair["conclusion"], pair, 'size') and equivalent_conclusions(
                sub_pair["conclusion"], pair, 'dir'):
                return True
            else:
                return False
        elif "dir" in sub_pair["conclusion"]:
            if equivalent_conclusions(sub_pair["conclusion"], pair, 'dir'):
                return True
            else:
                return False
        elif "size" in sub_pair["conclusion"]:
            if equivalent_conclusions(sub_pair["conclusion"], pair, 'size'):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def isSubObj(subObj, obj):
    for property_sub in subObj:
        exist = False
        for property_main in obj:
            if property_main == property_sub:
                exist = True
        if not exist:
            return False

    return True
