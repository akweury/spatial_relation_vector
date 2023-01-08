# Created by shaji on 03-Jan-23
import json
import numpy as np
from engine import config
from engine.SpatialObject import Property, generate_spatial_obj, calc_srv, spatial_obj, attrDiff, calc_property_matrix


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


def get_random_continual_spatial_objs(continual_spatial_objs, vertex_max, vertex_min):
    for img in continual_spatial_objs:
        for obj in img:
            obj.position = np.random.rand(3) * (vertex_max - vertex_min) + vertex_min
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
                            combined_rule['conclusion']['size'] = config.relation_dict['size']
                            combined_rule['freq'] = combined_rule['freq'] + learned_rule['freq']
                    if 'dir' in combined_rule['conclusion'].keys():
                        if combined_rule['conclusion']['dir'] != learned_rule['conclusion']['dir']:
                            combined_rule['conclusion']['dir'] = config.relation_dict['dir']
                            combined_rule['freq'] = combined_rule['freq'] + learned_rule['freq']
        if not hasConflict:
            combined_rules.append(learned_rule)
        print('break')
    return combined_rules


def equivalent_conclusions(conclusion, fact, key):
    if conclusion[key] == fact[key]:
        return True
    return False


def objs2scene(random_continual_spatial_objs):
    scene = []
    for pair in random_continual_spatial_objs:
        for obj in pair:
            scene.append(obj.__dict__)
    for obj in scene:
        for key in obj.keys():
            if isinstance(obj[key], np.ndarray):
                obj[key] = obj[key].tolist()
    return scene
