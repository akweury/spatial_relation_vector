# Created by shaji on 14-Dec-22

import json
import torch
from torch.utils.data import DataLoader

from engine.FactExtractorDataset import FactExtractorDataset
from engine.SpatialObject import SpatialObject
from engine import config, pipeline, models, args_utils
import create_dataset
from engine.models import model_fe, rule_search, rule_check, save_rules

# rules_json = "D:\\UnityProjects\\hide_dataset_unity\\Assets\\Scripts\\Rules\\front.json"
# rules_json = "/Users/jing/PycharmProjects/hide_dataset_unity/Assets/Scripts/Rules/front.json"
# entity_num = 11

# with open(rules_json) as f:
#     rules_data = json.load(f)

# preprocessing
args = args_utils.paser()
# init log manager
log_manager = pipeline.LogManager(model_exp="object_detector_big", args=args)

# create data tensors if they are not exists
create_dataset.data2tensor_fact_extractor(log_manager.data_path, args)

train_dataset = FactExtractorDataset(log_manager.data_path, "train")
test_dataset = FactExtractorDataset(log_manager.data_path, "test")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=pipeline.collate_fn)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1,
                         collate_fn=pipeline.collate_fn)

categories = config.categories

model_od = models.get_model_instance_segmentation(args.num_classes).to(args.device)
model_od, optimizer, parameters = pipeline.load_checkpoint(config.model_ball_sphere_detector, args, model_od)
model_od.eval()
learned_rules = []
for i, (data, objects) in enumerate(train_loader):
    with torch.no_grad():
        # input data
        images = list((_data[3:] / 255).to(args.device) for _data in data)
        vertex = list((_data[:3]).to(args.device) for _data in data)

        # object detection
        prediction = model_od(images)

        # fact extractor
        facts = model_fe(prediction, images, vertex, objects, log_manager)
        # common_rv = learn_common_rv(facts)
        # learned_rules, _ = rule_check(facts, learned_rules)
        learned_rules, learned_rules_batch = rule_search(facts, learned_rules)
        save_rules(learned_rules, log_manager.output_folder / f"learned_rules_{i}.json")
        log_manager.visualization(images, prediction, categories,learned_rules=learned_rules_batch,  idx=i, show=True)
        print("batch")
print(learned_rules)

# apply rules
for i, (data, objects) in enumerate(test_loader):
    with torch.no_grad():
        # input data
        images = list((_data[3:] / 255).to(args.device) for _data in data)
        vertex = list((_data[:3]).to(args.device) for _data in data)

        # object detection
        prediction = model_od(images)

        # fact extractor
        facts = model_fe(prediction, images, vertex, objects, log_manager)
        # common_rv = learn_common_rv(facts)
        satisfied_rules, unsatisfied_rules = rule_check(facts, learned_rules)
        save_rules(satisfied_rules, log_manager.output_folder / "satisfied_rules.json")
        save_rules(unsatisfied_rules, log_manager.output_folder / "unsatisfied_rules.json")
        log_manager.visualization(images, prediction, categories,satisfied_rules=satisfied_rules, unsatisfied_rules=unsatisfied_rules, idx=i, show=True)
        if len(unsatisfied_rules) > 0:
            print("break")
