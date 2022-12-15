# Created by shaji on 14-Dec-22

import json
import torch
from torch.utils.data import DataLoader

from engine.FactExtractorDataset import FactExtractorDataset
from engine.SpatialObject import SpatialObject
from engine import config, pipeline, models, args_utils
import create_dataset
from engine.models import model_fe, load_rules, calc_rrv

rules_json = "D:\\UnityProjects\\hide_dataset_unity\\Assets\\Scripts\\Rules\\front.json"
with open(rules_json) as f:
    rules_data = json.load(f)

# preprocessing
args = args_utils.paser()
# init log manager
log_manager = pipeline.LogManager(model_exp="object_detector_big", args=args)

# create data tensors if they are not exists
create_dataset.data2tensor_fact_extractor(log_manager.data_path, args)

train_dataset = FactExtractorDataset(log_manager.data_path, "train")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=pipeline.collate_fn)
categories = config.categories

model_od = models.get_model_instance_segmentation(args.num_classes).to(args.device)
model_od, optimizer, parameters = pipeline.load_checkpoint(config.model_ball_sphere_detector, args, model_od)
model_od.eval()
for i, (data, objects) in enumerate(train_loader):
    with torch.no_grad():
        # input data
        images = list((_data[3:] / 255).to(args.device) for _data in data)
        vertex = list((_data[:3]).to(args.device) for _data in data)

        # object detection
        prediction = model_od(images)

        # fact extractor
        facts = model_fe(prediction, images, vertex, objects, log_manager)
        target_relation_vectors = load_rules(rules_data)

        relative_relation_vectors = calc_rrv(facts, target_relation_vectors)



        log_manager.visualization(images, prediction, categories, idx=i)
