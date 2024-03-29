# Created by shaji on 14-Dec-22

import os
import torch
from torch.utils.data import DataLoader

from engine.FactExtractorDataset import FactExtractorDataset
from engine import config, pipeline, args_utils, create_dataset
from engine.models import rule_search, save_rules
from engine import rule_utils



od_model_path = config.model_ball_sphere_detector

# preprocessing
args = args_utils.paser()
# init log manager
log_manager = pipeline.LogManager(args=args)

# create data tensors if they are not exists
create_dataset.data2tensor_fact_extractor(log_manager.data_path, args)

train_dataset = FactExtractorDataset(log_manager.data_path)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=pipeline.collate_fn)

categories = config.categories

model_od, optimizer, parameters = pipeline.load_checkpoint(od_model_path)
model_od.eval()
learned_rules = []
for i, (data, objects, _, _, _) in enumerate(train_loader):
    with torch.no_grad():
        # input data
        images = list((_data[3:] / 255).to(args.device) for _data in data)
        vertex = list((_data[:3]).to(args.device) for _data in data)

        # object detection
        prediction = model_od(images)

        # fact extractor
        continual_spatial_objs = rule_utils.get_obj_tensors(prediction, images, vertex, objects, log_manager)
        obj_properties_prediction = rule_utils.get_discrete_spatial_objs(continual_spatial_objs)

        learned_rules, learned_rules_batch = rule_search(obj_properties_prediction, learned_rules)
        save_rules(learned_rules, log_manager.output_folder / f"learned_rules_{i}.json")
        log_manager.visualization(images, prediction, categories,
                                  learned_rules=learned_rules_batch, property_prediction=obj_properties_prediction, idx=i, prefix='Train', show=False)
        print("break")

# save learned rules
rule_utils.save_rules(learned_rules, os.path.join(str(log_manager.model_folder), 'learned_rules.json'))
