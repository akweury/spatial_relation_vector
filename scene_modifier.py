# Created by shaji on 03-Jan-23

import os
import torch
from torch.utils.data import DataLoader

from engine.FactExtractorDataset import FactExtractorDataset
from engine import config, pipeline, args_utils, rule_utils, file_utils
from engine.models import rule_check

# preprocessing
args = args_utils.paser()
log_manager = pipeline.LogManager(model_exp="object_detector_big", args=args)
test_loader = DataLoader(FactExtractorDataset(log_manager.data_path, "test"), shuffle=True, batch_size=1,
                         collate_fn=pipeline.collate_fn)
# load object detector
model_od, optimizer, parameters = pipeline.load_checkpoint(config.model_ball_sphere_detector, args)
model_od.eval()

# load learned rules
learned_rules = rule_utils.load_rules(os.path.join(str(config.rules_ball_sphere)))
learned_rules = rule_utils.rule_combination(learned_rules)
# apply rules
for i, (data, objects, vertex_max, vertex_min, file_json) in enumerate(test_loader):
    with torch.no_grad():
        # input data
        images = list((_data[3:] / 255).to(args.device) for _data in data)
        vertex = list((_data[:3]).to(args.device) for _data in data)

        # object detection
        prediction = model_od(images)

        # fact extractor
        continual_spatial_objs = rule_utils.get_continual_spatial_objs(prediction, images, vertex, objects, log_manager)
        facts = rule_utils.get_discrete_spatial_objs(continual_spatial_objs)

        satisfied_rules, unsatisfied_rules = rule_check(facts, learned_rules)
        log_manager.visualization(images, prediction, config.categories,
                                  satisfied_rules=satisfied_rules, unsatisfied_rules=unsatisfied_rules, facts=facts,
                                  idx=i, show=True)
        try_counter = 0
        while len(unsatisfied_rules) > 0:
            try_counter += 1
            continual_spatial_objs = rule_utils.get_random_continual_spatial_objs(continual_spatial_objs,vertex_max[0], vertex_min[0])
            facts = rule_utils.get_discrete_spatial_objs(continual_spatial_objs)
            satisfied_rules, unsatisfied_rules = rule_check(facts, learned_rules)

        print(f"tried {try_counter} times")
        scene = rule_utils.objs2scene(continual_spatial_objs)

        scene_dict = {'scene':scene, 'file_name':file_json[0]}
        file_utils.save_json(scene_dict, str(config.output/ args.exp / f"output_scene_{i}.json"))
