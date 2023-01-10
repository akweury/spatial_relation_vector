# Created by shaji on 03-Jan-23

import os
import torch
from torch.utils.data import DataLoader
import copy

from engine.FactExtractorDataset import FactExtractorDataset
from engine import config, pipeline, args_utils, rule_utils, file_utils, create_dataset
from engine.models import rule_check

# preprocessing
args = args_utils.paser()
log_manager = pipeline.LogManager(model_exp="01.object_detection", args=args)
create_dataset.data2tensor_fact_extractor(log_manager.data_path, args, ["test"])

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

        try_counter = 0
        new_unsatisfied_rules =copy.deepcopy(unsatisfied_rules)
        new_continual_spatial_objs = copy.deepcopy(continual_spatial_objs)
        while len(new_unsatisfied_rules) > 0:
            try_counter += 1
            new_continual_spatial_objs = rule_utils.get_random_continual_spatial_objs(new_continual_spatial_objs,
                                                                                      vertex[0].max(),
                                                                                      vertex[0].min())
            new_facts = rule_utils.get_discrete_spatial_objs(new_continual_spatial_objs)
            new_satisfied_rules, new_unsatisfied_rules = rule_check(new_facts, learned_rules)

            if (try_counter > 100):
                break

        print(f"tried {try_counter} times")

        # add new positions

        log_manager.visualization(images, prediction, config.categories,
                                  satisfied_rules=satisfied_rules,
                                  unsatisfied_rules=unsatisfied_rules,
                                  facts=facts,
                                  suggested_objs=continual_spatial_objs,
                                  idx=i, prefix="Test", show=False)

        scene_dict = {'scene': rule_utils.objs2scene(continual_spatial_objs, vertex_max[0], vertex_min[0]),
                      "pred_scene": rule_utils.objs2scene(new_continual_spatial_objs, vertex_max[0], vertex_min[0]),
                      'file_name': file_json[0]}
        file_utils.save_json(scene_dict, str(config.output / args.exp / f"Test_output_{i}.json"))
