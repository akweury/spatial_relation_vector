# Created by shaji on 14-Dec-22

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from engine.FactExtractorDataset import FactExtractorDataset
from engine import config, pipeline, args_utils, create_dataset
from engine import rule_utils
import scene_detection_utils

od_model_path = config.model_ball_sphere_detector

# preprocessing
args = args_utils.paser()
workplace = Path(__file__).parents[1]
print(f"work place: {workplace}")
pos_data_path = workplace / "storage" / "dataset" / "triangle_3" / "true"
neg_data_path = workplace / "storage" / "dataset" / "triangle_3" / "false"
create_dataset.data2tensor_fact_extractor(pos_data_path, args)
create_dataset.data2tensor_fact_extractor(neg_data_path, args)
# init log manager
log_manager = pipeline.LogManager(args=args)

# create data tensors if they are not exists

pos_dataset = FactExtractorDataset(pos_data_path)
neg_dataset = FactExtractorDataset(neg_data_path)
pos_loader = DataLoader(pos_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)
neg_loader = DataLoader(neg_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)

neg_pred = torch.zeros(size=(pos_dataset.__len__(), 6, 8))
pos_pred = torch.zeros(size=(pos_dataset.__len__(), 6, 8))

categories = config.categories

model_od, optimizer, parameters = pipeline.load_checkpoint(od_model_path)
model_od.eval()

for i, (data, objects, _, _, _) in enumerate(pos_loader):
    with torch.no_grad():
        # input data
        images = list((_data[3:] / 255).to(args.device) for _data in data)
        vertex = list((_data[:3]).to(args.device) for _data in data)

        # object detection
        prediction = model_od(images)

        # fact extractor
        continual_spatial_objs = rule_utils.get_continual_spatial_objs(prediction, images, vertex, objects, log_manager)
        # [x,y,z, color1, color2, color3, shape1, shape2]
        pos_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0])

for i, (data, objects, _, _, _) in enumerate(neg_loader):
    with torch.no_grad():
        # input data
        images = list((_data[3:] / 255).to(args.device) for _data in data)
        vertex = list((_data[:3]).to(args.device) for _data in data)

        # object detection
        prediction = model_od(images)

        # fact extractor
        continual_spatial_objs = rule_utils.get_continual_spatial_objs(prediction, images, vertex, objects, log_manager)
        # [x,y,z, color1, color2, color3, shape1, shape2]
        neg_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0])

prediction_dict = {
    'pos_res': pos_pred.detach(),
    'neg_res': neg_pred.detach()
}
model_file = str(config.storage / f"{args.dataset}_pm_res_val.pth.tar")
torch.save(prediction_dict, str(model_file))
