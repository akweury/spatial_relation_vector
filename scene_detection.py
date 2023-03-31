# Created by shaji on 14-Dec-22

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from engine.FactExtractorDataset import FactExtractorDataset
from engine import config, pipeline, args_utils, create_dataset
from engine import rule_utils
import scene_detection_utils

od_model_path = config.model_ball_sphere_detector
cd_model_path = config.model_rgb_color_detector

# preprocessing
args = args_utils.paser()
workplace = Path(__file__).parents[0]
print(f"work place: {workplace}")

for data_type in ['train', 'val', "test"]:

    pos_data_path = workplace / "storage" / "hide" / args.subexp / data_type / "true"
    neg_data_path = workplace / "storage" / "hide" / args.subexp / data_type / "false"
    create_dataset.data2tensor_fact_extractor(pos_data_path, args)
    create_dataset.data2tensor_fact_extractor(neg_data_path, args)
    # init log manager
    log_manager = pipeline.LogManager(args=args)

    # create data tensors if they are not exists

    pos_dataset = FactExtractorDataset(pos_data_path)
    neg_dataset = FactExtractorDataset(neg_data_path)
    pos_loader = DataLoader(pos_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)
    neg_loader = DataLoader(neg_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)

    neg_pred = torch.zeros(size=(neg_dataset.__len__(), 6, 9))
    pos_pred = torch.zeros(size=(pos_dataset.__len__(), 6, 9))

    categories = config.categories

    model_od, optimizer_od, parameters_od = pipeline.load_checkpoint("od", od_model_path, args, device=args.device)
    model_cd, optimizer_cd, parameters_cd = pipeline.load_checkpoint("cd", cd_model_path, args, device=args.device)
    model_od.eval()
    model_cd.eval()

    for i, (data, objects, _, _, _) in enumerate(pos_loader):
        with torch.no_grad():
            # input data
            print(args.device)
            images = list((_data[3:] / 255).to(args.device) for _data in data)
            vertex = list((_data[:3]).to(args.device) for _data in data)

            # object detection
            od_prediction = model_od(images)
            cd_prediction = model_cd(images)

            # fact extractor
            continual_spatial_objs = rule_utils.get_continual_spatial_objs(od_prediction, cd_prediction, images, vertex,
                                                                           objects, log_manager)
            # [x,y,z, color1, color2, color3, shape1, shape2]
            pos_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0][:10])

    for i, (data, objects, _, _, _) in enumerate(neg_loader):
        with torch.no_grad():
            # input data
            images = list((_data[3:] / 255).to(args.device) for _data in data)
            vertex = list((_data[:3]).to(args.device) for _data in data)

            # object detection
            od_prediction = model_od(images)

            # fact extractor
            continual_spatial_objs = rule_utils.get_continual_spatial_objs(od_prediction, images, vertex, objects,
                                                                           log_manager)
            # [x,y,z, color1, color2, color3, shape1, shape2, conf]
            neg_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0][:10])

    prediction_dict = {
        'pos_res': pos_pred.detach(),
        'neg_res': neg_pred.detach()
    }
    model_file = str(config.storage / 'hide' / f"{args.exp}_pm_res_{data_type}.pth.tar")
    torch.save(prediction_dict, str(model_file))
