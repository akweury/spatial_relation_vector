# Created by shaji on 14-Dec-22

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from engine.FactExtractorDataset import FactExtractorDataset
from engine import config, pipeline, args_utils, create_dataset, models
from engine import rule_utils
import scene_detection_utils

od_model_path = config.model_ball_sphere_detector
max_obj_num = 6
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

    neg_pred = torch.zeros(size=(neg_dataset.__len__(), max_obj_num, 9))
    pos_pred = torch.zeros(size=(pos_dataset.__len__(), max_obj_num, 9))

    categories = config.categories

    model_od, optimizer_od, parameters_od = pipeline.load_checkpoint("od", od_model_path, args, device=args.device)
    model_od.eval()

    for i, (data, objects, _, _, _) in enumerate(pos_loader):
        with torch.no_grad():
            # input data
            print(args.device)
            images = list((_data[3:] / 255).to(args.device) for _data in data)
            vertex = list((_data[:3]).to(args.device) for _data in data)

            # object detection
            od_prediction = model_od(images)
            if len(od_prediction["labels"])<args.e:
                continue
            # fact extractor
            continual_spatial_objs = rule_utils.get_continual_spatial_objs(f"positive_{i}", od_prediction, images,
                                                                           vertex,
                                                                           objects, log_manager)
            # [x,y,z, color1, color2, color3, shape1, shape2]
            pos_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0][:max_obj_num], max_obj_num)

    for i, (data, objects, _, _, _) in enumerate(neg_loader):
        with torch.no_grad():
            # input data
            images = list((_data[3:] / 255).to(args.device) for _data in data)
            vertex = list((_data[:3]).to(args.device) for _data in data)

            # object detection
            od_prediction = model_od(images)

            # fact extractor
            continual_spatial_objs = rule_utils.get_continual_spatial_objs(f"negative_{i}", od_prediction, images,
                                                                           vertex, objects, log_manager)
            # [x,y,z, color1, color2, color3, shape1, shape2, conf]
            neg_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0][:max_obj_num], max_obj_num)

    prediction_dict = {
        'pos_res': pos_pred.detach(),
        'neg_res': neg_pred.detach()
    }
    model_file = str(config.storage / 'hide' / f"{args.subexp}_pm_res_{data_type}.pth.tar")
    torch.save(prediction_dict, str(model_file))
    print(f"file {model_file} saved successfully!")
