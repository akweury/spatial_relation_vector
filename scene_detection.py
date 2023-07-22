# Created by shaji on 14-Dec-22
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from engine.FactExtractorDataset import FactExtractorDataset
from engine import config, pipeline, args_utils, create_dataset
from engine import rule_utils
import scene_detection_utils

od_model_path = config.model_ball_sphere_detector
dataset_name = "alphabet"
# preprocessing
args = args_utils.paser()
max_obj_num = args.max_e
workplace = Path(os.path.abspath(__file__)).parents[1]
print(f"work place: {workplace}")

for data_type in ['train', 'val', "test"]:

    pos_data_path = workplace / "storage" / dataset_name / args.subexp / data_type / "true"
    neg_data_path = workplace / "storage" / dataset_name / args.subexp / data_type / "false"
    create_dataset.data2tensor_fact_extractor(pos_data_path, args)
    create_dataset.data2tensor_fact_extractor(neg_data_path, args)
    # init log manager
    log_manager = pipeline.LogManager(args=args)

    # create data tensors if they are not exists

    pos_dataset = FactExtractorDataset(pos_data_path)
    neg_dataset = FactExtractorDataset(neg_data_path)
    pos_loader = DataLoader(pos_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)
    neg_loader = DataLoader(neg_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)

    neg_pred = torch.zeros(size=(neg_dataset.__len__(), max_obj_num, 11))
    pos_pred = torch.zeros(size=(pos_dataset.__len__(), max_obj_num, 11))

    pos_names = []
    neg_names = []

    categories = config.categories

    model_od, optimizer_od, parameters_od = pipeline.load_checkpoint("od", od_model_path, args, device=args.device)
    model_od.eval()

    print(f"++++++++++++++++ positive {data_type} image prediction ++++++++++++++++")
    pos_indices = []
    for i, (data, objects, _, _, _, json_file) in enumerate(pos_loader):

        with torch.no_grad():
            # input data

            images = list((_data[3:] / 255).to(args.device) for _data in data)
            vertex = list((_data[:3]).to(args.device) for _data in data)

            # object detection
            od_prediction = model_od(images)

            # fact extractor
            continual_spatial_objs = rule_utils.get_continual_spatial_objs(f"{data_type}_positive_{i}", od_prediction,
                                                                           images,
                                                                           vertex, objects, log_manager)
            if continual_spatial_objs is None:

                continue
            else:
                pos_indices.append(i)
                pos_names.append(json_file)
            # [x,y,z, color1, color2, color3, shape1, shape2]
            pos_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0][:max_obj_num], max_obj_num)
    pos_pred = pos_pred[pos_indices]
    print(f"++++++++++++++++ negative {data_type} image prediction ++++++++++++++++ ")
    neg_indices = []
    for i, (data, objects, _, _, _, json_file) in enumerate(neg_loader):
        with torch.no_grad():
            # input data
            images = list((_data[3:] / 255).to(args.device) for _data in data)
            vertex = list((_data[:3]).to(args.device) for _data in data)

            # object detection
            od_prediction = model_od(images)

            # fact extractor
            continual_spatial_objs = rule_utils.get_continual_spatial_objs(f"{data_type}_negative_{i}", od_prediction,
                                                                           images,
                                                                           vertex, objects, log_manager)
            if continual_spatial_objs is None:
                continue
            else:
                neg_indices.append(i)
                neg_names.append(json_file)
            # [x,y,z, color1, color2, color3, shape1, shape2, conf]
            neg_pred[i, :] = scene_detection_utils.obj2tensor(continual_spatial_objs[0][:max_obj_num], max_obj_num)
    neg_pred = neg_pred[neg_indices]

    prediction_dict = {
        'pos_res': pos_pred.detach(),
        'neg_res': neg_pred.detach(),
        'pos_names': pos_names,
        'neg_names': neg_names
    }
    model_file = str(config.storage / dataset_name / f"{args.subexp}" / f"{args.subexp}_pm_res_{data_type}.pth.tar")
    torch.save(prediction_dict, str(model_file))
    print(f"file {model_file} saved successfully!")
