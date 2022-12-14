# Created by shaji on 14-Dec-22

import torch
from torch.utils.data import DataLoader

from engine.FactExtractorDataset import FactExtractorDataset
from engine.SpatialObject import SpatialObject
from engine import config, pipeline, models, args_utils
import create_dataset

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

model = models.get_model_instance_segmentation(args.num_classes).to(args.device)
model, optimizer, parameters = pipeline.load_checkpoint(config.model_ball_sphere_detector, args, model)
model.eval()
for i, (images, item) in enumerate(train_loader):
    with torch.no_grad():
        images = list(image.to(args.device) for image in images)

        # visualize the output
        prediction = model(images)
        log_manager.visualization(images, prediction, categories, idx=i)
