# Created by shaji on 14-Dec-22

import torch
from torch.utils.data import DataLoader

from engine.SyntheticDataset import SyntheticDataset
from engine.SpatialObject import SpatialObject
from engine import config, pipeline, models, args_utils

# preprocessing
args = args_utils.paser()
# init log manager
log_manager = pipeline.LogManager(model_exp="object_detector_big", args=args)

train_dataset = SyntheticDataset(log_manager.data_path, "train")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=pipeline.collate_fn)

model = models.get_model_instance_segmentation(args.num_classes).to(args.device)
model, optimizer, parameters = pipeline.load_checkpoint(config.model_ball_sphere_detector, args, model)
model.eval()
for i, (images, _, categories) in enumerate(train_loader):
    with torch.no_grad():
        images = list(image.to(args.device) for image in images)
        # visualize the output
        prediction = model(images)
        log_manager.visualization(images, prediction, categories, idx=i)
