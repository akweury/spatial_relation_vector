# Created by shaji on 14-Dec-22

import torch
from torch.utils.data import DataLoader

from engine.SyntheticDataset import SyntheticDataset
from engine.SpatialObject import SpatialObject
from engine import config, pipeline, models, args_utils

# preprocessing
args = args_utils.paser()
# init log manager
log_manager = pipeline.LogManager(args=args)

train_dataset = SyntheticDataset(args.data_path, "train")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=pipeline.collate_fn)


model, optimizer, parameters = pipeline.load_checkpoint(config.model_ball_sphere_detector, args.device)
model.eval()
for i, (images, _, categories) in enumerate(train_loader):
    with torch.no_grad():
        images = list(image.to(args.device) for image in images)
        # visualize the output
        prediction = model(images)
        log_manager.visualization(images, prediction, categories)


