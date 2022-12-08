# Created by shaji at 02.12.2022

import math
import sys
import torch
from torch.utils.data import DataLoader

import dataset
import config
import models
import dataset_utils

# preprocessing
dataset_path = config.left_dataset / "train" / "tensor"
train_dataset = dataset.SyntheticDataset(dataset_path)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=dataset_utils.collate_fn)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 3
model = models.get_model_instance_segmentation(num_classes)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10
print_freq = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations

    model.train()
    metric_logger = dataset_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dataset_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'


    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = dataset_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

    for i, (images, targets) in enumerate(train_loader):
    # for images, targets in metric_logger.log_every(train_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dataset_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

print("That's it!")
