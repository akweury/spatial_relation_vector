# Created by shaji on 02.12.2022

import datetime
import torch
from torch.utils.data import DataLoader

from engine import config, pipeline, models, args_utils
from engine.SyntheticDataset import SyntheticDataset

# preprocessing
args = args_utils.paser()

dataset_path, output_path = args.io_path()

train_dataset = SyntheticDataset(dataset_path, "train")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)
test_dataset = SyntheticDataset(dataset_path, "test")
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)

model = models.get_model_instance_segmentation(args.num_classes).to(args.device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# init log manager
log_manager = pipeline.LogManager(delimiter="  ",
                                  device=args.device,
                                  num_epochs=args.num_epochs,
                                  print_freq=args.print_freq,
                                  lr=optimizer.param_groups[0]['lr'],
                                  batch_size=args.batch_size,
                                  output_folder=output_path,
                                  conf_threshold=args.conf_threshold)
# let's train it for some epochs
for epoch in range(args.num_epochs):
    # update the log manager for the new epoch
    log_manager.update(epoch)
    # train for one epoch, printing logs

    pipeline.train_one_epoch(model, optimizer, train_loader, log_manager)
    # update the learning rate
    lr_scheduler.step()
    # evaluate the model
    is_best = pipeline.evaluation(model, optimizer, test_loader, log_manager)
    # plot the training & evaluation loss history
    log_manager.plot()
    # Save checkpoint in case evaluation crashed
    pipeline.save_checkpoint(is_best, epoch)

print("That's it!")
