# Created by shaji on 02.12.2022

import torch
from torch.utils.data import DataLoader

from engine import pipeline, models, args_utils, create_dataset
from engine.SyntheticDataset import SyntheticDataset

# preprocessing
args = args_utils.paser()
# init log manager
log_manager = pipeline.LogManager(args=args)

# create_dataset.data2tensorManualMask(log_manager.data_path, args)
create_dataset.data2tensorAutoMask(log_manager.data_path, args)

train_dataset = SyntheticDataset(log_manager.data_path, "train")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          collate_fn=pipeline.collate_fn)
test_dataset = SyntheticDataset(log_manager.data_path, "test")
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size,
                         collate_fn=pipeline.collate_fn)

if args.exp == "od":
    model = models.get_model_instance_segmentation(args.od_classes).to(args.device)
elif args.exp == "cd":
    model = models.get_model_instance_segmentation(args.cd_classes).to(args.device)
# elif args.exp == "pd":
#     model = models.get_model_instance_segmentation(args.num_classes).to(args.device)
else:
    raise ValueError

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for some epochs
for epoch in range(args.num_epochs):
    # update the log manager for the new epoch
    log_manager.update(epoch)
    # train for one epoch, printing logs

    pipeline.train_one_epoch(args, model, optimizer, train_loader, log_manager)
    # update the learning rate
    lr_scheduler.step()
    # evaluate the model
    is_best = pipeline.evaluation(model, optimizer, test_loader, log_manager)
    # plot the training & evaluation loss history
    log_manager.plot()
    # Save checkpoint in case evaluation crashed
    pipeline.save_checkpoint(is_best, model, optimizer, log_manager)

print("That's it!")
