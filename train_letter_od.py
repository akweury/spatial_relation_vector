# Created by jing at 20.07.23

import os
import torch
from torch.utils.data import DataLoader

from engine import pipeline, models, args_utils, create_dataset
from engine.SyntheticDataset import SyntheticDataset

# preprocessing
args = args_utils.paser()
# init log manager
log_manager = pipeline.LogManager(args=args)

# create_dataset.data2tensorManualMask(log_manager.data_path, args)
create_dataset.data2tensorManualMask(log_manager.data_path, args)

train_dataset = SyntheticDataset(log_manager.data_path, "val", "true")
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)

val_dataset = SyntheticDataset(log_manager.data_path, "val", "true")
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=pipeline.collate_fn)

test_true_dataset = SyntheticDataset(log_manager.data_path, "test", "true")

test_true_loader = DataLoader(test_true_dataset, shuffle=True, batch_size=args.batch_size,
                              collate_fn=pipeline.collate_fn)
test_false_dataset = SyntheticDataset(log_manager.data_path, "test", "false")
test_false_loader = DataLoader(test_false_dataset, shuffle=True, batch_size=args.batch_size,
                               collate_fn=pipeline.collate_fn)

model = models.get_model_instance_segmentation(args.od_classes).to(args.device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

args.num_epochs = 0
# let's train it for some epochs
for epoch in range(args.num_epochs):
    # update the log manager for the new epoch
    log_manager.update(epoch)
    # train for one epoch, printing logs

    pipeline.train_one_epoch(args, model, optimizer, train_loader, log_manager)
    # update the learning rate
    lr_scheduler.step()
    # evaluate the model
    is_best = pipeline.evaluation(model, optimizer, val_loader, log_manager)
    # plot the training & evaluation loss history
    log_manager.plot()
    # Save checkpoint in case evaluation crashed
    pipeline.save_checkpoint(args, is_best, model, optimizer, log_manager)

print("That's it!")

od_model_path = os.path.join(log_manager.model_folder, f'{args.exp}-checkpoint-99.pth.tar')
model_od, optimizer_od, parameters_od = pipeline.load_checkpoint(args.exp, od_model_path, args, device=args.device)
model_od.eval()

print(f"++++++++++++++++ positive {args.subexp} image prediction ++++++++++++++++")
pos_indices = []
positive_predict_counter = 0
negative_predict_counter = 0
for i, (images, targets, categories) in enumerate(test_true_loader):
    with torch.no_grad():
        # input data
        # object detection
        od_predictions = model_od(images)
        for img_i in range(len(od_predictions)):
            od_prediction = od_predictions[img_i]
            od_prediction["boxes"] = od_prediction["boxes"][od_prediction["scores"] > log_manager.args.conf_threshold]
            od_prediction["labels"] = od_prediction["labels"][od_prediction["scores"] > log_manager.args.conf_threshold]
            od_prediction["masks"] = od_prediction["masks"][od_prediction["scores"] > log_manager.args.conf_threshold]
            od_prediction["scores"] = od_prediction["scores"][od_prediction["scores"] > log_manager.args.conf_threshold]

            labels = od_prediction["labels"].to("cpu").numpy()
            if len(labels) > 0:
                positive_predict_counter += 1
print(f"TP : {positive_predict_counter / test_true_loader.dataset.__len__()}, "
      f"{positive_predict_counter}/{test_true_loader.dataset.__len__()}")

print(f"++++++++++++++++ negative {args.subexp} image prediction ++++++++++++++++ ")
neg_indices = []
for i, (images, targets, categories) in enumerate(test_false_loader):
    with torch.no_grad():
        # input data
        od_predictions = model_od(images)
        for img_i in range(len(od_predictions)):
            od_prediction = od_predictions[img_i]
            od_prediction["boxes"] = od_prediction["boxes"][od_prediction["scores"] > log_manager.args.conf_threshold]
            od_prediction["labels"] = od_prediction["labels"][od_prediction["scores"] > log_manager.args.conf_threshold]
            od_prediction["masks"] = od_prediction["masks"][od_prediction["scores"] > log_manager.args.conf_threshold]
            od_prediction["scores"] = od_prediction["scores"][od_prediction["scores"] > log_manager.args.conf_threshold]

            labels = od_prediction["labels"].to("cpu").numpy()
            if len(labels) > 0:
                negative_predict_counter += 1

print(f"FP : {negative_predict_counter / test_false_loader.dataset.__len__()}, "
      f"{negative_predict_counter}/{test_false_loader.dataset.__len__()}")
