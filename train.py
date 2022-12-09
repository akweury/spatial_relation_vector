# Created by shaji on 02.12.2022


import torch
from torch.utils.data import DataLoader

from engine import config, pipeline, models, dataset

# preprocessing
batch_size = 2
num_classes = 3
num_epochs = 10
print_freq = 10
dataset_path = config.left_dataset / "train" / "tensor"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_dataset = dataset.SyntheticDataset(dataset_path)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=pipeline.collate_fn)
model = models.get_model_instance_segmentation(num_classes).to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for some epochs
for epoch in range(num_epochs):
    eval_best_loss = 0.0
    # train for one epoch, printing every 10 iterations
    pipeline.train_one_epoch(model, optimizer, train_loader, device, epoch, eval_best_loss, print_freq=10)
    # update the learning rate
    lr_scheduler.step()

    pipeline.evaluation(model, optimizer, train_loader, device, epoch, eval_best_loss, print_freq=10)

print("That's it!")
