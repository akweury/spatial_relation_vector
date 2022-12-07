# Created by shaji at 02.12.2022
import torch
from torch.utils.data import DataLoader

import dataset
import config
import models
from reference import engine

# preprocessing
dataset_path = config.left_dataset /"train" / "tensor"
train_dataset = dataset.SyntheticDataset(dataset_path)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)
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
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    engine.train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

    print("That's it!")

for i, (image, target) in enumerate(train_loader):
    input_tensor, loss = image.float(), torch.tensor([0.0])

