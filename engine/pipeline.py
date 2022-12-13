import math
import sys
from collections import defaultdict, deque
import datetime
import time

import cv2 as cv
import numpy as np
import torch
import torch.distributed as dist
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from engine import plot_utils, config


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def collate_fn(batch):
    return tuple(zip(*batch))


class LogManager():
    def __init__(self, delimiter="\t", device="cpu", num_epochs=10,
                 print_freq=10, lr=None, batch_size=2, output_folder=None, conf_threshold=0.5):
        self.epoch = None
        self.device = device
        self.date_start = datetime.datetime.today().date()
        self.time_start = datetime.datetime.now().strftime("%H:%M:%S")
        self.date_now = None
        self.time_now = None
        self.best_loss = 1e+10
        self.delimiter = delimiter
        self.print_freq = print_freq
        self.lr = lr
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.eval_losses = np.zeros((1, num_epochs))
        self.train_losses = np.zeros((1, num_epochs))
        self.conf_threshold = conf_threshold

    def print_new_epoch(self):
        print(
            f"{self.time_now} "
            f"Epoch [{self.epoch}] lr={self.lr} Best eval loss: {float(self.best_loss):.1e} "
            f"Start from {self.date_start}-{self.time_start}")

    def stop_training(self, loss_value, loss_dict_reduced):
        print("Loss is {}, stopping training".format(loss_value))
        print(loss_dict_reduced)
        sys.exit(1)

    def print_loss(self, prefix, loss_value, batch_size):
        loss_avg = loss_value / int(batch_size)
        print(f"\t {prefix} loss: {loss_avg:.2e}")

    def update(self, epoch):
        self.epoch = epoch
        self.date_now = datetime.datetime.today().date()
        self.time_now = datetime.datetime.now().strftime("%H:%M:%S")
        self.print_new_epoch()

    def plot(self):
        # draw line chart for training
        plot_utils.draw_line_chart(self.train_losses, self.output_folder, self.date_start, self.time_start,
                                   log_y=True, label="train_loss", epoch=self.epoch, start_epoch=0, title="train"
                                                                                                          "_loss",
                                   cla_leg=True)

        # draw line chart for evaluation
        plot_utils.draw_line_chart(self.eval_losses, self.output_folder, self.date_start, self.time_start,
                                   log_y=True, label="eval_loss", epoch=self.epoch, start_epoch=0, title="eval_loss",
                                   cla_leg=True)

    def visualization(self, images, img_preds, categories):
        img_tensor_int = []
        for image in images:
            img_tensor_int.append((image*255).to(dtype=torch.uint8))

        img_preds[0]["boxes"] = img_preds[0]["boxes"][img_preds[0]["scores"] > self.conf_threshold]
        img_preds[0]["labels"] = img_preds[0]["labels"][img_preds[0]["scores"] > self.conf_threshold]
        img_preds[0]["masks"] = img_preds[0]["masks"][img_preds[0]["scores"] > self.conf_threshold]
        img_preds[0]["scores"] = img_preds[0]["scores"][img_preds[0]["scores"] > self.conf_threshold]

        img_labels = img_preds[0]["labels"].to("cpu").numpy()
        img_annot_labels = [f"{categories[label]}: {prob:.2f}" for label, prob in
                            zip(img_labels, img_preds[0]["scores"].detach().numpy())]
        colors=config.colors[:len(categories)]
        img_output_tensor = draw_bounding_boxes(image=img_tensor_int[0],
                                                boxes=img_preds[0]["boxes"],
                                                labels=img_annot_labels,
                                                colors=colors,
                                                width=2)

        img_masks_float = img_preds[0]["masks"].squeeze(1)
        img_masks_float[img_masks_float < 0.8] = 0
        img_masks_bool = img_masks_float.bool()
        img_output_tensor = draw_segmentation_masks(img_output_tensor, masks=img_masks_bool, alpha=0.8)
        img_output = to_pil_image(img_output_tensor)
        img_output.save(str(self.output_folder / f"output_{self.epoch}_0.png"), "PNG")


def train_one_epoch(model, optimizer, train_loader, log_manager):
    loss_sum = 0.0
    # training
    model.train()
    # set lr_scheduler
    if log_manager.epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

    for i, (images, targets, categories) in enumerate(train_loader):
        # for images, targets in metric_logger.log_every(train_loader, print_freq, header):
        images = list(image.to(log_manager.device) for image in images)
        targets = [{k: v.to(log_manager.device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        loss_sum += loss_value

        # stop training if loss is not finite anymore.
        if np.sum(loss_value) > 1e+4:
            print("loss is greater than 1e+4.")
            break
        if not math.isfinite(loss_value):
            log_manager.stop_training(loss_value, loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    # print loss
    log_manager.train_losses[0, log_manager.epoch] = loss_sum / len(train_loader)
    log_manager.print_loss("training", log_manager.train_losses[0, log_manager.epoch], log_manager.batch_size)


def evaluation(model, optimizer, test_loader, log_manager):
    is_best = False
    loss_sum = 0.0
    for i, (images, targets, categories) in enumerate(test_loader):
        with torch.no_grad():
            model.train()
            images = list(image.to(log_manager.device) for image in images)
            targets = [{k: v.to(log_manager.device) for k, v in t.items()} for t in targets]
            # Wait for all kernels to finish
            # torch.cuda.synchronize()
            # start count the model time
            start = time.time()
            # Forward pass
            loss_dict = model(images, targets)
            # record data load time
            gpu_time = time.time() - start
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            loss_sum += loss_value

            # visualize the output
            model.eval()
            prediction = model(images)
            log_manager.visualization(images, prediction, categories)

    # print loss
    log_manager.eval_losses[0, log_manager.epoch] = loss_sum / len(test_loader)
    log_manager.print_loss("eval", log_manager.eval_losses[0, log_manager.epoch], log_manager.batch_size)

    # update the best loss
    if loss_value < log_manager.best_loss:
        log_manager.best_loss = loss_value
        is_best = True

    return is_best


def save_checkpoint(is_best, epoch):
    return None
