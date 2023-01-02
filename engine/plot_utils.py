# Created by shaji on 09-Dec-22

import datetime
import json
import os
import shutil
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from engine import config

def draw_line_chart(data_1, path, date_now, time_now,
                    title=None, x_label=None, y_label=None, show=False, log_y=False,
                    label=None, epoch=None, cla_leg=False, start_epoch=0, loss_type="mse"):
    if data_1.shape[1] <= 1:
        return

    x = np.arange(epoch - start_epoch)
    y = data_1[0, start_epoch:epoch]
    x = x[y.nonzero()]
    y = y[y.nonzero()]
    plt.plot(x, y, label=label)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)
    if not os.path.exists(str(path)):
        os.mkdir(path)

    if loss_type == "mse":
        plt.savefig(str(Path(path) / f"line_{title}_{x_label}_{y_label}_{date_now}_{time_now}.png"))
    elif loss_type == "angle":
        plt.savefig(str(Path(path) / f"line_{title}_{x_label}_{y_label}_{date_now}_{time_now}_angle.png"))
    else:
        raise ValueError("loss type is not supported.")

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def addText(img, text, pos='upper_left', font_size=1.6, color=(255, 255, 255), thickness=1):
    h, w = img.shape[:2]
    if pos == 'upper_left':
        position = (10, 50)
    elif pos == 'upper_right':
        position = (w - 250, 80)
    elif pos == 'lower_right':
        position = (h - 200, w - 20)
    elif pos == 'lower_left':
        position = (10, w - 20)
    else:
        raise ValueError('unsupported position to put text in the image.')

    cv.putText(img, text=text, org=position,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=thickness, lineType=cv.LINE_AA)


def visual_img(img, name, upper_right=None, font_scale=0.8):
    img = image_resize(img, width=512, height=512)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    if img.ndim == 2:
        img = cv.merge((img, img, img))

    addText(img, f"{name}", font_size=font_scale)
    if upper_right is not None:
        addText(img, f"{upper_right}", pos="upper_right", font_size=0.65)
    return img


def addTextPIL(img, text, pos, color=(255, 255, 255)):
    draw = ImageDraw.Draw(img)
    x_pos, y_pos = pos
    draw.text(pos, text, color)
    y_pos += 10
    return img, y_pos


def addRulePIL(img, rule, pos):
    draw = ImageDraw.Draw(img)
    ref_text = ""
    for property in rule["premise"]["ref"]:
        ref_text += property.value + " "
    obj_text = ""
    for property in rule["premise"]["obj"]:
        obj_text += property.value + " "

    conclusion_text = ""
    for key in rule['conclusion'].keys():
        conclusion_text += rule['conclusion'][key]  + " "

    ruleText = f'{ref_text} {conclusion_text} {obj_text} (freq:{rule["freq"]})'

    draw.text(pos, ruleText, (255, 255, 255))
    x_pos, y_pos = pos
    y_pos += 10
    return img, y_pos


def maskRCNNVisualization(img, img_pred, threshold, categories):
    img_pred["boxes"] = img_pred["boxes"][img_pred["scores"] > threshold]
    img_pred["labels"] = img_pred["labels"][img_pred["scores"] > threshold]
    img_pred["masks"] = img_pred["masks"][img_pred["scores"] > threshold]
    img_pred["scores"] = img_pred["scores"][img_pred["scores"] > threshold]

    img_labels = img_pred["labels"].to("cpu").numpy()
    # print(f"{len(img_labels)} objects has been detected.")
    labels_with_prob = zip(img_labels, img_pred["scores"].detach().to("cpu").numpy())
    img_annot_labels = []
    for label, prob in labels_with_prob:
        print(f"categories: {categories}, label: {label}, prob: {prob:.2f}")
        img_annot_labels.append(f"{categories[label]}: {prob:.2f}")

    colors = [config.colors[i] for i in img_labels]
    img_output_tensor = draw_bounding_boxes(image=img,
                                            boxes=img_pred["boxes"],
                                            labels=img_annot_labels,
                                            colors=colors,
                                            width=2)

    img_masks_float = img_pred["masks"].squeeze(1)
    img_masks_float[img_masks_float < 0.8] = 0
    img_masks_bool = img_masks_float.bool()
    if img_masks_bool.size(0) > 0:
        img_output_tensor = draw_segmentation_masks(img_output_tensor, masks=img_masks_bool, alpha=0.2)
    img_output = to_pil_image(img_output_tensor)

    return img_output


def printRules(img_output, satisfied_rules, unsatisfied_rules, learned_rules):
    text_y_pos = 10
    img_output, text_y_pos = addTextPIL(img_output, "satisfied_rules", (10, text_y_pos),
                                                   color=(255, 0, 0))
    if satisfied_rules is not None:
        for ruleIdx in range(len(satisfied_rules)):
            img_output, text_y_pos = addRulePIL(img_output, satisfied_rules[ruleIdx],
                                                           (10, text_y_pos))
    img_output, text_y_pos = addTextPIL(img_output, "unsatisfied_rules", (10, text_y_pos),
                                                   color=(255, 0, 0))
    if unsatisfied_rules is not None:
        for ruleIdx in range(len(unsatisfied_rules)):
            img_output, text_y_pos = addRulePIL(img_output, unsatisfied_rules[ruleIdx],
                                                           (10, text_y_pos))

    img_output, text_y_pos = addTextPIL(img_output, "learned_rules", (10, text_y_pos),
                                                   color=(255, 0, 0))
    if learned_rules is not None:
        for ruleIdx in range(len(learned_rules)):
            img_output, text_y_pos = addRulePIL(img_output, learned_rules[ruleIdx], (10, text_y_pos))

    return img_output, text_y_pos


def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst

def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height),resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst