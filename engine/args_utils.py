# Created by shaji on 12-Dec-22
import os
import argparse
import json
import torch
from pprint import pprint

from engine import config


class Args():
    def __init__(self, args):
        self.dataset = args.dataset
        self.machine = args.machine
        self.clear = args.clear
        self.exp = args.exp
        self.od_classes = args.od_classes
        self.cd_classes = args.cd_classes
        self.print_freq = args.print_freq
        self.resume = args.resume
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.conf_threshold = args.conf_threshold
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.subexp = args.subexp
        self.device = args.device
        self.e = args.e


def load_args_from_file(args_file_path, given_args):
    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)

        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            if key not in ['workspace', 'exp', 'evaluate', 'resume', 'gpu']:  # Do not overwrite these keys
                setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))


def print_args(args):
    pprint(f'==> Experiment Args:  {args} ')


def paser():
    parser = argparse.ArgumentParser()

    # fine tune args
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--batch_size', '-b', default=2, type=int, help='Mini-batch size (default: 2)')
    parser.add_argument('--num_epochs', default=10, type=int, help='Set number of training epochs')
    parser.add_argument('--conf_threshold', default=0.7, help='The confidence threshold of bounding boxes')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight Decay')

    # configuration args
    parser.add_argument('--clear', type=str, default="false", help='Clear tensors in dataset folders.')
    parser.add_argument('--od_classes', type=int, help='Numer of Object Detection Classes')
    parser.add_argument('--cd_classes', type=int, help='Numer of Color Detection Classes')
    parser.add_argument('--machine', type=str, default="local", help='choose the training machin, local or remote')
    parser.add_argument('--exp', '-e', help='Experiment name')
    parser.add_argument('--subexp', help='Sub-experiment name')
    parser.add_argument('--e', type=int, help='maximum number of objects in each scene')
    parser.add_argument('--device', default="cpu", help='Choose device as cpu or gpu')
    parser.add_argument('--print_freq', '-pf', help='print frequency')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')

    # Parse the arguments
    args = parser.parse_args()
    args_path = config.work_place_path / args.exp / 'args.json'
    load_args_from_file(args_path, args)
    if not args.device == "cpu":
        args.device = int(args.device)
    print_args(args)
    args = Args(args)

    return args
