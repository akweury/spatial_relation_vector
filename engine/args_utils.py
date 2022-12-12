# Created by shaji on 12-Dec-22
import os
import argparse
import json
import torch
from pprint import pprint

from engine import config


class Args():
    def __init__(self, args):
        self.machine = args.machine
        self.exp = args.exp
        self.num_classes = args.num_classes
        self.print_freq = args.print_freq
        self.resume = args.resume
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.conf_threshold = args.conf_threshold
        if args.device == "gpu":
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = "cpu"

    def io_path(self):
        if self.machine == "local":
            data_path = config.dataset / self.exp
            output_path = config.output_local / self.exp

        elif self.machine == "remote":
            data_path = config.storage_01 / self.exp
            output_path = config.output_remote / self.exp
        else:
            raise ValueError("Value of Args.machine is incorrect. Please check the arguments.")

        if not os.path.exists(str(data_path)):
            raise ValueError("Dataset Tensors are not generated yet.")

        if not os.path.exists(str(output_path)):
            os.makedirs(str(output_path))

        return data_path, output_path


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

    # Mode selection
    parser.add_argument('--machine', type=str, default="local", help='choose the training machin, local or remote')
    parser.add_argument('--batch_size', '-b', default=2, type=int, help='Mini-batch size (default: 2)')
    parser.add_argument('--exp', '--e', help='Experiment name')
    parser.add_argument('--device', default="cpu", help='Choose device as cpu or gpu')
    parser.add_argument('--num_epochs', default=10, type=int, help='Set number of training epochs')
    parser.add_argument('--num_classes', '-nc', help='Numer of Classes')
    parser.add_argument('--print_freq', '-pf', help='print frequency')
    parser.add_argument('--conf_threshold', help='The confidence threshold of bounding boxes')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')

    # Parse the arguments
    args = parser.parse_args()
    args_path = config.work_place_path / args.exp / 'args.json'
    load_args_from_file(args_path, args)
    print_args(args)
    args = Args(args)

    return args
