""" TRAINING Script Timbre Pitch Disentanglement
"""
# External libs
import argparse
import time
import logging
import os
import json
import sys
import torch
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Internal libs
from lib import create_model
from lib.visualizer import Visualizer
from lib import util
from lib.normalizer import DataNormalizer
from data.nsynth_loader import NSynth

def main():
    # Get the path of the option's file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--opt_file', type=str, default='./options/example.json', help='Full path to options file') 
    args = parser.parse_args()

    # Loading the configuration file
    opt = util.load_json(args.opt_file)
    
    gpu_ids = opt['train']['devices']
    #device = 'cuda:{}'.format(gpu_ids[0])
    # Set cuda if available
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    
    NSynth(opt)  # create a dataset according to the options file


if __name__ == '__main__':
    main()