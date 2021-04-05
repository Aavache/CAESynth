""" 
TRAINING Script Timbre Pitch Disentanglement by iterating over two datasets, NSynth and FSD.
"""
# External libs
import argparse
import numpy as np
import time
import logging
import os
import json
import sys
import torch
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Internal libs
from lib import create_model
from lib.visualizer import Visualizer
from lib import util
from lib.normalizer import DataNormalizer
from data.dataloaders import NSynthFSD

def updated_losses(loss_accum, new_loss, iter):
    if iter == 0:
        return copy.deepcopy(new_loss)
    else:
        # Merging dictionaries
        return {k: loss_accum.get(k, 0) + new_loss.get(k, 0) for k in set(loss_accum) & set(new_loss)}
    #return loss_accum

def main():
    # Get the path of the option's file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--opt_file', type=str, default='./options/example.json', help='Full path to options file') 
    args = parser.parse_args()

    # Loading the configuration file
    opt = util.load_json(args.opt_file)

    # Set manual seed for reproducibility
    torch.manual_seed(opt['train']['seed'])
    np.random.seed(opt['train']['seed'])

    train_opt = opt['train']
    
    gpu_ids = opt['train']['devices']
    #device = 'cuda:{}'.format(gpu_ids[0])
    # Set cuda if available
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    
    training_size = 36676 # The size of fsd dataset, which is smaller than Nsynth
    validation_size = 5701 # The size of fsd dataset

    # Training set
    fsd_nsynth_trainset = NSynthFSD(opt['data'])  # create a dataset according to the options file
    train_dataloader = torch.utils.data.DataLoader(
                                    fsd_nsynth_trainset,
                                    batch_size=opt['train']['batch_size'],
                                    shuffle=not opt['train']['batch_shuffle'],
                                    num_workers=1)

    # Validation set
    fsd_nsynth_valset = NSynthFSD(opt['data'], is_train=False)  # create a dataset according to the options file
    val_dataloader = torch.utils.data.DataLoader(
                                    fsd_nsynth_valset,
                                    batch_size=1,
                                    shuffle=not opt['train']['batch_shuffle'],
                                    num_workers=1)

    print('# Training Sample = {} | # Evaluation Sample = {}'.format(training_size, validation_size))
    model = create_model(opt, is_train=True)      # create a model according to the options file
    model.load_networks('latest')

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(train_opt['start_epoch'], train_opt['n_epochs'] + 1):    # outer loop for different epochs
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        total_iters = 0   # 1 per sample
        train_iter = 0  # 1 per batch
        train_loss_accum = {}

        model.train()
        train_prg = tqdm(train_dataloader, desc='Bar desc')
        for data in train_prg:  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration

            if total_iters % train_opt['print_freq'] == 0:
                t_data = iter_start_time - iter_data_time
            
            model.set_input(data)         # unpack data from dataset
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses() # TODO: refactor and return losses from opt_parameters or validate

            if train_iter % 5 == 0:# TODO: when using batch 1 is too fast
                train_prg.set_description(visualizer.parse_loss(losses))
                train_prg.refresh()

            train_loss_accum = updated_losses(train_loss_accum, losses, train_iter)
            total_iters += train_opt['batch_size']
            train_iter += 1 #train_opt['batch_size']
            #if total_iters % train_opt['print_freq'] == 0:    # print training losses and save logging information to the disk
            #    t_comp = (time.time() - iter_start_time) / train_opt['batch_size']
            #    visualizer.update_losses(epoch, epoch_iter, losses, t_comp, t_data)
            #    visualizer.plot_current_losses()

            if total_iters % train_opt['save_latest_freq'] == 0:   # cache our latest model every <save_latest_freq> iterations
                #print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')
            iter_data_time = time.time()
        
        # Validation
        val_iter = 0
        val_loss_accum = {}
        model.eval()
        val_prg = tqdm(val_dataloader, desc='Bar desc')
        for data in val_prg:
            model.set_input(data)
            model.validate(data)

            losses = model.get_current_losses()

            val_prg.set_description(visualizer.parse_loss(losses))
            val_prg.refresh()

            val_loss_accum = updated_losses(val_loss_accum, losses, val_iter)
            val_iter += 1
        
        # Averaging the loss values
        train_loss_accum = {k: v/train_iter for k, v in train_loss_accum.items()}
        val_loss_accum = {k: v/val_iter for k, v in val_loss_accum.items()}

        visualizer.update_loss_data(epoch, train_loss_accum, val_loss_accum)
        visualizer.plot_losses()

        if epoch % train_opt['save_epoch_freq'] == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt['n_epochs'], time.time() - epoch_start_time))

if __name__ == '__main__':
    main()