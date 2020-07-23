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

    train_opt = opt['train']
    
    gpu_ids = opt['train']['devices']
    device = 'cuda:{}'.format(gpu_ids[0])
    # Set cuda if available
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    
    dataset = NSynth(opt)  # create a dataset according to the options file
    data_norm = DataNormalizer(dataset, device)
    dataloader = torch.utils.data.DataLoader(
                                    dataset,
                                    batch_size=opt['train']['batch_size'],
                                    shuffle=not opt['train']['batch_shuffle'],
                                    num_workers=int(opt['train']['n_threads']))
    

    dataset_size = len(dataset)                   # get the number of images in the dataset.
    print('The number of training images = {}'.format(dataset_size))
    model = create_model(opt, is_train=True)      # create a model according to the options file
    #model.load_networks('latest')

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(train_opt['start_epoch'], train_opt['n_epochs'] + 1):    # outer loop for different epochs
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        prg = tqdm(dataloader, desc='Bar desc')
        # for i, data in enumerate(dataset):  # inner loop within one epoch
        for data in prg:
            iter_start_time = time.time()  # timer for computation per iteration

            if total_iters % train_opt['print_freq'] == 0:
                t_data = iter_start_time - iter_data_time
                
            data['src_data'] = data['src_data'].to(device)
            data['trg_data'] = data['trg_data'].to(device)
            data['src_pitch'] = data['src_pitch'].to(device)
            data['trg_pitch'] = data['trg_pitch'].to(device)
            total_iters += train_opt['batch_size']
            epoch_iter += train_opt['batch_size']
            
            data['src_data'] = data_norm.normalize(data['src_data'])
            data['trg_data'] = data_norm.normalize(data['trg_data'])
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            prg.set_description(visualizer.parse_loss(losses))
            prg.refresh()
            if total_iters % train_opt['print_freq'] == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / train_opt['batch_size']
                visualizer.update_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses()

            if total_iters % train_opt['save_latest_freq'] == 0:   # cache our latest model every <save_latest_freq> iterations
                #print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')
            iter_data_time = time.time()
            
        if epoch % train_opt['save_epoch_freq'] == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt['n_epochs'], time.time() - epoch_start_time))

if __name__ == '__main__':
    main()