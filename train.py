""" 
TRAINING Script for CAESynth: Real-time timbre interpolation and pitch control."""
# External libs
import argparse
import numpy as np
import time
import torch
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Internal libs
from lib.models import create_model
from data import create_dataset
from lib.visualizer import Visualizer
from lib import util

def updated_losses(loss_accum, new_loss, iter):
    if iter == 0:
        return copy.deepcopy(new_loss)
    else:
        # Merging dictionaries
        return {k: loss_accum.get(k, 0) + new_loss.get(k, 0) for k in set(loss_accum) & set(new_loss)}

def main(opt):
    # Set manual seed for reproducibility
    torch.manual_seed(opt['train']['seed'])
    np.random.seed(opt['train']['seed'])

    train_opt = opt['train']
    
    gpu_ids = opt['train']['devices']
    # Set cuda if available
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    
    # Training set
    trainset = create_dataset(opt['data'], is_train=True)
    train_dataloader = torch.utils.data.DataLoader(
                                    trainset,
                                    batch_size=opt['train']['batch_size'],
                                    shuffle=not opt['train']['batch_shuffle'],
                                    num_workers=int(opt['train']['n_threads']))

    # Validation set
    validset = create_dataset(opt['data'], is_train=False)
    valid_dataloader = torch.utils.data.DataLoader(
                                    validset,
                                    batch_size=1,
                                    shuffle=not opt['train']['batch_shuffle'],
                                    num_workers=int(opt['train']['n_threads']))

    model = create_model(opt, is_train=True) # create a model according to the config file
    visualizer = Visualizer(opt)  # create a visualizer that display/save training progression plots

    for epoch in range(train_opt['start_epoch'], train_opt['n_epochs'] + 1):
        epoch_start_time = time.time()
        total_iters, train_iter = 0, 0  
        train_loss_accum = {}

        model.train()
        train_prg = tqdm(train_dataloader, desc='Bar desc')
        for data in train_prg:
            model.set_input(data)  # unpack data from dataset
            model.optimize_parameters()   
            losses = model.get_current_losses()

            if train_iter % 16 == 0: # TODO: when using batch 1 is too fast
                train_prg.set_description(visualizer.parse_loss(losses))
                train_prg.refresh()

            train_loss_accum = updated_losses(train_loss_accum, losses, train_iter)
            total_iters += train_opt['batch_size']
            train_iter += 1 # train_opt['batch_size']

            if total_iters % train_opt['save_latest_freq'] == 0:
                model.save_networks('latest')
        
        # Validation
        val_iter = 0
        val_loss_accum = {}
        model.eval()
        val_prg = tqdm(valid_dataloader, desc='Bar desc')
        for data in val_prg:
            model.set_input(data)
            model.validate()
            losses = model.get_current_losses()

            val_prg.set_description(visualizer.parse_loss(losses))
            val_prg.refresh()

            val_loss_accum = updated_losses(val_loss_accum, losses, val_iter)
            val_iter += 1
        
        # Averaging loss items
        train_loss_accum = {k: v/train_iter for k, v in train_loss_accum.items()}
        val_loss_accum = {k: v/val_iter for k, v in val_loss_accum.items()}

        visualizer.update_loss_data(epoch, train_loss_accum, val_loss_accum)
        visualizer.plot_losses()

        if epoch % train_opt['save_epoch_freq'] == 0: # Saving checkpoints
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt['n_epochs'], time.time() - epoch_start_time))

if __name__ == '__main__':
    # Get the path of the option's file
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--opt_file', type=str, default='./options/example.json', help='Full path to options file') 
    args = parser.parse_args()

    # Loading the configuration file
    opt = util.load_json(args.opt_file)

    main(opt)