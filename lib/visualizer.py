# External libs
import os
import math
import pandas as pd
import time
import logging
import matplotlib.pyplot as plt
# Internal libs
from lib import util

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class
        """
        self.opt = opt  # cache the option
        self.train_opt = opt['train']
        #self.display_id = opt.display_id
        self.name = opt['experiment_name']
        self.saved = False

        self.dir = os.path.join(self.train_opt['checkpoints_dir'], opt['experiment_name'])
        util.mkdirs(self.dir)

        # create a logging file to store training losses
        self.plot_data = {}
        self.train_data = {}
        self.val_data = {}
        self.ignore_keys = ['iters', 't_comp', 't_data'] # This keys will be ingored during plotting
        self.df_name = os.path.join(self.dir, '{}.csv'.format(opt['experiment_name'])) # Dataframe where the loss records will be saved.
        self.plot_name = os.path.join(self.dir, 'plot_{}.png'.format(opt['experiment_name']))

        # Creating a training logging file
        self.log_name = os.path.join(self.dir, 'log_loss.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def update_losses(self, epoch, iters, losses, t_comp, t_data):
        """updates the losses records with a new entry.

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        if self.plot_data is None:
            self.plot_data = {}

        # Adding losses to the data record
        if (len(self.plot_data.keys()) == 0):
            self.plot_data['epoch'] = [epoch]
            self.plot_data['iters'] = [iters]
            self.plot_data['t_comp'] = [t_comp]
            self.plot_data['t_data'] = [t_data]
        else:
            self.plot_data['epoch'] +=  [epoch]
            self.plot_data['iters'] += [iters]
            self.plot_data['t_comp']+= [t_comp]
            self.plot_data['t_data']+= [t_data]

        # Adding losses to the data record
        for k, v in losses.items():
            if (self.plot_data.get(k) is None):
                self.plot_data[k] = [v]
            else:
                self.plot_data[k] += [v]

        self.print_current_losses(epoch, iters, losses, t_comp, t_data)

    def export_losses_to_csv(self):
        """ 
        The losses dictionary is exported with csv format
        """
        df = pd.DataFrame(self.plot_data)
        df.to_csv(self.df_name, index= False)

    def plot_current_losses(self):
        """exports a plot of the current losses

        """
        if not hasattr(self, 'plot_data'):
            logging.warning('No available data to plot.')

        plot_data_filt = dict((k, v) for k,v in self.plot_data.items() if k not in self.ignore_keys)
        df = pd.DataFrame(plot_data_filt)
        df = df.groupby('epoch').mean()
        try:
            size = len(plot_data_filt.keys())
            cols = math.floor(math.sqrt(size))
            rows = math.ceil(size / cols)
            
            fig, axs = plt.subplots(rows,cols, figsize=(15, 7))
            fig.subplots_adjust(hspace = .7, wspace=.3)
            axs = axs.ravel()
            i = 0
            for col in df:
                axs[i].plot(df.index.values, df[col].values)
                axs[i].set_title(col)
                axs[i].set_xlabel('epoch')
                axs[i].set_ylabel('Loss')
                i+=1
            fig.suptitle(self.name.replace('_', ' '))
            fig.savefig(self.plot_name)
            plt.cla()
            plt.close(fig)
            
        except Exception as e:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('{}: Exception while ploting the loss values. {}\n'.format(now, e))

    def update_loss_data(self, epoch, train_losses, val_data=None):
        """updates the losses records with a new entry.

        Parameters:
            epoch (int) -- current epoch
        """
        if self.plot_data is None:
            self.plot_data = {}

        # Adding losses to the data record
        if (len(self.plot_data.keys()) == 0):
            self.plot_data['epoch'] = [epoch]
        else:
            self.plot_data['epoch'] +=  [epoch]

        # Adding losses to the data record
        for k, v in train_losses.items():
            if (self.train_data.get(k) is None):
                self.train_data[k] = [v]
            else:
                self.train_data[k] += [v]

        if val_data is not None:
            for k, v in val_data.items():
                if (self.val_data.get(k) is None):
                    self.val_data[k] = [v]
                else:
                    self.val_data[k] += [v]
        #self.print_current_losses(epoch, iters, losses, t_comp, t_data)  

    def plot_losses(self):
        """
        Plots the training losses and optionally validation losses
        """
        if not hasattr(self, 'plot_data'):
            logging.warning('No available data to plot.')

        train_data_filt = dict((k, v) for k,v in self.train_data.items() if k not in self.ignore_keys)
        df_train = pd.DataFrame(train_data_filt)
        if self.val_data:
            val_data_filt = dict((k, v) for k,v in self.val_data.items() if k not in self.ignore_keys)
            df_val = pd.DataFrame(val_data_filt)
        try:
            size = len(train_data_filt.keys())
            size = len(df_train.columns)
            col_size = math.floor(math.sqrt(size))
            row_size = math.ceil(size / col_size)
            fig = plt.figure(figsize=(15, 10))
            axs = [None]*col_size*row_size

            col_names = df_train.columns
            i = 0
            for _ in range(0, col_size):
                for _ in range(0, row_size):
                    if i >= len(df_train.columns):
                        break
                    axs[i] = fig.add_subplot(int('{}{}{}'.format(row_size, col_size, i+1)))
                    col = col_names[i]
                    axs[i].plot(df_train.index.values, df_train[col].values, label='training')
                    if self.val_data:
                        axs[i].plot(df_val.index.values, df_val[col].values, label='validation')
                    axs[i].set_title(col)
                    axs[i].set_xlabel('epoch')
                    axs[i].set_ylabel('Loss')
                    axs[i].legend()
                    i += 1
                    
            fig.subplots_adjust(hspace = .7, wspace=.3)
            fig.suptitle(self.name.replace('_', ' '))
            fig.savefig(self.plot_name)
            plt.cla()
            plt.close(fig)
            
        except Exception as e:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('{}: Exception while ploting the loss values. {}\n'.format(now, e))
    
    def parse_loss(self, losses):
        message = ''
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        return message

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, verbose=False):
        """print current losses on console; also saves the values into a csv file

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        message += self.parse_loss(losses)
        if verbose:
            print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message