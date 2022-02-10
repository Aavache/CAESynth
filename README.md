# CAESynth: Real-Time Timbre Intepolation and Pitch Control with Conditional Autoencoders

This is the original python implementation of the [CAESynth](https://ieeexplore.ieee.org/document/9596414) paper, presented at the IEEE International Workshop on Machine Learning for Signal Processing MLSP 2021. Please cite our work!. 

```
@INPROCEEDINGS{9596414,
  author={Puche, Aaron Valero and Lee, Sukhan},
  booktitle={2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP)}, 
  title={Caesynth: Real-Time Timbre Interpolation and Pitch Control with Conditional Autoencoders}, 
  year={2021}, volume={}, number={}, pages={1-6},
  doi={10.1109/MLSP52302.2021.9596414}}
```

## Dependencies

The necessary python libraries to run our experience can be directly downloaded executing the following command:

```
pip install -r requirements.txt
```

## Datasets
Both [NSynth](https://magenta.tensorflow.org/datasets/nsynth#files) and [FreeSoundDataset50k](https://zenodo.org/record/4060432#.YXjuK3UzZhE) can be downloaded at the provided links. The datasets should be stored in the `./data/` directory.

## Training
In this implementation, we opt for summarizing the training configuration with external json files located in the `./option/` directory. Customize your own configuration file following similar structure to the already existing examples. Once the configuration file is ready, start the training with the following command. 

```
python train.py --opt_file "./options/config_file_name.json"
```