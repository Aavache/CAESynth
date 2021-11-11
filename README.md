# CAESynth: Real-Time Timbre Intepolation and Pitch Control with Conditional Autoencoders

This is the original python implementation of the [CAESynth](https://arxiv.org/abs/2111.05174) paper, presented at the IEEE International Workshop on Machine Learning for Signal Processing MLSP 2021. Please cite our work!. 

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

## Inference
[comment]: <> (We develop the inference code examples in jupyter notebook so that it is more interactive and intuitive. Synthesized exampled can be found in the `./samples/` directory.)
An inference jupyter notebook and synthesized samples will be released soon.