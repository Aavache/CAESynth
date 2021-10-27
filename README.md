# CAESynth: Real-Time Timbre Intepolation and Pitch Control with Conditional Autoencoders

This is the original pytorch implementation of the CAESynth paper, presented at the IEEE International Workshop on Machine Learning for Signal Processing 2021. Cite our work with the following BibTex. 

## Dependencies


## Datasets
Both [NSynth](https://magenta.tensorflow.org/datasets/nsynth#files) and [FreeSoundDataset50k](https://zenodo.org/record/4060432#.YXjuK3UzZhE) can be downloaded at the provided links. The datasets should be stored in the `./data/` directory.

## Training
In this implementation we opt for gathering the training configuration with external configuration file in json format located at the `./option/` directory. Customize your own configuration file following similar structure to examples. Once the configuration file is ready, run the training script. 

```
python train.py --opt_file "./options/config_file_name.json"
```

## Inference
We develop the inference code examples in jupyter notebook so that it is more interactive and intuitive. Synthesized exampled can be found the `./samples/` directory.