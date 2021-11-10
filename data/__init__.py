from data import dataloaders

def create_dataset(opt, is_train= True):
    """Create a dataset.
    """
    dataset_name = opt['dataset']
    # datasetlib = importlib.import_module(dataset_name)
    for name, cls in dataloaders.__dict__.items():
        if name.lower() == dataset_name.lower() \
           and issubclass(cls, dataloaders.DataloaderBase):
            dataset = cls
    instance = dataset(opt, is_train)
    return instance