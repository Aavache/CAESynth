"""This package contains modules related to objective functions, optimizations, and network architectures.

The methods listed implemented in this script, instanciate a model by model_name argument
"""

import importlib
from lib.models.base_model import BaseModel

def find_model_using_name(model_name):
    """Import the module "lib/[model_name]_model.py".

    In the file, the class called [model_name]Model() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "lib." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def create_model(opt, is_train= True):
    """Create a model given the option.
    """
    model = find_model_using_name(opt['model']['name'])
    instance = model(opt, is_train)
    print("model [%s] was created" % type(instance).__name__)
    return instance