import argparse
from collections import defaultdict
import csv
from distutils.util import strtobool
import logging
from pydoc import locate

def get_parser():
    """Return a nicely formatted argument parser

    This function is a simple wrapper for the argument parser I like to use,
    which has a stupidly long argument that I always forget.
    """
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def load_hyperparams(filepath):
    params = dict()
    with open(filepath, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for name, value, dtype in reader:
            if dtype == 'bool':
                params[name] = bool(strtobool(value))
            else:
                params[name] = locate(dtype)(value)
    return params

def save_hyperparams(filepath, params):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                bool: 'bool',
                int: 'int',
                str: 'str',
                float: 'float',
            })[type(value)] # yapf: disable
            if type_str is not None:
                writer.writerow((name, value, type_str))

def update_param(params, name, value):
    """
    Set params[name] = value if name is a valid hyperparam, otherwise raise KeyError
    """
    if name not in params:
        raise KeyError("Parameter '{}' specified, but not found in hyperparams file.".format(name))
    else:
        logging.info("Updating parameter '{}' to {}".format(name, value))
    if type(params[name]) == bool:
        params[name] = bool(strtobool(value))
    else:
        params[name] = type(params[name])(value)

def load_hyperparams_and_inject_args(args):
    params = load_hyperparams(args.hyperparams)
    for arg_name, arg_value in vars(args).items():
        if arg_name in ['hyperparams', 'other_args']:
            continue
        params[arg_name] = arg_value
    for arg_name, arg_value in args.other_args:
        update_param(params, arg_name, arg_value)
    del args.other_args
    return params

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def parse_args_and_load_hyperparams(parser):
    """
    1. Parse known args registered with the argparse parser
    2. Keep track of unknown args (e.g. '--some_hyperparameter')
    2. Load default hyperparameters from args.hyperparams file
    3. For known args, add them directly to the list of hyperparameters
    4. For unknown args, check if they match a valid hyperparameter, and update the value
    5. Return a namespace so we can access values easily (e.g. `args.some_hyperparameter`)
    """
    args, unknown = parser.parse_known_args()
    other_args = {(remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}
    args.other_args = other_args
    params = load_hyperparams_and_inject_args(args)
    args = argparse.Namespace(**params)
    return args
