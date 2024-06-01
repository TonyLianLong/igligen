import argparse
import logging
import sys
import os
import yaml

class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)

logger = logging.getLogger("main")
def set_loglevel(debug):
    loglevel = logging.DEBUG if debug else logging.WARN
    logger.setLevel(loglevel)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(loglevel)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger.addHandler(handler)
    logger.propagate = False

def get_logger():
    return logger

def merge_cli_opt(config, key, value):
    key_hierarchy = key.split(".")
    item_container = config
    for hierarchy in key_hierarchy[:-1]:
        item_container = item_container[hierarchy]
    
    original_value = item_container[key_hierarchy[-1]]
    
    if isinstance(original_value, bool):
        # bool is a type of int so should be processed first
        if value == "True" or value == "true" or value == "1":
            value = True
        elif value == "False" or value == "false" or value == "0":
            value = False
        else:
            raise ValueError(f"Unknown bool value for {key}: {value} (original value: {original_value})")
    elif isinstance(original_value, int):
        value = int(value)
    elif isinstance(original_value, float):
        value = float(value)
    
    assert type(original_value) == type(value), f"{type(original_value)} != {type(value)}"
    
    logger.info(f"Overriding {key} with {value} (original value: {original_value})")
    item_container[key_hierarchy[-1]] = value

def merge_cli_opts(config, cli_opts):
    assert len(cli_opts) % 2 == 0, f"{len(cli_opts)} should be even"
    for key, value in zip(cli_opts[::2], cli_opts[1::2]):
        merge_cli_opt(config, key, value)


def dump_args(args, filename="config.yaml"):
    assert not os.path.exists(
        filename), f"Do not dump to existing file: {filename}"
    with open(filename, "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

# Does not support base_config
def load_args_simple(filename):
    with open(filename, "r") as f:
        config = yaml.load(f, Loader=UniqueKeyLoader)
    args = argparse.Namespace(**config)

    return args

# https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
def merge_dict(a, b, path=None, allow_replace=False):
    """Merges b into a"""

    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)], allow_replace=allow_replace)
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                if allow_replace:
                    logger.info(f"Replacing key at {'.'.join(path + [str(key)])} to {b[key]}")
                    a[key] = b[key]
                else:
                    raise ValueError(f"Conflict at {'.'.join(path + [str(key)])}")
        else:
            a[key] = b[key]
    return a

def load_config(config_path):
    """Load yaml into dictionary.


    Only merges dictionary. Lists will be replaced.

    Args:
        config_path (str): yaml path

    Returns:
        config (dict): config in dictionary
    """

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=UniqueKeyLoader)

    if "base_config" not in config:
        return config

    base_path = os.path.join(os.path.dirname(config_path), config["base_config"])
    overwrite_path = config_path

    logger.info(f"Loading base config {base_path} and overwrite config {overwrite_path}")

    base_config = load_config(base_path)

    merged_config = merge_dict(base_config, config, allow_replace=True)

    return merged_config

def load_args(config_path, cli_opts):
    """Load yaml into args

    Only merges dictionary. Lists will be replaced.

    Args:
        config_path (str): yaml path

    Returns:
        argparse.Namespace: args
    """

    config = load_config(config_path)
    merge_cli_opts(config, cli_opts)
    args = argparse.Namespace(**config)
    return args
