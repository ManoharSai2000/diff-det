from importlib import resources
import os
import functools
import random
import inflect
from dataset import ImageNette

dataset = ImageNette()

IE = inflect.engine()
ASSETS_PATH = resources.files("assets")

classes = {"tench" : "n01440764", 
           "English springer" : "n02102040",
           "cassette player" : "n02979186", 
           "chain saw" : "n03000684", 
           "church" : "n03028079", 
           "French horn" : "n03394916", 
           "garbage truck" : "n03417042", 
           "gas pump" : "n03425413", 
           "golf ball" : "n03445777", 
           "parachute" : "n03888257"
}
class_names = list(classes.keys()) 

def get_prompt(cls):
    return f"an image of {cls}"

def class_prompts():
    cls = random.choice(list(range(len(class_names))))
    return get_prompt(class_names[cls]), dataset.idx2label[classes[class_names[cls]]]

@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def hps_v2_all():
    return from_file("hps_v2_all.txt")

def simple_animals():
    return from_file("simple_animals.txt")

def eval_simple_animals():
    return from_file("eval_simple_animals.txt")

def eval_hps_v2_all():
    return from_file("hps_v2_all_eval.txt")
