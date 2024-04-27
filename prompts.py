from importlib import resources
import os
import functools
import random
import inflect
from dataset import ImageNette

dataset = ImageNette()

IE = inflect.engine()
ASSETS_PATH = resources.files("assets")

classes = {
            "imagenette2" : [
                "n01440764", 
                "n02102040",
                "n02979186", 
                "n03000684", 
                "n03028079", 
                "n03394916", 
                "n03417042", 
                "n03425413", 
                "n03445777", 
                "n03888257"
            ],
        "imagewoof2" : [
            "n02096294" ,
            "n02093754",
            "n02111889",
            "n02088364" , 
            "n02086240" , 
            "n02089973",
            "n02087394", 
            "n02115641", 
            "n02099601" , 
            "n02105641"
        ],
        "imagenet" : list(dataset.idx2label.keys())
} 

def get_prompt(class_name):
    return f"an image of {class_name}"

def get_eval_prompts(dataset_name):
    eval_prompts, eval_labels = [], [] 
    class_idx = classes[dataset_name]
    for cls_idx in class_idx:
        eval_prompts.append(get_prompt(dataset.label2name[dataset.idx2label[cls_idx]])) 
        eval_labels.append(dataset.idx2label[cls_idx])
    return eval_prompts,eval_labels


def class_prompts(dataset_name):
    class_idx = classes[dataset_name]
    label = random.choice(list(range(len(class_idx))))
    return get_prompt(dataset.label2name[label]), label

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
