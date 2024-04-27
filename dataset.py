import os
import pandas as pd
from torch.utils.data import Dataset
import glob
import torchvision
from torchvision.transforms import v2
from PIL import Image
import json

class ImageNette(Dataset):
    def __init__(self, dataset_name = "imagenette2", base_dir=None):
        self.base_dir =  f"/home/mprabhud/phd_projects/diff-det/diff-det/data/{dataset_name}/val"
        self.classes = os.listdir(self.base_dir)
        self.image_paths,self.labels = [], []
        self.process_label_map()

        for i,cls in enumerate(self.classes):
            for path in glob.glob(os.path.join(self.base_dir,cls+"/*")):
                self.image_paths.append(path)
                self.labels.append(self.idx2label[cls])
        self.transforms = torchvision.transforms.Compose([
                v2.ToTensor(),
                v2.Resize((224,224)),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    
    def process_label_map(self,path="/home/mprabhud/phd_projects/diff-det/diff-det/data/imagenet_class_index.json"):
        self.class_idx = json.load(open(path,'r'))
        self.idx2label = {}
        self.label2name = {}
        for k in range(len(self.class_idx)):
            self.idx2label[self.class_idx[str(k)][0]] = k
            self.label2name[k] = self.class_idx[str(k)][1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label