from cProfile import label
from typing import Any, Callable, List
from nbformat import write
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.nn.functional import softmax
import PIL
import pandas as pd
import os
import shutil
import pickle

from tqdm import tqdm

from src.logger import logger
from config import CLASSES, DATA_DIR, GAME_STATUS_MAP, LABELLED_DATA_ROOT_DIR

def get_label_from_fname(fname):
    pass

def get_model(model_class, args):
    if model_class.upper() == 'BINARYGAMESTATECLASSIFIER':
        return BinaryGameStateClassifier(**args)
    
    elif model_class.upper() == 'GAMESTATECLASSIFIER':
        raise NotImplementedError("Have yet to adapt to > 2 outputs.")

class BinaryGameStateClassifier(pl.LightningModule):

    def __init__(
        self,
        criterion:Any=nn.BCELoss(),
        model_size:str="tiny"
    ) -> None:
        super(BinaryGameStateClassifier, self).__init__()

        self.num_classes = 2
        self.criterion = criterion

        # init the deit pretrained
        self.model =  torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True) if model_size == 'tiny' else torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        # 768 specified in paper// 192 for tiny
        self.model.head = nn.Linear(192, self.num_classes) if model_size == 'tiny' else nn.Linear(768, self.num_classes) 

    def forward(self, x):
        return softmax(self.model(x))

    ## Steps
    def training_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.forward(x)
        loss = self.criterion(logits, labels)
        # _, batch_preds = torch.max(logits, 1)

        # acc = torch.sum(labels == batch_preds).float() / batch_preds.shape[0]

        # self.log('train_loss', loss, prog_bar=True)
        # self.log('train_accuracy', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits = self.forward(x)
        loss = self.criterion(logits, labels)
        # _, batch_preds = torch.max(logits, 1)
        # acc = torch.sum(labels == batch_preds).float() / batch_preds.shape[0]

        # self.log('val_loss', loss, on_step=True)
        # self.log('val_accuracy', acc, on_step=True)
     
        return loss

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        logits = self.forward(x)
        _, batch_preds = torch.max(logits, 1)
        acc = torch.sum(y == batch_preds).float() / batch_preds.shape[0]
        return {'test_loss': F.cross_entropy(logits, y), 'test_accuracy': acc}

    def configure_optimizers(self):
        # AdamW was used in the paper
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class GameStateDataset():

    def __init__(
        self,
        dataset_type:str,
        transform:Callable,
    ) -> None:
        self.dataset_type = dataset_type.lower()
        assert dataset_type in ['train','test','val']

        self.img_dir = os.path.join(LABELLED_DATA_ROOT_DIR, self.dataset_type)
        self.images = os.listdir(self.img_dir)
        self.labels = self._load_labels()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        fname = self.images[idx]
        image = PIL.Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        image = self.transform(image)
        label = torch.Tensor(self.labels[fname])
        return image, label

    def _load_labels(self):
        with open(os.path.join(LABELLED_DATA_ROOT_DIR, "labels.pkl"), 'rb') as f:
            labels = pickle.load(f)

        return labels

    
class GameStateDataModule(pl.LightningDataModule):

    '''
        data_dir: root for where data is being moved from path/to/{game state 1, game state 2,..}
        classes: list of game states
        input_file_ext: file ext for the images
        batch_size: int
        build_dataset_splits: whether to build train/test/val or not
        train_size: amount to cut off from all data for train
        val_size: amount to cut off from remaining data for val (remainder test)
    '''

    def __init__(
        self,
        data_dir:str,
        classes:dict,
        input_file_ext:str,
        batch_size:int,
        build_dataset_splits:bool=False,
        train_size:float=0.7,
        val_size:float=0.7):  
    
        super().__init__()

        self.data_dir = data_dir
        self.classes = classes
        self.input_file_ext = input_file_ext
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.build_dataset_splits = build_dataset_splits
        
    def _store_labels(self, label_dict):
        with open(os.path.join(self.data_dir, "labels.pkl"), 'wb') as f:
            pickle.dump(label_dict, f)

    def _create_data_folders(self):
        all_files = []
        labels = {}

        def create_label(idx_pos):
            l = [0.0] * len(CLASSES)
            l[idx_pos] = 1.0
            return l


        logger.info("Creating dataset from class folders.")
        for cls in self.classes:
            cls_path = os.path.join(self.data_dir, cls)
            if os.path.exists(cls_path):
                files = [
                    os.path.join(cls_path, f) 
                    for f in os.listdir(cls_path)
                ]
                filtered = list(filter(lambda f: f.endswith(self.input_file_ext), files))
                for f in filtered:
                    # Map game state -> Class str -> Label int
                    l = create_label(CLASSES[GAME_STATUS_MAP[cls]])
                    labels.update({f.split("/")[-1]: l})

                all_files.extend(filtered)
            
        # Create a label lookup
        self._store_labels(label_dict=labels)

        train_idx = int(len(all_files) * self.train_size)
        val_idx = int((len(all_files) - train_idx) * self.val_size)

        train_files = all_files[:train_idx]
        val_files = all_files[train_idx:train_idx+val_idx]
        test_files = all_files[train_idx+val_idx:]
        # For easier iterating
        file_sets = [train_files, val_files, test_files]

        logger.info(f"Train: {len(train_files)} - val: {len(val_files)} - Test: {len(test_files)}")

        tr_path = os.path.join(self.data_dir, "train")
        val_path = os.path.join(self.data_dir, "val")
        test_path = os.path.join(self.data_dir, "test")
        # For easier iterating
        paths = [tr_path, val_path, test_path]

        for p, files in zip(paths, file_sets):
            if os.path.exists(p):
                shutil.rmtree(p)
            os.mkdir(p)

            for f in tqdm(files, desc=f"Copying files to {p}"):
                # Copy from the class label folder to the 
                # train/test/val folder, keep the file name
                shutil.copyfile(f, os.path.join(p, f.split("/")[-1] ) )
        

    def setup(self):
        if self.build_dataset_splits :
            self._create_data_folders()

        img_transform = transforms.Compose(
            [
                # transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Resize((224, 224)) 
            ]
        )
        self.train_ds = GameStateDataset('train', img_transform)
        self.val_ds = GameStateDataset('val', img_transform)
        self.test_ds = GameStateDataset('test', img_transform)
        

    def train_dataloader(self) -> DataLoader:
        # pyre-fixme[6]
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        # pyre-fixme[6]:
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        # pyre-fixme[6]
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    # def teardown(self, stage: Optional[str] = None) -> None:
    #     for ds in ['train','test','val']:
    #         p = os.path.join(DATA_DIR, ds)
    #         if os.path.exists(p):
    #             os.removedirs(p)