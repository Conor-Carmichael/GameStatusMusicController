from config import BUILD_TR_VA_TE, IMAGE_EXT, GAME_STATUS_MAP, BATCH_SIZE, LABELLED_DATA_ROOT_DIR
from src.training.model import *
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()

config_path = os.environ['PYTHONPATH'] + "/hydra/"

@hydra.main(
    config_path=config_path, 
    config_name="training/simple_model.yaml"
)
def main(cfg : DictConfig, env:dict) -> None:
    cfg = cfg.training

    data_module = GameStateDataModule(
        data_dir=cfg.data.root,
        input_file_ext=cfg.data.image_fmt,
        classes=cfg.class_actions.keys(),
        batch_size=cfg.model.batch_size,
        build_dataset_splits=True 
    )
    data_module.setup()

    model = BinaryGameStateClassifier()
    trainer = pl.Trainer()
    trainer.fit(
        model, 
        train_dataloader=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader()
    )

if __name__ == '__main__':

    main()