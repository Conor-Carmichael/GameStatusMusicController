from config import BUILD_TR_VA_TE, IMAGE_EXT, GAME_STATUS_MAP, BATCH_SIZE, LABELLED_DATA_ROOT_DIR
from src.training.model import *
import pytorch_lightning as pl


def main():
    data_module = GameStateDataModule(
        data_dir=LABELLED_DATA_ROOT_DIR,
        input_file_ext=IMAGE_EXT,
        classes=list(GAME_STATUS_MAP.keys()),
        batch_size=BATCH_SIZE,
        build_dataset_splits=BUILD_TR_VA_TE 
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