import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet152
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
from skimage import io
from skimage.measure import regionprops
from skimage.transform import resize
from argparse import ArgumentParser
import pywt
import glob


class CellDataset(Dataset):
    def __init__(self, imgDir, transform):
        self.imgDir = imgDir  # imgDir/Normal and imgDir/Tumor are used for training

        self.imgNormalNames = os.listdir(os.path.join(imgDir, 'Normal'))
        self.imgNormalNum = len(self.imgNormalNames)
        self.imgTumorNames = os.listdir(os.path.join(imgDir, 'Tumor'))
        self.imgTumorNum = len(self.imgTumorNames)
        self.imgMarginNNames = os.listdir(os.path.join(imgDir, 'NeckMargin/Negative'))
        self.imgMarginNNum = len(self.imgMarginNNames)
        self.imgMarginPNames = os.listdir(os.path.join(imgDir, 'NeckMargin/Positive'))
        self.imgMarginPNum = len(self.imgMarginPNames)

        # self.imgSelectionNames = os.listdir(os.path.join(imgDir, 'Selection'))
        # self.imgSelectionNum = len(self.imgSelectionNames)

        self.transform = transform

    def __len__(self):
        # return self.imgSelectionNum
        return self.imgNormalNum + self.imgTumorNum + self.imgMarginNNum + self.imgMarginPNum

    def __getitem__(self, item):
        if item < self.imgNormalNum:
            img = io.imread(os.path.join(self.imgDir, 'Normal', self.imgNormalNames[item]))
        else:
            item -= self.imgNormalNum
            if item < self.imgTumorNum:
                img = io.imread(os.path.join(self.imgDir, 'Tumor', self.imgTumorNames[item]))
            else:
                item -= self.imgTumorNum
                if item < self.imgMarginNNum:
                    img = io.imread(os.path.join(self.imgDir, 'NeckMargin/Negative', self.imgMarginNNames[item]))
                else:
                    item -= self.imgMarginNNum
                    img = io.imread(os.path.join(self.imgDir, 'NeckMargin/Positive', self.imgMarginPNames[item]))

        # img = io.imread(os.path.join(self.imgDir, 'Selection', self.imgSelectionNames[item]))

        # img = img_resize(img)
        # img = add_wavelet(img)

        return [self.transform(img) for i in range(2)]  # range(n_views)


class SimCLR(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = resnet18(
            weights=None, num_classes=4 * hidden_dim)  # num_classes is the output size of the last linear layer

        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.convnet(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs = torch.cat(batch, dim=0)

        # Encode all images
        feats = self.forward(imgs)
        # Calculate cosine similarity; None to add new dim and create n×n similarity(n=n_views*batch_size)
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)  # self similarity
        # Find positive example -> batch_size//2 away from the original example
        # pos_mask == True -> a positive pair
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")


def train_simclr(data, batch_size, args, max_epochs=500, **kwargs):
    CHECKPOINT_PATH = './checkpoint'
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, args.checkpoint_dir),
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=False, mode="min", monitor="train_loss",
                            filename='{epoch}-{train_loss:.5f}', verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    ckpt_path = glob.glob("./checkpoint/" + args.checkpoint_dir + "/lightning_logs/version_*/checkpoints/*.ckpt")
    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    pl.seed_everything(42)  # To be reproducable
    model = SimCLR(max_epochs=max_epochs, **kwargs)
    if len(ckpt_path) > 0:
        print(f"Found pretrained model at {ckpt_path[0]}, continue training...")
        # Automatically loads the model with the saved hyperparameters
        trainer.fit(model, train_loader, ckpt_path=ckpt_path[0])
    else:
        trainer.fit(model, train_loader)
        # Load best checkpoint after training
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


def img_resize(img, out_size=100, spec=False):
    # Crop the bounding box of cell out and fill it into a 100x100 square
    # if spec=True, put the 100x100 square into a background image with the same size as img (noMorphology v2)
    mask = np.uint8(np.sum(img, axis=2) != 0)
    # mask = np.uint8(img != 0)
    # if len(mask.shape) > 2:
    #     mask = mask[:, :, 1]
    prop = regionprops(mask)
    bbox = prop[0].bbox  # (min_row, min_col, max_row+1, max_col+1)
    width = bbox[3] - bbox[1]
    height = bbox[2] - bbox[0]
    square_size = np.max((width, height))

    out = np.zeros((square_size, square_size, 3), dtype=np.float32)
    if width > height:
        out[round((square_size - height) / 2):round((square_size - height) / 2) + height, :, :] = \
            img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    else:
        out[:, round((square_size - width) / 2):round((square_size - width) / 2) + width, :] = \
            img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    out = resize(out, (out_size, out_size, 3))

    if spec:
        imgsize = img.shape[0]
        out1 = np.zeros((imgsize, imgsize, 3), dtype=np.float32)
        out1[round((imgsize - out_size) / 2):round((imgsize - out_size) / 2) + out_size,
             round((imgsize - out_size) / 2):round((imgsize - out_size) / 2) + out_size, :] = out
        return out1
    return out


def add_wavelet(img):
    [cA, (cH, cV, cD)] = pywt.dwt2(img[:, :, 1], 'db1')
    img[:, :, 0] = resize(cH, (img.shape[0], img.shape[1]))
    return img


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default="test")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    arguments = parser.parse_args()

    aug_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAutocontrast(),
        ]
    )

    dataset = CellDataset('./dataset/segment/2c_cellManualImg', aug_transforms)
    simclr_model = train_simclr(data=dataset, batch_size=8, hidden_dim=128, lr=1e-3,
                                temperature=0.07, weight_decay=1e-4, max_epochs=200, args=arguments,
                                )
