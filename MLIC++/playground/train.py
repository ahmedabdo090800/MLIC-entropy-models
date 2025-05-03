import os
import random
import logging
import math
from PIL import ImageFile, Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.logger import setup_logger
from utils.utils import CustomDataParallel, save_checkpoint
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch
from utils.testing import test_one_epoch
from loss.rd_loss import RateDistortionLoss
from config.args import train_options
from config.config import model_config
from models import *

def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        seed = args.seed
    else:
        seed = int(100 * random.random())
    torch.manual_seed(seed)
    random.seed(seed)

    experiment_dir = os.path.join('./experiments', args.experiment)
    os.makedirs(experiment_dir, exist_ok=True)

    setup_logger('train', experiment_dir, 'train_' + args.experiment, level=logging.INFO, screen=True, tofile=True)
    setup_logger('val', experiment_dir, 'val_' + args.experiment, level=logging.INFO, screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir=os.path.join('./tb_logger', args.experiment))

    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # ==== Dataset Loading ====
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root='data', transform=transform)  # Make sure images are inside: data/class_x/*.jpg

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    # ==== Model Setup ====
    net = MLICPlusPlus(config=config)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)

    start_epoch = 0
    best_loss = 1e10
    current_step = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_loader.dataset) / args.batch_size)

    logger_train.info(args)
    logger_train.info(config)
    logger_train.info(net)
    logger_train.info(optimizer)
    optimizer.param_groups[0]['lr'] = args.learning_rate

    # ==== Training Loop ====
    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(
            net, criterion, train_loader, optimizer, aux_optimizer,
            epoch, args.clip_max_norm, logger_train, tb_logger, current_step
        )

        save_dir = os.path.join(experiment_dir, 'val_images', '%03d' % (epoch + 1))
        loss = test_one_epoch(epoch, test_loader, net, criterion, save_dir, logger_val, tb_logger)

        lr_scheduler.step()
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        net.update(force=True)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                os.path.join(checkpoints_dir, "checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_val.info('Best checkpoint saved.')

if __name__ == '__main__':
    main()
