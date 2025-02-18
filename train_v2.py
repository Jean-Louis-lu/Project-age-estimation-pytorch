import argparse
import better_exceptions
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset_v2 import FaceDataset, NewFaceDataset
from defaults import _C as cfg
import ssl
import urllib.request
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="D:/Python_jupyter_file/MAP583/Project-age-estimation-pytorch/appa-real-release", help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default="tf_log", help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_dataset_X,train_dataset_U, model, criterion, criterion_U,optimizer, epoch, device,tau,ratio):
    print("training")
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    train_loader_U = DataLoader(train_dataset_U, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)

    with tqdm(train_loader_U) as _tqdm:
        for x, y in _tqdm:
            
            mu_B = len(x)
            B = int(mu_B/ratio)

            train_loader_X = DataLoader(train_dataset_X,batch_size=B,shuffle=True)
            ls = 0
            
            for x_X, y_X in train_loader_X:
                x_X = x_X.to(device)
                y_X = y_X.type(torch.LongTensor)
                y_X = y_X.to(device)

                # compute output
                outputs = model(x_X)

                # calc loss
                ls = criterion(outputs, y_X).item()

            
            x = x.to(device)
            y = y.to(device)

            # compute output

            outputs_weak = model(x)
            outputs_strong = model(y)
        
            # calc loss
            lu_temp = criterion_U(outputs_strong,torch.argmax(outputs_weak,dim=1))
            

            lu = torch.mean(lu_temp * (torch.max(outputs_weak,1).values>tau),dim=0)

            #lbd_u to be added
            loss = ls+0*lu
            cur_loss = loss.item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, sample_num=sample_num)


    return loss_monitor.avg, accuracy_monitor.avg



def validate(validate_loader, model, criterion, epoch, device):
    print("validation")
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.type(torch.LongTensor)
                y = y.to(device)

                # compute output
                outputs = model(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH)
    
    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
 
    """
    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path is None:
        resume_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")
    
    if not resume_path.is_file():
        print(f"=> model path is not set; start downloading trained model to {resume_path}")
        url = "https://github.com/yu4u/age-estimation-pytorch/releases/download/v1.0/epoch044_0.02343_3.9984.pth"
        urllib.request.urlretrieve(url, str(resume_path))
        print("=> download finished")
        
    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
    """
    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_U = nn.CrossEntropyLoss(reduction='none').to(device)

    train_dataset_X = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    
    train_dataset_U = NewFaceDataset(args.data_dir, "new_train", img_size=cfg.MODEL.IMG_SIZE,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    

    """
    #show some augmented images
    
    print("here")
    for i in range(5):
        weak = train_dataset_U[i][0]
        strong = train_dataset_U[i][1]
        
        weak = weak/weak.max()
        
        w = torch.zeros([3,224,224])
        w[0] = weak[2]
        w[1] = weak[1]
        w[2] = weak[0]
        weak = w
        

        image_weak = to_pil_image(weak)
        image_weak.save('weak_{}.png'.format(i))

        strong = strong/strong.max()
        
        s = torch.zeros([3,224,224])
        s[0] = strong[2]
        s[1] = strong[1]
        s[2] = strong[0]
        strong = s

        image_strong = to_pil_image(strong)
        image_strong.save('strong_{}.png'.format(i))
    """ 
        
        

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                             drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0 
    train_writer = None 

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_val")
    
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(train_dataset_X, train_dataset_U,model, criterion, criterion_U,optimizer, epoch, device,0.8,2)

        # validate
        val_loss, val_acc, val_mae = validate(val_loader, model, criterion, epoch, device)
        
        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
