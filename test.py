import argparse
import better_exceptions
from pathlib import Path
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg
from train import validate


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, help="Data root directory",default="D:/Python_jupyter_file/MAP583/Project-age-estimation-pytorch/appa-real-release")
    parser.add_argument("--resume", type=str, help="Model weight to be tested",default= "D:/Python_jupyter_file/MAP583/Project-age-estimation-pytorch/checkpoint/epoch002_0.37249_11.2144.pth")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    test_dataset = FaceDataset(args.data_dir, "test", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, drop_last=False)

    print("=> start testing")
    _, _, test_mae = validate(test_loader, model, None, 0, device)
    print(f"test mae: {test_mae:.3f}")


if __name__ == '__main__':
    main()
