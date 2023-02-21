# experiment 0 of our toy problem
import torch

import time
from tensorboardX import SummaryWriter
import sys
import os
sys.path.insert(0, os.path.join("..", "..", "exeu"))

from utils.validation import validate
from dataset.torch_dataset import TorchDataset
from model.modded_base_flow import FlowM
from model.modded_nflows_init import (
    PiecewiseRationalQuadraticCouplingTransformM,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransformM,
    )
from utils.train_funcs import train, load_model
from utils.args_train import get_args

import zuko


def trainer(tr_dataset, te_dataset, val_func):

    args = get_args()
    print(args)
    args.log_name = "ex3"
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    writer = SummaryWriter(logdir=log_dir)
    # save hparams to tensorboard
    writer.add_hparams(vars(args), {})

    # Neural spline flow (NSF) with 3 sample features and 5 context features
    model = zuko.flows.CNF(args.x_dim, args.y_dim, transforms=10, hidden_features=[128] * 3)

    if args.device == "cuda":  # Single process, single GPU per process
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
            print("!!  USING GPU  !!")

    else:
        print("!!  USING CPU  !!")

    # resume checkpoints
    res_epoch = 0
    lr = 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )


    # tr_dataset = TorchDataset(csv_file='../dataset/data.csv', stop=75000)
    # te_dataset = TorchDataset(csv_file='../dataset/data.csv', start=75000)

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
        sampler=None,
        drop_last=True,
        # worker_init_fn=init_np_seed,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=1000,  # manually set batch size to avoid diff shapes
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # print total params number and stuff
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    print(len(train_loader.dataset))

    for epoch in range(res_epoch, args.epochs):
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            loss = -flow(y).log_prob(x)  # -log p(x | y)
            loss = loss.mean()
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: %d, Train loss: %.3f" % (epoch, train_loss / len(train_loader)))


if __name__ == "__main__":
    tr_dataset = TorchDataset(csv_file='../dataset/data.csv', stop=75000)
    te_dataset = TorchDataset(csv_file='../dataset/data.csv', start=75000)
    trainer(tr_dataset, te_dataset, validate)