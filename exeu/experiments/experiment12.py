# experiment 0 of our toy problem
# params 770k
import torch
from torch.nn import functional as F
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

from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows import transforms, utils
import nflows.nn.nets as nn_


def create_linear_transform(param_dim):
    """Create the composite linear transform PLU.
    Arguments:
        input_dim {int} -- dimension of the space
    Returns:
        Transform -- nde.Transform object
    """

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


def trainer(tr_dataset, te_dataset, val_func):

    args = get_args()
    print(args)
    args.log_name = "ex11"
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

    # define model
    base_dist = StandardNormal(shape=[args.x_dim])

    num_layers = 30
    transforms = []
    for i in range(10):

        transforms.append(
            PiecewiseRationalQuadraticCouplingTransformM(
                mask=utils.create_alternating_binary_mask(
                    args.x_dim, even=(i % 2 == 0)
                ),
                transform_net_create_fn=(
                    lambda in_features, out_features: nn_.ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        hidden_features=64,
                        context_features=args.y_dim,
                        num_blocks=5,
                        activation=F.relu,
                        dropout_probability=0.15,
                        use_batch_norm=False,
                    )
                ),
                num_bins=8,
                tails="linear",
                tail_bound=3,
                apply_unconditional_transform=False,
            )
        )
        transforms.append(create_linear_transform(param_dim=args.x_dim))

    transform = CompositeTransform(transforms)

    model = FlowM(transform, base_dist)

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

    if args.resume_checkpoint is None and os.path.exists(
        os.path.join(save_dir, "checkpoint-latest.pt")
    ):
        args.resume_checkpoint = os.path.join(
            save_dir, "checkpoint-latest.pt"
        )  # use the latest checkpoint
    if args.resume_checkpoint is not None and args.resume == True:
        model, _, _, res_epoch, _, _ = load_model(
            model,
            device,
            model_dir=save_dir,
            filename="checkpoint-latest.pt",
        )
        print(f"Resumed from: {res_epoch}")

    # tr_dataset = TorchDataset(csv_file='../dataset/data.csv', stop=75000)
    # te_dataset = TorchDataset(csv_file='../dataset/data.csv', start=75000)

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=0,
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

    trh, tsh = train(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        device=torch.device(args.device),
        name="model",
        model_dir=save_dir,
        args=args,
        writer=writer,
        output_freq=100,
        save_freq=args.save_freq,
        res_epoch=res_epoch,
        val_func=validate,
    )


if __name__ == "__main__":
    tr_dataset = TorchDataset(csv_file="../dataset/data.csv", stop=950000)
    te_dataset = TorchDataset(csv_file="../dataset/data.csv", start=950000)
    trainer(tr_dataset, te_dataset, validate)
