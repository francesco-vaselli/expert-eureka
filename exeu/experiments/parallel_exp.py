# experiment 0 of our toy problem
# 457k params
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.backends import (
    cudnn,
)  # for faster training, requires fixed input. should work?
import time
import numpy as np

from tensorboardX import SummaryWriter
import sys
import os
import warnings

sys.path.insert(0, os.path.join("..", "..", "exeu"))

from utils.validation import validate
from dataset.torch_dataset import TorchDataset
from model.modded_base_flow import FlowM
from model.modded_nflows_init import (
    PiecewiseRationalQuadraticCouplingTransformM,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransformM,
)
from utils.train_funcs import train, load_model, save_model
from utils.args_train import get_args

from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows import transforms


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


def trainer(gpu, save_dir, ngpus_per_node, args):
    # basic setup
    cudnn.benchmark = False  # to be tried later
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if args.log_name is not None:
        log_dir = "runs/%s" % args.log_name
    else:
        log_dir = "runs/time-%d" % time.time()

    if not args.distributed or (args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(logdir=log_dir)
        # save hparams to tensorboard
        writer.add_hparams(vars(args), {})
    else:
        writer = None

    # define model
    base_dist = StandardNormal(shape=[args.x_dim])

    num_layers = 30
    transforms = []
    for _ in range(10):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=args.x_dim,
                use_residual_blocks=False,
                num_blocks=2,
                hidden_features=64,  # was 4, 20
                context_features=args.y_dim,
            )
        )
        transforms.append(create_linear_transform(param_dim=args.x_dim))
    for _ in range(20):

        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransformM(
                features=args.x_dim,
                tails="linear",
                use_residual_blocks=False,
                hidden_features=64,  # was 4, 20
                num_blocks=2,
                tail_bound=3.0,
                num_bins=8,
                context_features=args.y_dim,
            )
        )
        transforms.append(create_linear_transform(param_dim=args.x_dim))

    transform = CompositeTransform(transforms)

    model = FlowM(transform, base_dist)
    # print total params number and stuff
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    start_epoch = 0
    if args.resume_checkpoint is None and os.path.exists(
        os.path.join(save_dir, "checkpoint-latest.pt")
    ):
        args.resume_checkpoint = os.path.join(
            save_dir, "checkpoint-latest.pt"
        )  # use the latest checkpoint
    if args.resume_checkpoint is not None and args.resume == True:
        model, _, _, start_epoch, _, _ = load_model(
            model,
            model_dir=save_dir,
            filename="checkpoint-latest.pt",
        )
        print(f"Resumed from: {start_epoch}")

    # multi-GPU setup
    if args.distributed:  # Multiple processes, single GPU per process
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            ddp_model = DDP(
                model,
                device_ids=[args.gpu],
                output_device=args.gpu,
                check_reduction=True,
            )
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = 0
            print("going parallel")
        else:
            assert (
                0
            ), "DistributedDataParallel constructor should always set the single device scope"
    elif args.gpu is not None:  # Single process, single GPU per process
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print("going single gpu")
    else:  # Single process, multiple GPUs per process
        model = model.cuda()
        ddp_model = nn.DataParallel(model)
        print("going multi gpu")

    # resume checkpoints
    lr = 1e-4
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    tr_dataset = TorchDataset(csv_file="../dataset/data.csv", stop=950000)
    te_dataset = TorchDataset(csv_file="../dataset/data.csv", start=950000)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=512,
        num_workers=5,
        pin_memory=True,
        drop_last=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        worker_init_fn=init_np_seed
        # worker_init_fn=init_np_seed,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset,
        batch_size=1000,  # manually set batch size to avoid diff shapes
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_np_seed,
    )

    print(len(train_loader.dataset))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # main training loop
    start_time = time.time()
    loss_avg_meter = AverageValueMeter()
    log_p_nats_avg_meter = AverageValueMeter()
    log_det_nats_avg_meter = AverageValueMeter()
    output_freq = 50
    train_history = []
    test_history = []

    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if writer is not None:
            writer.add_scalar("lr/optimizer", scheduler.get_lr()[0], epoch)

        # train for one epoch
        for batch_idx, (z, y) in enumerate(train_loader):
            ddp_model.train()
            optimizer.zero_grad()

            if gpu is not None:
                z = z.cuda(args.gpu, non_blocking=True)
                y = y.cuda(args.gpu, non_blocking=True)

            # Compute log prob
            log_p, log_det = ddp_model(z, context=y)
            loss = -log_p - log_det

            # Keep track of total loss.
            train_loss += (loss.detach()).sum()
            train_log_p += (-log_p.detach()).sum()
            train_log_det += (-log_det.detach()).sum()

            # loss = (w * loss).sum() / w.sum()
            loss = (loss).mean()

            loss.backward()
            optimizer.step()

            if (output_freq is not None) and (batch_idx % output_freq == 0):
                duration = time.time() - start_time
                start_time = time.time()
                print(
                    "[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Loss %2.5f"
                    % (
                        args.rank,
                        epoch,
                        batch_idx,
                        len(train_loader),
                        duration,
                        loss.item(),
                    )
                )

        train_loss = train_loss.item() / len(train_loader.dataset)
        train_log_p = train_log_p.item() / len(train_loader.dataset)
        train_log_det = train_log_det.item() / len(train_loader.dataset)
        print(
            "Model:{} Train Epoch: {} \tAverage Loss: {:.4f}, \tAverage log p: {:.4f}, \tAverage log det: {:.4f}".format(
                args.log_name, epoch, train_loss, train_log_p, train_log_det
            )
        )
        # evaluate on the validation set
        with torch.no_grad():
            ddp_model.eval()
            test_loss = 0.0
            test_log_p = 0.0
            test_log_det = 0.0

            for z, y in test_loader:

                if gpu is not None:
                    z = z.cuda(args.gpu, non_blocking=True)
                    y = y.cuda(args.gpu, non_blocking=True)

                # Compute log prob
                log_p, log_det = ddp_model(z, context=y)
                loss = -log_p - log_det

                # Keep track of total loss.
                test_loss += (loss.detach()).sum()
                test_log_p += (-log_p.detach()).sum()
                test_log_det += (-log_det.detach()).sum()

            test_loss = test_loss.item() / len(test_loader.dataset)
            test_log_p = test_log_p.item() / len(test_loader.dataset)
            test_log_det = test_log_det.item() / len(test_loader.dataset)
            # test_loss = test_loss.item() / total_weight.item()
            print(
                "Test set: Average loss: {:.4f}, \tAverage log p: {:.4f}, \tAverage log det: {:.4f}".format(
                    test_loss, test_log_p, test_log_det
                )
            )

        scheduler.step()
        train_history.append(train_loss)
        test_history.append(test_loss)
        # save checkpoints
        if not args.distributed or (args.rank % ngpus_per_node == 0):
            if (epoch + 1) % args.save_freq == 0:
                save_model(
                epoch,
                model,
                scheduler,
                train_history,
                test_history,
                name="model",
                model_dir=save_dir,
                optimizer=optimizer,
            )

def main():
    args = get_args()

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
    print("Arguments:")
    print(args)

    args.log_name = "exp"
    save_dir = os.path.join("checkpoints", args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(trainer, nprocs=ngpus_per_node, args=(save_dir, ngpus_per_node, args))
    else:
        trainer(args.gpu, save_dir, ngpus_per_node, args)


if __name__ == "__main__":

    main()
