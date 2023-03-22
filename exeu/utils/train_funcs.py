# the train funcs for training the model
import torch
from pathlib import Path
import time 

def train_epoch_profiled(
    flow,
    train_loader,
    optimizer,
    epoch,
    device=None,
    output_freq=50,
    args=None,
    add_noise=True,
    annealing=False,
):
    """Train model for one epoch.
    Arguments:
        flow {Flow} -- NSF model
        train_loader {DataLoader} -- train set data loader
        optimizer {Optimizer} -- model optimizer
        epoch {int} -- epoch number
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        output_freq {int} -- frequency for printing status (default: {50})
    Returns:
        float -- average train loss over epoch
    """

    flow.train()
    train_loss = 0.0
    train_log_p = 0.0
    train_log_det = 0.0

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'runs/{args.log_name}/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
        ) as prof:

        for batch_idx, (z, y) in enumerate(train_loader):
            optimizer.zero_grad()

            if device is not None:
                z = z.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            # Compute log prob
            log_p, log_det = flow.log_prob(z, context=y)
            loss = -log_p - log_det

            # Keep track of total loss.
            train_loss += (loss.detach()).sum()
            train_log_p += (-log_p.detach()).sum()
            train_log_det += (-log_det.detach()).sum()


            # loss = (w * loss).sum() / w.sum()
            loss = (loss).mean()

            loss.backward()
            optimizer.step()
            prof.step()

            if (output_freq is not None) and (batch_idx % output_freq == 0):
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}".format(
                        epoch,
                        batch_idx * train_loader.batch_size,
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    train_loss = train_loss.item() / len(train_loader.dataset)
    train_log_p = train_log_p.item()  / len(train_loader.dataset)
    train_log_det = train_log_det.item()  / len(train_loader.dataset)
    print(
        "Model:{} Train Epoch: {} \tAverage Loss: {:.4f}".format(
            args.log_name, epoch, train_loss
        )
    )
    

    return train_loss, train_log_p, train_log_det


def train_epoch(
    flow,
    train_loader,
    optimizer,
    epoch,
    device=None,
    output_freq=50,
    args=None,
    add_noise=True,
    annealing=False,
):
    """Train model for one epoch.
    Arguments:
        flow {Flow} -- NSF model
        train_loader {DataLoader} -- train set data loader
        optimizer {Optimizer} -- model optimizer
        epoch {int} -- epoch number
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        output_freq {int} -- frequency for printing status (default: {50})
    Returns:
        float -- average train loss over epoch
    """

    flow.train()
    train_loss = 0.0
    train_log_p = 0.0
    train_log_det = 0.0

    for batch_idx, (z, y) in enumerate(train_loader):
        t = time.time()
        optimizer.zero_grad()

        if device is not None:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        # Compute log prob
        log_p, log_det = flow.log_prob(z, context=y)
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
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} in time {:.2f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    time.time() - t
                )
            )

    train_loss = train_loss.item() / len(train_loader.dataset)
    train_log_p = train_log_p.item()  / len(train_loader.dataset)
    train_log_det = train_log_det.item()  / len(train_loader.dataset)
    print(
        "Model:{} Train Epoch: {} \tAverage Loss: {:.4f}".format(
            args.log_name, epoch, train_loss
        )
    )

    return train_loss, train_log_p, train_log_det


def test_epoch(flow, test_loader, epoch, device=None):
    """Calculate test loss for one epoch.
    Arguments:
        flow {Flow} -- NSF model
        test_loader {DataLoader} -- test set data loader
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPu) (default: {None})
    Returns:
        float -- test loss
    """
    

    with torch.no_grad():
        flow.eval()
        test_loss = 0.0
        test_log_p = 0.0
        test_log_det = 0.0

        for z, y in test_loader:

            if device is not None:
                z = z.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

            # Compute log prob
            log_p, log_det = flow.log_prob(z, context=y)
            loss = -log_p - log_det

            # Keep track of total loss.
            test_loss += (loss.detach()).sum()
            test_log_p += (-log_p.detach()).sum()
            test_log_det += (-log_det.detach()).sum()

        test_loss = test_loss.item() / len(test_loader.dataset) 
        test_log_p = test_log_p.item()  / len(test_loader.dataset)
        test_log_det = test_log_det.item()  / len(test_loader.dataset) 
        # test_loss = test_loss.item() / total_weight.item()
        print("Test set: Average loss: {:.4f}\n".format(test_loss))

        return test_loss, test_log_p, test_log_det

def train(
    model,
    train_loader,
    test_loader,
    epochs,
    optimizer,
    device,
    name,
    model_dir,
    args,
    writer=None,
    output_freq=10,
    save_freq=10,
    res_epoch=0,
    val_func=None,
    val_func_args=None,
):
    """Train the model.
    Args:
            epochs:     number of epochs to train for
            output_freq:    how many iterations between outputs
    """
    train_history = []
    test_history = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    for epoch in range(0 + res_epoch, epochs + 1 + res_epoch):

        print(
            "Learning rate: {}".format(optimizer.state_dict()["param_groups"][0]["lr"])
        )
        if args.profile == True:
            train_loss, train_log_p, train_log_det = train_epoch_profiled(
                model, train_loader, optimizer, epoch, device, output_freq, args=args
            )
        else:
            train_loss, train_log_p, train_log_det = train_epoch(
                model, train_loader, optimizer, epoch, device, output_freq, args=args
            )
        test_loss, test_log_p, test_log_det = test_epoch(model, test_loader, epoch, device)

        scheduler.step()
        train_history.append(train_loss)
        test_history.append(test_loss)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("train/log_p", train_log_p, epoch)
            writer.add_scalar("test/log_p", test_log_p, epoch)
            writer.add_scalar("train/log_det", train_log_det, epoch)
            writer.add_scalar("test/log_det", test_log_det, epoch)


        if epoch % args.val_freq == 0:
            if val_func is not None:
                val_func(test_loader,
                            model,
                            epoch,
                            writer,
                            args,
                            device,)

        if epoch % save_freq == 0:

            save_model(
                epoch,
                model,
                scheduler,
                train_history,
                test_history,
                name,
                model_dir=model_dir,
                optimizer=optimizer,
            )
            print("saving model")

    return train_history, test_history


def save_model(
    epoch,
    model,
    scheduler,
    train_history,
    test_history,
    name,
    model_dir=None,
    optimizer=None,
):
    """Save a model and optimizer to file.
    Args:
        model:      model to be saved
        optimizer:  optimizer to be saved
        epoch:      current epoch number
        model_dir:  directory to save the model in
        filename:   filename for saved model
    """

    if model_dir is None:
        raise NameError("Model directory must be specified.")

    filename = name + f"_@epoch_{epoch}.pt"
    resume_filename = "checkpoint-latest.pt"

    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    dict = {
        "train_history": train_history,
        "test_history": test_history,
        # "model_hyperparams": model.model_hyperparams,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        dict["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(dict, p / filename)
    torch.save(dict, p / resume_filename)


def load_model(model, device, model_dir=None, filename=None):
    """Load a saved model.
    Args:
        filename:       File name
    """

    if model_dir is None:
        raise NameError(
            "Model directory must be specified."
            " Store in attribute PosteriorModel.model_dir"
        )

    p = Path(model_dir)
    checkpoint = torch.load(p / filename, map_location=device)

    # model_hyperparams = checkpoint["model_hyperparams"]
    train_history = checkpoint["train_history"]
    test_history = checkpoint["test_history"]

    # Load model
    # model = create_NDE_model(**model_hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    model.eval()

    # Load optimizer
    scheduler_present_in_checkpoint = "scheduler_state_dict" in checkpoint.keys()

    # If the optimizer has more than 1 param_group, then we built it with
    # flow_lr different from lr
    if len(checkpoint["optimizer_state_dict"]["param_groups"]) > 1:
        flow_lr = checkpoint["optimizer_state_dict"]["param_groups"][-1]["initial_lr"]
    else:
        flow_lr = None

    # Set the epoch to the correct value. This is needed to resume
    # training.
    epoch = checkpoint["epoch"]

    return (
        model,
        scheduler_present_in_checkpoint,
        flow_lr,
        epoch,
        train_history,
        test_history,
    )


