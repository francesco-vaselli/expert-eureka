import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, poisson


def validate(
    test_loader,
    model,
    epoch,
    writer,
    args,
    device,
):
    model.eval()

    # samples
    with torch.no_grad():

        samples = []
        full_sim = []
        context = []

        for _, (x, y) in enumerate(test_loader):
            # print('x', x.shape, 'y', y.shape, 'N', N.shape)
            inputs_y = y.to(device)
            # print('inputs_y', inputs_y.shape)
            x_sampled = model.sample(
                    num_samples=1, context=inputs_y.view(-1, args.y_dim)
                )

            x_sampled = x_sampled.cpu().detach().numpy()
            inputs_y = inputs_y.cpu().detach().numpy()
            x = x.cpu().detach().numpy()

            samples.append(x_sampled)
            full_sim.append(x)
            context.append(inputs_y)

    generated_samples = np.reshape(samples, (len(test_loader.dataset), args.x_dim))
    full_sim = np.reshape(full_sim, (len(test_loader.dataset), args.x_dim))
    context = np.reshape(context, (len(test_loader.dataset), args.y_dim))

    names = ["rho0", "phi0", "rho1", "phi1", "quad", "poisson"]

    for i in range(0, len(names)):
        generated_sample = generated_samples[:, i]
        test_values = full_sim[:, i]
        ws = wasserstein_distance(test_values, generated_sample)
        # print(generated_sample.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)


        _, rangeR, _ = ax1.hist(
            test_values, histtype="step", label="FullSim", lw=1, bins=100
        )
        # print(rangeR.shape)
        generated_sample = np.where(
            generated_sample < rangeR.min(), rangeR.min(), generated_sample
        )
        generated_sample = np.where(
            generated_sample > rangeR.max(), rangeR.max(), generated_sample
        )

        ax1.hist(
            generated_sample,
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        fig.suptitle(f"Comparison of {names[i]} @ epoch {epoch}", fontsize=16)
        ax1.legend(frameon=False, loc="upper right")

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yscale("log")

        ax2.hist(test_values, histtype="step", lw=1, bins=100)
        ax2.hist(
            generated_sample,
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
        )

        writer.add_figure(f"comparison_{names[i]}", fig, global_step=epoch)
        plt.close()
        writer.add_scalar(f"ws/{names[i]}", ws, epoch)

    px0_sampled = generated_samples[:, 0] * np.cos(generated_samples[:, 1])
    py0_sampled = generated_samples[:, 0] * np.sin(generated_samples[:, 1])
    px1_sampled = generated_samples[:, 2] * np.cos(generated_samples[:, 3])
    py1_sampled = generated_samples[:, 2] * np.sin(generated_samples[:, 3])
    rhoT_sampled = np.hypot(px0_sampled + px1_sampled, py0_sampled + py1_sampled)
    phiT_sampled = np.arctan2(py0_sampled + py1_sampled, px0_sampled + px1_sampled)

    gen_sampled = [rhoT_sampled, phiT_sampled]
    names = ["rhoT", "phiT"]

    for n in range(0, len(names)):
        generated_sample = gen_sampled[n].flatten()
        test_values = context[:, n]
        ws = wasserstein_distance(test_values, generated_sample)
        # print(generated_sample.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)


        _, rangeR, _ = ax1.hist(
            test_values, histtype="step", label="FullSim", lw=1, bins=100
        )
        # print(rangeR.shape)
        generated_sample = np.where(
            generated_sample < rangeR.min(), rangeR.min(), generated_sample
        )
        generated_sample = np.where(
            generated_sample > rangeR.max(), rangeR.max(), generated_sample
        )

        ax1.hist(
            generated_sample,
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
            label=f"FlashSim, ws={round(ws, 4)}",
        )
        fig.suptitle(f"Comparison of {names[n]} @ epoch {epoch}", fontsize=16)
        ax1.legend(frameon=False, loc="upper right")

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yscale("log")

        ax2.hist(test_values, histtype="step", lw=1, bins=100)
        ax2.hist(
            generated_sample,
            bins=100,
            histtype="step",
            lw=1,
            range=[rangeR.min(), rangeR.max()],
        )

        writer.add_figure(f"comparison_{names[n]}", fig, global_step=epoch)
        writer.add_scalar(f"ws/{names[n]}", ws, epoch)
        plt.close()

    fixed_rho = np.full((10000, 1), 1)
    fixed_phi = np.full((10000, 1), 0.75)
    fixed_context = np.hstack((fixed_rho, fixed_phi))
    fixed_context = torch.from_numpy(fixed_context).float().to(device)

    x_sampled = model.sample(
                    num_samples=1, context=fixed_context.view(-1, args.y_dim)
                )

    x_sampled = x_sampled.cpu().detach().numpy().reshape((10000, args.x_dim))
    test_values = x_sampled[:, 5]*16 # rescaled by the dataset divide factor

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=False)
    _, rangeR, _ = ax1.hist(
        test_values, histtype="step", label="FlashSim", lw=1, bins=100, density=True
    )
    xmin, xmax = ax1.get_xlim()
    xp = np.linspace(xmin, xmax, 10000)
    p = poisson.pmf(xp, 1)
    ax1.plot(xp, p, label="Analytic Poisson", lw=1)

    fig.suptitle(f"poisson @ epoch {epoch}", fontsize=16)
    ax1.legend(frameon=False, loc="upper right")

    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_yscale("log")

    ax2.hist(test_values, histtype="step", lw=1, bins=100)

    writer.add_figure(f"poisson", fig, global_step=epoch)
    plt.close()