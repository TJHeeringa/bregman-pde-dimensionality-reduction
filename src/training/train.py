import copy
import math
from pathlib import Path
import sys

import wandb
import torch
from tqdm import tqdm

import bregman

sys.path.insert(0, str(Path.cwd()))

from src.utils import L12_nuclear, init_linear  # noqa: E402


def train(train_dataset, test_dataset, validation_dataset, model_storage_folder, config=None):
    torch.autograd.set_detect_anomaly(True)
    with wandb.init(config=config) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        # create the loaders for the train and test sets
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size_train,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size_test,
            shuffle=False,
        )

        # define network to use
        model = bregman.AutoEncoder(encoder_layers=config.encoder_layers, decoder_layers=config.decoder_layers)

        # initialize weights and biases
        model.apply(init_linear)

        # reduce the row density of the weight matrices of the network to approximately
        # the required initial sparsity level
        bregman.sparsify(model, config.initial_sparsity)

        model.to(device)

        # pick the right optimizer
        match config.optimizer:
            case "AdaBreg":
                optimizer = bregman.AdaBreg(
                    L12_nuclear(model, config.regularization_constant),
                    lr=config.learning_rate,
                )
            case "LinBreg":
                optimizer = bregman.LinBreg(
                    L12_nuclear(model, config.regularization_constant),
                    lr=config.learning_rate,
                    momentum=config.momentum,
                )
            case "Adam":
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.regularization_constant,
                )
            case "SGD":
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.regularization_constant,
                )

        # define the loss function to use
        loss_functional = torch.nn.MSELoss(reduction="sum")

        total_batch_train_loss = 0.0
        for (batch,) in tqdm(train_loader):
            batch.to(device)
            loss = loss_functional(batch, torch.zeros_like(batch))
            total_batch_train_loss += loss.item()

        total_batch_test_loss = 0.0
        for (batch,) in tqdm(test_loader):
            batch.to(device)
            loss = loss_functional(batch, torch.zeros_like(batch))
            total_batch_test_loss += loss.item()

        best_model = model
        best_training_loss = 1e10
        best_testing_loss = 1e10

        for epoch in range(1, config.epochs + 1):
            if epoch % 20 == 1:
                print(50 * "-")
                print(f"epochs {epoch} to {min(epoch + 19, config.epochs)} " f"of the {config.epochs} total epochs")
                print(50 * "-")
            best_model_updated = False

            # <><><><><><><><><><><><><><><><>
            # Training step
            # <><><><><><><><><><><><><><><><>
            model.train(True)
            train_loss = 0.0
            for (batch,) in tqdm(train_loader, desc=f"Training epoch {epoch}"):
                batch.to(device)

                optimizer.zero_grad()

                loss = loss_functional(batch, model(batch))
                print(loss)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= total_batch_train_loss

            if train_loss < best_training_loss:
                best_model = copy.deepcopy(model)
                best_model.train(False)
                best_training_loss = train_loss
                best_model_updated = True

            # <><><><><><><><><><><><><><><><>
            # Testing step
            # <><><><><><><><><><><><><><><><>
            testing_loss = 0.0
            with torch.no_grad():
                for (batch,) in tqdm(test_loader, desc=f"Testing epoch {epoch}"):
                    batch.to(device)

                    loss = loss_functional(batch, model(batch))
                    testing_loss += loss.item()
            testing_loss /= total_batch_test_loss

            if best_model_updated:
                best_testing_loss = testing_loss

            # <><><><><><><><><><><><><><><><>
            # Logging
            # <><><><><><><><><><><><><><><><>
            if epoch % config.epoch_log_interval == 0 or epoch == config.epochs:
                log_dict = {
                    "Device": str(device),
                    "Loss (training)": train_loss,
                    "Loss (training, best)": best_training_loss,
                    "Loss (testing)": testing_loss,
                    "Loss (testing, best)": best_testing_loss,
                    "Network density": bregman.network_density(model, absolute=False),
                    "Network density (best)": bregman.network_density(best_model, absolute=False),
                    "Network param count": bregman.network_density(model, absolute=True),
                    "Network param count (best)": bregman.network_density(best_model, absolute=True),
                    "Latent dimension": model.latent_size("minimal"),
                    "Latent dimension (post)": model.latent_size("latent POD", 1e-8),
                    "Latent dimension (best)": best_model.latent_size("minimal"),
                    "Latent dimension (best post)": best_model.latent_size("latent POD", 1e-8),
                    "AIC/2": model.latent_size("minimal") + math.log(train_loss, 10),
                    "AIC/2 (best)": best_model.latent_size("minimal") + math.log(best_training_loss, 10),
                }
                if epoch == config.epochs:
                    pruned_model = bregman.simplify(bregman.latent_pod(best_model, 1e-8))

                    testing_loss = 0.0
                    with torch.no_grad():
                        for (batch,) in tqdm(test_loader, desc=f"Testing epoch {epoch}"):
                            batch.to(device)

                            loss = loss_functional(batch, pruned_model(batch))
                            testing_loss += loss.item()
                    testing_loss /= total_batch_test_loss

                    log_dict["Loss (testing, pruned)"] = testing_loss
                    log_dict["Network density (pruned)"] = bregman.network_density(pruned_model, absolute=False)
                    log_dict["Network param count (pruned)"] = bregman.network_density(pruned_model, absolute=True)
                    log_dict["Latent dimension (pruned)"] = pruned_model.latent_size("minimal")

                wandb.log(log_dict, step=epoch)

        # create folder for models to be stored, if it doesn't already exist
        model_storage_folder.mkdir(parents=True, exist_ok=True)

        # use torch to save the weights and biases
        final_model_file = model_storage_folder / f"model_{run.id}_final.pt"
        best_pruned_model_file = model_storage_folder / f"model_{run.id}_best_pruned.pt"
        best_model_file = model_storage_folder / f"model_{run.id}_best.pt"

        torch.save(model.state_dict(), final_model_file)
        torch.save(best_model.state_dict(), best_model_file)
        torch.save(pruned_model.state_dict(), best_pruned_model_file)

        # store the torch file also in wandb
        artifact = wandb.Artifact("final_model", type="model")
        artifact.add_file(final_model_file)
        run.log_artifact(artifact)

        artifact = wandb.Artifact("best_pruned_model", type="model")
        artifact.add_file(best_pruned_model_file)
        run.log_artifact(artifact)

        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(best_model_file)
        run.log_artifact(artifact)
