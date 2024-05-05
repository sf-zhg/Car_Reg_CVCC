"""
This script performs a regression model experiment with data loading, model setup, training, and inference.

The script initializes the dataset, splits it into training and testing sets, sets up the model, and defines the optimizer and loss function.
It then trains the model over a specified number of epochs, running inference at defined intervals, and saves the results in CSV files.
"""

import argparse
import os
import time
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from Persp_estimator.data.get_data import get_data
from Persp_estimator.models.get_model import get_model
from Persp_estimator.get_loss import get_loss
from Persp_estimator.trainer import trainer
from Persp_estimator.inference import inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression Model")
    parser.add_argument(
        "--dataset", default="car_persp", type=str, help="dataset to train on"
    )
    parser.add_argument(
        "--split_ratio",
        default="0.5",
        type=float,
        help="proportion of train set within the whole dataset",
    )
    parser.add_argument(
        "--root", default="Persp_estimator/raw_data", type=str, help="path to data"
    )
    parser.add_argument("--batch_size", default="128", type=int, help="mini batch size")
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone model")
    parser.add_argument(
        "--feature_dim",
        default="128",
        type=int,
        help="hidden feature dimension in the regression heads",
    )
    parser.add_argument(
        "--drop_out_rate",
        default="0.4",
        type=float,
        help="percentage of neurons randomly dropped",
    )
    parser.add_argument(
        "--optimizer", default="adam", type=str, help="optimization algorithm"
    )
    parser.add_argument(
        "--learning_rate", default="0.01", type=float, help="learning rate"
    )
    parser.add_argument(
        "--loss",
        default="l2",
        type=str,
        help="loss function, choose between l1, l2 and huber",
    )
    parser.add_argument(
        "--pretrain_path",
        default="results/pretrein_weights",
        type=str,
        help="path to folder, where weights are saved",
    )
    parser.add_argument(
        "--loss_csv_path",
        default="results/loss_csv",
        type=str,
        help="optimization algorithm",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="trainings epoch, choose large number when rich",
    )
    parser.add_argument(
        "--inf_schedule",
        default=5,
        type=int,
        help="schedule for running inference, if 1 inferene is run for every epoch",
    )

    args = parser.parse_args()
    print(args)

    train_data, test_data = get_data(args.dataset, args.split_ratio, args.root)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = get_model(
        arch=args.arch, feature_dim=args.feature_dim, drop_out_rate=args.drop_out_rate
    )

    model = nn.DataParallel(model)

    if args.optimizer == "adam":
        # lets ignore weight decay for now
        opti = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        opti = optim.SGD(params=model.parameters(), lr=args.learning_rate)
    else:
        raise AttributeError(
            f"only sgd und adam defined, choose one of those or write additional optimizer."
        )

    loss_fn = get_loss(loss=args.loss)

    os.makedirs(args.pretrain_path, exist_ok=True)
    os.makedirs(args.loss_csv_path, exist_ok=True)

    writer = SummaryWriter(comment="pretraining rgression")
    t0 = time.time()

    train_results = pd.DataFrame(
        columns=["Epoch", "Total Loss", "Hood Loss", "Backdoor Loss"]
    )
    inf_results = pd.DataFrame(
        columns=["Epoch", "Total Loss", "Hood Loss", "Backdoor Loss"]
    )

    for epoch in range(1, args.epochs + 1):
        total_loss, hood_loss, backdoor_loss = trainer(
            data_loader=train_loader, optimizer=opti, loss=loss_fn, model=model
        )
        writer.add_scalar("Train/Total Loss", total_loss, epoch)
        writer.add_scalar("Train/Hood Loss", hood_loss, epoch)
        writer.add_scalar("Train/Backdoor Loss", backdoor_loss, epoch)

        train_results = train_results.append(
            {
                "Epoch": epoch,
                "Total Loss": total_loss,
                "Hood Loss": hood_loss,
                "Backdoor Loss": backdoor_loss,
            },
            ignore_index=True,
        )

        torch.save(
            model.module.f.state_dict(), "results/pretrained_{}.pth".format(epoch)
        )
        torch.save(model.state_dict(), "results/pretrained_whole_{}.pth".format(epoch))

        if epoch % args.inf_schedule == 0:
            total_loss_inf, hood_loss_inf, backdoor_loss_inf = inference(
                data_loader=test_loader, model=model, loss=loss_fn
            )
            writer.add_scalar("Val/Total Loss", total_loss_inf, epoch)
            writer.add_scalar("Val/Hood Loss", hood_loss_inf, epoch)
            writer.add_scalar("Val/Backdoor Loss", backdoor_loss_inf, epoch)

            inf_results = inf_results.append(
                {
                    "Epoch": epoch,
                    "Total Loss": total_loss_inf,
                    "Hood Loss": hood_loss_inf,
                    "Backdoor Loss": backdoor_loss_inf,
                },
                ignore_index=True,
            )

    t1 = time.time() - t0

    print("Time elapsed in hours: ", t1 / 3600)

    print("save scv files with results")

    train_results.to_csv(
        os.path.join(args.loss_csv_path, "train_results.csv"), index=False
    )
    inf_results.to_csv(
        os.path.join(args.loss_csv_path, "inference_results.csv"), index=False
    )

    print("csv files saved")
