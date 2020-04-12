import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

import utils

if __name__ == "__main__":
    # Parser
    # ---------------
    parser = argparse.ArgumentParser("Script to train a neural net")
    parser.add_argument("--epochs", help="Epochs", default=42, type=int)
    parser.add_argument("--cuda", help="Is using CUDA", action="store_true")
    parser.add_argument("--batch-size", help="Batch size",
                        default=32, type=int)
    parser.add_argument("--path", help="Net path",
                        default="networks/neural_net.pt", type=str)

    # Parse arguments
    # ---------------
    args = parser.parse_args()

    # Using CUDA if asked and available
    # ---------------
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load model
    # ---------------
    model = torch.load(args.path).to(device)

    # Loss and optimizer
    # ---------------
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.95,
    )

    # Data
    # ---------------
    dataset = utils.Dataset()
    data_loader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    last_batch_index = len(dataset) // args.batch_size
    if len(dataset) % args.batch_size == 0:
        last_batch_index -= 1

    # Train
    # ---------------
    for epoch in range(args.epochs):
        t = tqdm(
            enumerate(data_loader),
            desc="Epoch {:03d} / {} - Batch loss = ---".format(
                epoch + 1, args.epochs),
            total=len(dataset) // args.batch_size,
            unit="batch",
        )

        model.train()
        running_loss = 0.0

        for i_batch, batch in t:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(torch.log(y_pred), y)
            running_loss += loss.item()
            t.set_description(
                "Epoch {:03d} / {} - Batch loss = {:.9f}".format(
                    epoch + 1, args.epochs, loss.item()))
            loss.backward()
            optimizer.step()

            if i_batch == last_batch_index:
                t.set_postfix_str("mean_total_loss = {:.9f}".format(
                    running_loss / (last_batch_index + 1)))

        torch.save(model, args.path)
        scheduler.step()
