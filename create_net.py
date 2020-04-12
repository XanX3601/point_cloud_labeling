import argparse

import torch

import utils

if __name__ == "__main__":
    # Parser
    # ---------------
    parser = argparse.ArgumentParser("Script to create a neural net")
    parser.add_argument("--cuda", help="If using GPU", action="store_true")
    parser.add_argument("--path", help="Model path",
                        type=str, default="networks/neural_net.pt")

    # Parse argument
    # ---------------
    args = parser.parse_args()

    # Using CUDA if asked and available
    # ---------------
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Creating and saving the neural net
    # ---------------
    model = utils.Net().to(device)
    torch.save(model, args.path)
