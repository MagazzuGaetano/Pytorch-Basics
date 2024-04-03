import argparse


parser = argparse.ArgumentParser(description="Pytorch-Basics")
parser.add_argument(
    "--learning_rate", type=float, default=1e-3, help="adam learning rate"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="adam weight decay",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="number of instances in a batch of data (default: 64)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=20,
    help="maximum epochs to train (default: 20)",
)
parser.add_argument(
    "--train_print_freq",
    type=int,
    default=50,
    help="how often does it print train metrics (default: 50 step)",
)
parser.add_argument(
    "--val_print_freq",
    type=int,
    default=1,
    help="how often does it print validation metrics (default: 1 epoch)",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=10,
    help="number of classes to recognize: dataset dependent",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for reproducibility",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="model.pth",
    help="checkpoint path",
)
