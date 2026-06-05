import argparse
import sys

from src.training.train import train_ae2d, train_ae3d, train_diff2d, train_ft3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["ae2d", "ae3d", "diff2d", "ft3d"], help="Training task to run")
    args, remaining = parser.parse_known_args()

    if args.task == "ae2d":
        sys.argv = ["train_ae2d.py"] + remaining
        train_ae2d.main()
    elif args.task == "ae3d":
        sys.argv = ["train_ae3d.py"] + remaining
        train_ae3d.main()
    elif args.task == "diff2d":
        sys.argv = ["train_diff2d.py"] + remaining
        train_diff2d.main()
    elif args.task == "ft3d":
        sys.argv = ["train_ft3d.py"] + remaining
        train_ft3d.main()


if __name__ == "__main__":
    main()
