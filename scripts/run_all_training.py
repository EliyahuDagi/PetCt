import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train import train_ae2d, train_diff2d, train_ft3d


def load_state(path):
    path_obj = Path(path)
    if not path_obj.exists():
        return {"completed": []}
    with path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)
    completed = data.get("completed", []) if isinstance(data, dict) else []
    return {"completed": list(completed)}


def save_state(path, completed):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump({"completed": list(completed)}, f, indent=2)


def run_module_main(main_fn, argv):
    sys.argv = argv
    main_fn()


def run_all(steps, state_path, resume=False, start_step=None):
    completed = []
    if resume:
        completed = load_state(state_path)["completed"]

    if start_step:
        pre = []
        for name, _, _ in steps:
            if name == start_step:
                break
            pre.append(name)
        completed = pre

    for name, fn, argv in steps:
        if name in completed:
            continue
        try:
            fn(argv)
            completed.append(name)
            save_state(state_path, completed)
        except Exception:
            save_state(state_path, completed)
            raise

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Dataset root or patient folder")
    parser.add_argument("--patient_index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--slice_size", type=int, default=128)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--latent_size_3d", type=int, default=64)
    parser.add_argument("--inflate_from", default=None)
    parser.add_argument("--state_path", default="outputs/run_all_state.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_step", default=None, choices=["ae2d", "diff2d", "ft3d"])
    args = parser.parse_args()

    steps = [
        (
            "ae2d",
            lambda argv: run_module_main(train_ae2d.main, argv),
            [
                "train_ae2d.py",
                "--data_dir",
                args.data_dir,
                "--patient_index",
                str(args.patient_index),
                "--device",
                args.device,
                "--steps",
                str(args.steps),
                "--slice_size",
                str(args.slice_size),
            ],
        ),
        (
            "diff2d",
            lambda argv: run_module_main(train_diff2d.main, argv),
            [
                "train_diff2d.py",
                "--data_dir",
                args.data_dir,
                "--patient_index",
                str(args.patient_index),
                "--device",
                args.device,
                "--steps",
                str(args.steps),
                "--latent_size",
                str(args.latent_size),
            ],
        ),
        (
            "ft3d",
            lambda argv: run_module_main(train_ft3d.main, argv),
            [
                "train_ft3d.py",
                "--data_dir",
                args.data_dir,
                "--patient_index",
                str(args.patient_index),
                "--device",
                args.device,
                "--steps",
                str(args.steps),
                "--latent_size",
                str(args.latent_size_3d),
            ]
            + (["--inflate_from", args.inflate_from] if args.inflate_from else []),
        ),
    ]

    run_all(steps, args.state_path, resume=args.resume, start_step=args.start_step)


if __name__ == "__main__":
    main()
