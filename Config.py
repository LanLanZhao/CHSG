import argparse


DATASETS = (
    "indian",
    "paviau",
    "ksc",
    "sali",
    "botswana",
    "houston",
    "hanchuan",
    "honghu",
    "longkou",
    "houston2018",
    "paviac",
    "SZUR1",
    "SZUR2",
    "UP",
    "HC",
    "NF",
    "loukia",
    "dioni",
    "tea",
    "xuzhou",
    "chi",
)


def ExperimentParams():
    parser = argparse.ArgumentParser(
        description="Train CHSG for hyperspectral image classification."
    )
    parser.add_argument("--data-root", default="./dataset", help="Directory containing the .mat datasets.")
    parser.add_argument("--output-dir", default="./results", help="Directory used for generated experiment outputs.")
    parser.add_argument("--dataset", default="houston", choices=DATASETS)
    parser.add_argument("--model", default="CHSG", choices=("CHSG",))
    parser.add_argument("--Experiment_num", default=10, type=int, help="Number of repeated experiments.")
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", "--wd", default=3e-4, type=float)
    parser.add_argument("--components", default=0, type=int, help="PCA components; 0 disables PCA.")
    parser.add_argument("--random_state", nargs="+", default=[1, 2, 3, 4, 0, 6, 7, 8, 9, 10], type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--split_type", default="fixed", choices=("fixed", "ratio"))
    parser.add_argument("--train_num", default=30, type=int, help="Training samples per class in fixed mode.")
    parser.add_argument("--train_ratio", default=0.05, type=float)
    parser.add_argument("--data_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm", default="max_min_norm", choices=("max_min_norm", "mean_var_norm"))
    parser.add_argument("--patch_size", default=13, type=int)
    parser.add_argument(
        "--boundary-mode",
        default="cyclic",
        choices=("cyclic", "reflect", "replicate", "zero"),
        help="Boundary strategy used by CHSG graph aggregation.",
    )
    return parser.parse_args()
