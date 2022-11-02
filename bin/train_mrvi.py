import argparse


def train_mrvi()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MrVI model on a dataset.")

    parser.add_argument(
        "dataset_name",
        type=str,
    )
    parser.add_argument(
        "config",
        type=str,
    )
    args = parser.parser_args()