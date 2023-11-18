from ..model import Trainer
from ..utils import Arguments


def main():
    args = Arguments().parse()
    assert args.train_dir, "Path to a directory with train samples must be specified."
    assert (
        args.valid_dir
    ), "Path to a directory with validation samples must be specified."

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
