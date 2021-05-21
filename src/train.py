from library import *


if __name__ == "__main__":
    args = Arguments().parse()
    assert args.train_dir, "Path to a directory with train samples must be specified."
    assert args.valid_dir, "Path to a directory with validation samples must be specified."

    trainer = Trainer(args)
    trainer.train()
