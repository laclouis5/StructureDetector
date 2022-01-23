from library import *
from torchvision.transforms import ToPILImage


def main():
    args = Arguments().parse()
    assert args.valid_dir, "Path to a directory with validation samples must be specified."

    dataset = Dataset(args.valid_dir, TrainAugmentation(args))
    data = dataset[0]

    # # Image & annotation
    im, ann = data["image"], data["annotation"]
    im = un_normalize(im)
    im = ToPILImage()(im)
    im = draw_tree(im, ann.tree)
    im.show()

    # # Heatmaps
    heatmaps = data["heatmaps"]
    heatmaps = draw_heatmaps(heatmaps, args)
    heatmaps.show()
    

if __name__ == "__main__":
    main()