from library import *
from pathlib import Path
import json


def main():
    args = Arguments().parse()
    assert args.valid_dir, "Path to a directory with validation samples must be specified."

    dataset = Dataset(args, args.valid_dir)
    img, ann = dataset[0]
    
    Path("~/Downloads/ann.json").expanduser().write_text(json.dumps(ann.json_repr(), indent=2))

    img = draw_graph(img, ann)
    img.show()
    

if __name__ == "__main__":
    main()
