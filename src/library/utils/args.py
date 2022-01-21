import argparse
import json
from pathlib import Path
import torch
from .utils import get_unique_color_map, set_seed


class Arguments:

    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("--train_dir", type=str, help="The training directory.")
        parser.add_argument("--valid_dir", type=str, help="The validation directory.")
        parser.add_argument("--labels", "-m", type=str, default="labels.json")

        # Training parameters
        parser.add_argument("--width", "-W", default=512, type=int,
            help="The network input width.")
        parser.add_argument("--height", "-H", default=512, type=int,
            help="The network input height.")
        parser.add_argument("--in_channels", "-c", default=3, type=int,
            help="Number of input channels, 3 if color image, 1 is grayscale. You may need to adapt the image opening and visualization code.")
        parser.add_argument("--batch_size", "-b", default=8, type=int,
            help="Batch size for training.")
        parser.add_argument("--epochs", "-e", type=int, default=100,
            help="The number of epochs to train.")
        parser.add_argument("--learning_rate", "-l", type=float, default=1e-3,
            help="The learning rate for training.")
        parser.add_argument("--lr_step", type=int, default=3, 
            help="Number of divisions by 10 of the learning rate during training (happends every int(epochs / lr_step)). 0 = deactivated.")
        parser.add_argument("--load_model", "-o", default=None, dest="pretrained_model",
            help="Load a previously trained model for evaluation of inference.")

        # Encoder
        parser.add_argument("--sigma_gauss", type=float, default=10/100,
            help="Size of the gaussian filter used to lay keypoints on heatmaps. Expressed in percent of the image side length.")
        parser.add_argument("--no_augmentation", "-a", action="store_true",
            help="Activate augmentations during training (random LR flip, random color change, random rescale, ...).")

        # Decoder
        parser.add_argument("--conf_threshold", "-t", type=float, default=50/100,
            help="Confidence threshold for keypoint detection. Must be in [0, 1].")
        parser.add_argument("--decoder_dist_thresh", type=float, default=10/100,
            help="Radius in percent of min image length considered for keypoint linkage. Must be in [0, 1].")
        parser.add_argument("--max_objects", "-n", type=int, default=100,
            help="The maximum number of objects that can be detected in an image. Can affect performance and memory consumption.")
        
        # Loss function
        parser.add_argument("--hm_loss_fn", "-f", type=str, default="mse",
            help="Loss for heatmaps regression. Options: 'focal' and 'mse'.")
        parser.add_argument("--hm_weight", type=float, default=1.0,
            help="Weight for the heatmap loss.")
        parser.add_argument("--offset_weight", type=float, default=0.001,
            help="Weight for the offset loss.")
        parser.add_argument("--embedding_weight", type=float, default=0.001,
            help="Weight for the embedding loss.")

        # Other nerwork parameters
        parser.add_argument("--down_ratio", "-g", type=float, default=4.0,
            help="The Downsampling ratio introduced by pooling layers of the network. Change it only if you change network architecture.")
        parser.add_argument("--fpn_depth", type=int, default=128,
            help="Depth of FPN layers of the decoder.")

        # Evaluation
        parser.add_argument("--dist_threshold", "-d", type=float, default=5/100,
            help="Radius in percent of min image length considered for evaluation. Must be in [0, 1].")

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        assert args.width % 32 == 0 and args.width > 0, "Width should be divisible by 32 and greater than 0"
        assert args.height % 32 == 0 and args.height > 0, "Height should be divisble by 32 and greater than 0"
        assert args.in_channels > 0, "'in_channels' should be greater than 0"
        assert args.fpn_depth > 0, "'fpn_depth' should be greater than 0"
        assert args.batch_size > 0, "'batch_size' should be greater than 0"
        assert args.epochs > 0, "'epochs' should be greater than 0"
        assert args.learning_rate > 0, "'learning_rate' should be greater than 0"
        assert args.lr_step >= 0, "'lr_step should be greater of equal to 0'"
        assert args.down_ratio > 0, "'down_ratio' should be greater than 0"
        assert args.max_objects > 0, "'max_objects' should be greater than 0"

        assert args.hm_weight >= 0, "'hm_weight' should be greater than or equal to 0"
        assert args.offset_weight >= 0, "'offset_weight' should be greater than or equal to 0"
        assert args.embedding_weight >= 0, "'embedding_weight' should be greater than or equal to 0"

        assert 0 <= args.conf_threshold <= 1, "'conf_threshold' should be in [0.0, 1.0]"
        assert 0 <= args.dist_threshold <= 1, "'dist_threshold' should be in [0.0, 1.0]"
        assert 0 <= args.decoder_dist_thresh <= 1, "'decoder_dist_threshold' should be in [0.0, 1.0]"
        assert 0 < args.sigma_gauss <= 1, "'sigma_gauss' should be in ]0.0, 1.0]"

        args.valid_dir = Path(args.valid_dir).expanduser().resolve()
        # args.train_dir = Path(args.train_dir).expanduser().resolve()

        args.lr_step = int(args.epochs / args.lr_step) if args.lr_step != 0 else args.epochs

        name_list = json.loads(Path(args.labels).expanduser().resolve().read_text())
        args.labels = {v: i for i, v in enumerate(name_list["labels"])}

        args.use_cuda = torch.cuda.is_available()
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.num_workers = 8 if args.use_cuda else 4

        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

        set_seed(926354916)

        if args.hm_loss_fn.lower() not in {"focal", "mse"}:
            raise IOError(f"'hm_loss_fn' should either be 'focal' or 'mse', not {args.hm_loss_fn}.")

        args._r_labels = {v: k for k, v in args.labels.items()}

        return args
