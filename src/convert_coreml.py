import coremltools as ct
import torch
import json
import argparse
from pathlib import Path

from library.utils import clamped_sigmoid, nms
from library.model import Network


class RawDecoder:
    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        heatmaps = nms(clamped_sigmoid(output[:, :3]))
        return torch.cat(tensors=(heatmaps, output[:, 3:]), dim=1)


class CoreMLModel(torch.nn.Module):
    def __init__(self, model: Network) -> None:
        super().__init__()
        self.model = model
        self.decoder = RawDecoder()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        return self.decoder(output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Path to the PyTorch model to convert.")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="model.mlpackage",
        help="Output file name of the CoreML model.",
    )

    parser.add_argument(
        "--width", "-W", default=512, type=int, help="The network input width."
    )
    parser.add_argument(
        "--height", "-H", default=512, type=int, help="The network input height."
    )
    parser.add_argument(
        "--params",
        "-p",
        type=str,
        default="labels.json",
        help="Json file of anchor and part names. See 'labels.json' for an example.",
    )
    parser.add_argument(
        "--scale-factor",
        "-s",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--fpn-depth",
        type=int,
        default=128,
        help="Depth of FPN layers of the decoder.",
    )

    parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        default=None,
        help="Quantize the model to UInt8.",
    )

    args = parser.parse_args()

    labels_file = Path(args.params).expanduser().resolve()
    name_list = json.loads(labels_file.read_text())

    if isinstance(name_list["labels"], list):
        args.labels = name_list["labels"]
    else:
        raise ValueError("should be a list")

    if isinstance(name_list["parts"], list):
        args.parts = name_list["parts"]
    else:
        raise ValueError("should be a list")

    return args


def main():
    args = parse_args()

    model = Network(args, pretrained=False, raw_output=True)
    model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    model.eval()

    model = CoreMLModel(model)
    model.eval()

    input = torch.randn(1, 3, args.height, args.width)
    model_traced = torch.jit.trace(model, input)

    # ImageNet normalization
    scale = 1 / (0.226 * 255.0)
    bias = [-0.485 / (0.229), -0.456 / (0.224), -0.406 / (0.225)]

    mlmodel: ct.models.MLModel = ct.convert(
        model_traced,
        # inputs=[ct.TensorType(name="image", shape=input.shape)],
        inputs=[ct.ImageType(name="image", shape=input.shape, scale=scale, bias=bias)],
        outputs=[ct.TensorType(name="output")],
    )

    mlmodel.author = "Louis Lac"
    mlmodel.license = "MIT"
    mlmodel.short_description = "SDNet"
    mlmodel.version = "v1.0"

    params = {
        "anchors": args.labels,
        "parts": args.parts,
        "scale_factor": args.scale_factor,
        "width": args.width,
        "height": args.height,
    }

    mlmodel.user_defined_metadata["params"] = json.dumps(params)

    if args.quantize is not None:
        path = str(Path(args.quantize).expanduser().resolve())

        mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
            mlmodel,
            nbits=8,
            sample_data=path,
        )

    mlmodel.save(args.output)


if __name__ == "__main__":
    main()
