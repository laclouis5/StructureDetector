from library import *
import torch
from tqdm import tqdm
import argparse
import coremltools as ct
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="Path to the CoreML model.")
    parser.add_argument(
        "--valid_dir", type=str, help="Path to the validation directory."
    )

    parser.add_argument(
        "--anchor_name",
        "-s",
        type=str,
        default="anchor",
        help="Name of the keypoint representing the anchor of the object.",
    )
    parser.add_argument(
        "--max_objects",
        "-n",
        type=int,
        default=20,
        help="The maximum number of objects that can be detected in an image. Can affect performance and memory consumption.",
    )
    parser.add_argument(
        "--max_parts",
        "-k",
        type=int,
        default=40,
        help="The maximum number of parts that can be detected in an image. Can affect performance and memory consumption.",
    )
    parser.add_argument(
        "--sigma_gauss",
        type=float,
        default=10 / 100,
        help="Size of the gaussian filter used to lay keypoints on heatmaps. Expressed in percent of the image side length.",
    )

    parser.add_argument(
        "--conf_threshold",
        "-t",
        type=float,
        default=50 / 100,
        help="Confidence threshold for keypoint detection. Must be in [0, 1].",
    )
    parser.add_argument(
        "--dist_threshold",
        "-d",
        type=float,
        default=5 / 100,
        help="Radius in percent of min image length considered for evaluation. Must be in [0, 1].",
    )
    parser.add_argument(
        "--decoder_dist_thresh",
        type=float,
        default=10 / 100,
        help="Radius in percent of min image length considered for keypoint linkage. Must be in [0, 1].",
    )
    parser.add_argument(
        "--csi_threshold",
        type=float,
        default=75 / 100,
        help="Threshold on the custom CSI metric for evaluation. Must be in [0, 1].",
    )

    args = parser.parse_args()
    args.model = Path(args.model).expanduser().resolve()
    args.valid_dir = Path(args.valid_dir).expanduser().resolve()

    args.num_workers = 4

    return args


def main():
    args = parse_args()

    model = ct.models.MLModel(str(args.model))
    params = json.loads(model.user_defined_metadata["params"])

    args.down_ratio = params["scale_factor"]
    args.width = params["width"]
    args.height = params["height"]
    args.labels = {l: i for i, l in enumerate(params["anchors"])}
    args.parts = {p: i for i, p in enumerate(params["parts"])}
    args._r_labels = {v: k for k, v in args.labels.items()}
    args._r_parts = {v: k for k, v in args.parts.items()}

    nb_anchors = len(args.labels)
    nb_parts = len(args.parts)
    nb_hms = nb_anchors + nb_parts

    evaluator = Evaluator(args)
    decoder = Decoder(args)

    dataset = CropDataset(args, args.valid_dir, ValidationAugmentation(args))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=CropDataset.collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=4,
    )

    for batch in tqdm(dataloader, desc="Evaluation", unit="image"):
        output = model.predict({"input": batch["image"]})["output"]
        output = torch.from_numpy(output)
        output = {
            "anchor_hm": output[:, :nb_anchors],
            "part_hm": output[:, nb_anchors:nb_hms],
            "offsets": output[:, nb_hms : (nb_hms + 2)],
            "embeddings": output[:, (nb_hms + 2) : (nb_hms + 4)],
        }

        data = decoder(output, return_metadata=True)
        prediction = data["annotation"][0]
        annotation = batch["annotation"][0]
        evaluator.accumulate(prediction, annotation, data["raw_parts"][0], True, True)

    evaluator.pretty_print()


if __name__ == "__main__":
    main()
