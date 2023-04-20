from library import *
import torch
from tqdm import tqdm


def main():
    args = Arguments().parse()
    assert (
        args.valid_dir
    ), "Path to a directory with validation samples must be specified."
    assert (
        args.pretrained_model
    ), "No pretrained model specified. Use the option '--load_model <model_path>'."

    evaluator = Evaluator(args)
    dataset = CropDataset(args, args.valid_dir, ValidationAugmentation(args))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=CropDataset.collate_fn,
        pin_memory=args.use_cuda,
        num_workers=args.num_workers,
        multiprocessing_context="forkserver",
    )

    decoder = Decoder(args)
    net = Network(args)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=args.device))
    net = net.eval().to(args.device)

    for batch in tqdm(dataloader, desc="Evaluation", unit="image"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)

        with torch.no_grad():
            output = net(batch["image"])

        data = decoder(output, return_metadata=True)
        prediction = data["annotation"][0]
        annotation = batch["annotation"][0]
        evaluator.accumulate(prediction, annotation, data["raw_parts"][0], True, True)

    evaluator.pretty_print()
    # evaluator.classification_eval.reduce().save_conf_matrix()

    if args.csv_path is not None:
        evaluator.save_kps_csv(args.csv_path)


if __name__ == "__main__":
    main()
