from library import *
import torch
from tqdm import tqdm


def main():
    args = Arguments().parse()
    assert args.valid_dir, "Path to a directory with validation samples must be specified."
    assert args.pretrained_model, "No pretrained model specified. Use the option '--load_model <model_path>'."
    
    evaluator = Evaluator(args)
    dataset = Dataset(args.valid_dir, ValidationAugmentation(args))
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=1, collate_fn=Dataset.collate_fn,
        pin_memory=args.use_cuda,
        num_workers=args.num_workers)

    decoder = Decoder(args)
    net = Network(args)
    net.load_state_dict(
        torch.load(args.pretrained_model, map_location=args.device))
    net = net.eval().to(args.device)

    for batch in tqdm(dataloader, desc="Evaluation", unit="image"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)

        with torch.no_grad():
            output = net(batch["image"])

        ground_truth = batch["annotation"][0].to_graph()
        ground_truth = ground_truth.resized((args.width, args.height), ground_truth.image_size)

        predicted_graph = decoder(output)[0]
        prediction = GraphAnnotation(
            ground_truth.image_path, 
            predicted_graph, 
            ground_truth.image_size)
        prediction = prediction.resized((args.width, args.height), prediction.image_size)

        evaluator.evaluate(prediction, ground_truth)

    evaluator.pretty_print()


if __name__ == "__main__":
    main()
