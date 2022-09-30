from library import *
from tqdm.contrib import tzip
import torch
from PIL import Image


if __name__ == "__main__":
    args = args = Arguments().parse()

    dataset = PredictionDataset(args.valid_dir, PredictionTransformation(args))
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=1,
        pin_memory=args.use_cuda,
        num_workers=args.num_workers)

    decoder = Decoder(args)
    # kp_decoder = KeypointDecoder(args)

    net = Network(args)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=args.device))
    net = net.to(args.device)
    net.eval()

    for batch, image_path in tzip(dataloader, dataset.images, desc="Prediction", unit="image"):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)

        with torch.no_grad():
            output = net(batch["img"])
            
        img_w, img_h = batch["img_size"]
        img_size = img_w.item(), img_h.item()
        annotation = decoder(output)[0]
        annotation.resize((args.width, args.height), img_size)
        annotation.img_size = img_size
        annotation.image_path = image_path
        # keypoints = kp_decoder(output)[0]
        # keypoints = [kp.resize((args.width, args.height), img_size) for kp in keypoints]

        image = Image.open(image_path)
        image = draw(image, annotation, args)
        # image = draw_keypoints(image, keypoints, args)

        annotation.save_json("predictions")
        image.save(Path(f"predictions/{image_path.name}"))