from library.data import *
from library.utils import *
from library.model import *
import torch
from tqdm import tqdm


args = Arguments().parse()
evaluator = Evaluator(args)
dataset = CropDataset(args, args.valid_dir, ValidationAugmentation(args))
dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=1, collate_fn=CropDataset.collate_fn,
    pin_memory=args.use_cuda,
    num_workers=8)

decoder = Decoder(args)
net = Network(args).to(args.device)
net.load_state_dict(torch.load(args.pretrained_model))
net.eval()

for batch in tqdm(dataloader, desc="Evaluation", unit="image"):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(args.device)

    output = net(batch["image"])
    img_size = batch["annotation"][0].img_size
    data = decoder(output, img_size=img_size, return_metadata=True)
    prediction = data["annotation"][0]
    annotation = batch["annotation"][0]
    evaluator.accumulate(prediction, annotation, data["raw_parts"][0], True, True)

print(evaluator)