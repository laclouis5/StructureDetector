import torch
from .network import Network
from ..data import Decoder
import torchvision.transforms as torchtf


class Predictor(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.model = Network(args)
        self.model.load_state_dict(
            torch.load(args.pretrained_model, map_location=args.device)
        )
        self.model.eval().to(args.device)

        self.decoder = Decoder(args)

        self.transform = torch.nn.Sequential(
            torchtf.Resize((args.height, args.width)),
            torchtf.ToTensor(),
            torchtf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        self.transform.to(args.device)

    def forward(self, image):

        with torch.no_grad():
            input = image.to(self.args.device)
            input = self.transform(image)
            prediction = self.model(input[None, ...])
            prediction = self.decoder(prediction)[0]
