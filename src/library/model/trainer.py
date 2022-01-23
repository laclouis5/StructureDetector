from .loss import Loss, LossStats
from .network import Network
from .evaluator import Evaluator
from ..data import *

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from pathlib import Path


class Trainer:

    def __init__(self, args):
        self.args = args
        self.net = Network(args, pretrained=True)
        self.loss = Loss(args)
        self.decoder = Decoder(args)
        self.evaluator = Evaluator(args)

        # Logging an metrics
        self.writer = SummaryWriter()
        self.global_step = 0
        self.best_loss = torch.finfo().max
        self.best_kp_reg = 0.0

        # TODO: Test this.
        if args.pretrained_model:
            self.net.load_state_dict(torch.load(args.pretrained_model, map_location=args.device))

        self.net.to(args.device)
        self.loss.to(args.device)

        # TODO: Add Apex mixed precision computing for faster training and inference

        self.optimizer = torch.optim.Adam(self.net.parameters(), args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step)

        self.train_set = Dataset(args, args.train_dir, transforms=TrainAugmentation(args))
        self.train_dataloader = data.DataLoader(self.train_set,
            batch_size=args.batch_size, collate_fn=Dataset.collate_fn, shuffle=True,
            pin_memory=args.use_cuda,
            num_workers=args.num_workers, 
            persistent_workers=True,
            drop_last=True)

        self.valid_set = Dataset(args, args.valid_dir, transforms=ValidationAugmentation(args))
        self.valid_dataloader = data.DataLoader(self.valid_set,
            batch_size=1, collate_fn=Dataset.collate_fn, shuffle=True,
            pin_memory=args.use_cuda,
            persistent_workers=True,
            num_workers=args.num_workers)

        # Logging
        self.save_dir = Path("trainings/") / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        mkdir_if_needed("trainings/")
        mkdir_if_needed(self.save_dir)

    def train(self):
        for epoch in tqdm(range(self.args.epochs), desc="Training", unit="epoch"):
            self.train_epoch()

            if epoch % 2 == 0:
                self.valid()

            self.writer.flush()

    def train_epoch(self):
        self.net.train()

        for batch in tqdm(self.train_dataloader, desc="Epoch", leave=False, unit="batch"):
            for (k, v) in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.args.device)

            self.optimizer.zero_grad()
            output = self.net(batch["image"])
            loss = self.loss(output, batch)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalars("Loss/Train", self.loss.stats.__dict__, self.global_step)
            self.global_step += self.args.batch_size

        self.writer.add_scalar("Learning rate", self.optimizer.param_groups[0]["lr"], self.global_step)
        self.scheduler.step()
        self.train_set.transform.trigger_random_resize()

    def valid(self):
        self.net.eval()
        self.evaluator.reset()
        loss_stats = LossStats()

        for batch in tqdm(self.valid_dataloader, desc="Validation", leave=False, unit="image"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.args.device)

            with torch.no_grad():
                output = self.net(batch["image"])

            ground_truth = batch["annotation"][0].to_graph()
            ground_truth.resize((args.width, args.height), ground_truth.image_size)

            predicted_graph = self.decoder(output)[0]
            prediction = GraphAnnotation(
                ground_truth.image_path, 
                predicted_graph, 
                ground_truth.image_size)
            prediction.resize((args.width, args.height), prediction.image_size)

            self.evaluator.evaluate(prediction, ground_truth)

            self.loss(output, batch)
            loss_stats += self.loss.stats

        loss_stats /= len(self.valid_dataloader)

        # Compute metrics
        evaluations = self.evaluator.keypoint_evaluation
        total = evaluations.reduce()  # Old total which is a sum (rather than an average)
        kps_prec = {label: eval.precision for label, eval in evaluations.items()}
        kps_prec["total"] = total.precision
        kps_rec = {label: eval.recall for label, eval in evaluations.items()}
        kps_rec["total"] = total.recall
        kps_f1 = {label: eval.f1_score for label, eval in evaluations.items()}
        kps_f1["total"] = total.f1_score

        # Save best network
        f1_kp_reg = total.f1_score
        if loss_stats.total_loss < self.best_loss:
            self.best_loss = loss_stats.total_loss
            self.net.save(self.save_dir / "model_best_loss.pth")
        if f1_kp_reg > self.best_kp_reg:
            self.best_kp_reg = f1_kp_reg
            self.net.save(self.save_dir / "model_best_kp_reg.pth")

        # Draw metrics to Tensorboard
        self.writer.add_scalars("Loss/Validation", loss_stats.__dict__, self.global_step)
        self.writer.add_scalars("Keypoint Evaluation/Precison", kps_prec, self.global_step)
        self.writer.add_scalars("Keypoint Evaluation/Recall", kps_rec, self.global_step)
        self.writer.add_scalars("Keypoint Evaluation/F1", kps_f1, self.global_step)

        # Draw ground truth annotation
        image = batch["image"][0]  # Last image
        image = un_normalize(image)
        image = F.to_pil_image(image)

        graph = predicted_graph  # Last predicted graph

        output = draw_graph(image, graph)
        output = F.to_tensor(output)

        self.writer.add_image("Prediction", output, self.global_step)