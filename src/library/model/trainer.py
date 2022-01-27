from .loss import Loss, LossStats
from .network import Network, NetworkRegNet
from .evaluator import Evaluator
from ..data import *

import torch
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

        # Logging
        log_dir = args.log_dir or f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        self.save_dir = Path("trainings/") / log_dir
        mkdir_if_needed("trainings/")
        mkdir_if_needed(self.save_dir)

        self.writer = SummaryWriter(Path("runs/") / log_dir)
        self.global_step = 0
        self.best_loss = torch.finfo().max
        self.best_kp_reg = 0.0

        if args.pretrained_model:
            self.net.load_state_dict(torch.load(args.pretrained_model, map_location=args.device))

        self.net.to(args.device)
        self.loss.to(args.device)

        # TODO: Add Apex mixed precision computing for faster training and inference

        self.optimizer = torch.optim.AdamW(self.net.parameters(), args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step)

        self.train_set = TrainDataset(args.train_dir, transforms=TrainAugmentation(args))
        self.train_dataloader = data.DataLoader(self.train_set,
            batch_size=args.batch_size, shuffle=True,
            pin_memory=args.use_cuda,
            num_workers=args.num_workers, 
            persistent_workers=True,
            drop_last=True)  # <- Remove this?

        self.valid_set = ValidDataset(args.valid_dir, transforms=ValidationAugmentation(args))
        self.valid_dataloader = data.DataLoader(self.valid_set,
            batch_size=1, collate_fn=ValidDataset.collate_fn, shuffle=True,
            pin_memory=args.use_cuda,
            persistent_workers=True,
            num_workers=args.num_workers)

    def train(self):
        for epoch in tqdm(range(self.args.epochs), desc="Training", unit="epoch"):
            self.train_epoch()

            if epoch % 2 == 0:
                self.valid()

            self.writer.flush()

    def train_epoch(self):
        self.net.train()

        for batch in tqdm(self.train_dataloader, desc="Epoch", leave=False, unit="batch"):
            for k, v in batch.items():
                batch[k] = v.to(self.args.device)

            self.optimizer.zero_grad()
            output = self.net(batch["image"])
            loss = self.loss(output, batch)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalars("Loss/Training", self.loss.stats.__dict__, self.global_step)
            self.global_step += self.args.batch_size

        self.scheduler.step()
        self.train_set.transform.trigger_random_resize()
        self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], self.global_step)

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

            annotation = batch["annotation"][0].to_graph()
            ground_truth = annotation.resized((self.args.width, self.args.height), annotation.image_size)

            predicted_graph = self.decoder(output)[0]
            prediction = GraphAnnotation(
                ground_truth.image_path, 
                predicted_graph, 
                ground_truth.image_size)
            prediction = prediction.resized((self.args.width, self.args.height), prediction.image_size)

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

        # Draw predicted graph
        image = batch["image"][0]  # Last image
        image = un_normalize(image)
        image = F.to_pil_image(image)

        graph = predicted_graph  # Last predicted graph

        graph_im = draw_graph(image, graph)
        graph_im = F.to_tensor(graph_im)

        self.writer.add_image("Graph Prediction/Predicted Graph", graph_im, self.global_step)

        # Draw ground truth graph
        gt_graph_im = draw_graph(image, annotation.graph)
        gt_graph_im = F.to_tensor(gt_graph_im)

        self.writer.add_image("Graph Prediction/Ground Truth Graph", gt_graph_im, self.global_step)

        # Heatmaps
        heatmaps = output["heatmaps"][0]
        heatmaps_im = draw_heatmaps(heatmaps, self.args)
        heatmaps_im = F.to_tensor(heatmaps_im)
        self.writer.add_image("Heatmaps/Predicted Heatmaps", heatmaps_im, self.global_step)

        heatmaps = batch["heatmaps"][0]
        heatmaps_im = draw_heatmaps(heatmaps, self.args)
        heatmaps_im = F.to_tensor(heatmaps_im)
        self.writer.add_image("Heatmaps/Ground Truth Heatmaps", heatmaps_im, self.global_step)