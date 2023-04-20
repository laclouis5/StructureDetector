from .loss import Loss, LossStats
from .network import Network
from .evaluator import Evaluator
from ..data import *

import torch
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import DataLoader
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
        self.best_csi = 0.0
        self.best_classif = 0.0
        self.best_kp_reg = 0.0

        # TODO: Add MPS support when will be avaiable
        self.autocast_device, self.autocast_dtype = (
            ("cuda", torch.float16) if args.use_cuda else ("cpu", torch.bfloat16)
        )

        # TODO: Test this.
        if args.pretrained_model:
            self.net.load_state_dict(
                torch.load(args.pretrained_model, map_location=args.device)
            )

        self.net.to(args.device)
        self.loss.to(args.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_step
        )

        self.train_set = CropDataset(
            args, args.train_dir, transforms=TrainAugmentation(args)
        )
        self.train_set.localize_image_names()
        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=args.batch_size,
            collate_fn=CropDataset.collate_fn,
            shuffle=True,
            pin_memory=args.use_cuda,
            num_workers=args.num_workers,
            persistent_workers=True,
            drop_last=True,
            multiprocessing_context="forkserver",
        )

        self.valid_set = CropDataset(
            args, args.valid_dir, transforms=ValidationAugmentation(args)
        )
        self.valid_set.localize_image_names()
        self.valid_dataloader = DataLoader(
            self.valid_set,
            batch_size=1,
            collate_fn=CropDataset.collate_fn,
            shuffle=True,
            pin_memory=args.use_cuda,
            num_workers=args.num_workers,
            persistent_workers=True,
            multiprocessing_context="forkserver",
        )

        # Logging
        self.save_dir = Path("trainings") / f"{datetime.now():%Y-%m-%d_%H-%M-%s}"
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

        for batch in tqdm(
            self.train_dataloader, desc="Epoch", leave=False, unit="batch"
        ):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.args.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.autocast_device,
                dtype=self.autocast_dtype,
                enabled=self.args.use_amp,
            ):
                output = self.net(batch["image"])
                loss = self.loss(output, batch)

            loss.backward()
            self.optimizer.step()

            self.writer.add_scalars(
                "Loss/Train", self.loss.stats.__dict__, self.global_step
            )
            self.global_step += self.args.batch_size

        self.writer.add_scalar(
            "Learning rate", self.optimizer.param_groups[0]["lr"], self.global_step
        )
        self.scheduler.step()
        self.train_set.transform.trigger_random_resize()

    def valid(self):
        self.net.eval()
        self.evaluator.reset()
        loss_stats = LossStats()

        with torch.autocast(
            device_type=self.autocast_device,
            dtype=self.autocast_dtype,
            enabled=self.args.use_amp,
        ):
            for batch in tqdm(
                self.valid_dataloader, desc="Validation", leave=False, unit="image"
            ):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.args.device)

                with torch.no_grad():
                    output = self.net(batch["image"])

                data = self.decoder(output, return_metadata=True)
                prediction = data["annotation"][0]
                annotation = batch["annotation"][0]
                self.evaluator.accumulate(
                    prediction,
                    annotation,
                    data["raw_parts"][0],
                    eval_csi=True,
                    eval_classif=True,
                )

                self.loss(output, batch)
                loss_stats += self.loss.stats

        loss_stats /= len(self.valid_dataloader)

        # Compute metrics
        all_anchor_evals = self.evaluator.anchor_eval.reduce()
        anchor_prec = {
            label: eval.precision
            for (label, eval) in self.evaluator.anchor_eval.items()
        }
        anchor_prec["total"] = all_anchor_evals.precision
        anchor_rec = {
            label: eval.recall for (label, eval) in self.evaluator.anchor_eval.items()
        }
        anchor_rec["total"] = all_anchor_evals.recall
        anchor_f1 = {
            label: eval.f1_score for (label, eval) in self.evaluator.anchor_eval.items()
        }
        anchor_f1["total"] = all_anchor_evals.f1_score

        all_part_evals = self.evaluator.part_eval.reduce()
        part_prec = {
            label: eval.precision for (label, eval) in self.evaluator.part_eval.items()
        }
        part_prec["total"] = all_part_evals.precision
        part_rec = {
            label: eval.recall for (label, eval) in self.evaluator.part_eval.items()
        }
        part_rec["total"] = all_part_evals.recall
        part_f1 = {
            label: eval.f1_score for (label, eval) in self.evaluator.part_eval.items()
        }
        part_f1["total"] = all_part_evals.f1_score

        f1_csi = {
            label: eval.f1_score for (label, eval) in self.evaluator.csi_eval.items()
        }
        f1_csi["total"] = self.evaluator.csi_eval.reduce().f1_score

        f1_classif = {
            label: eval.f1_score
            for (label, eval) in self.evaluator.classification_eval.items()
        }
        f1_classif["total"] = self.evaluator.classification_eval.reduce().f1_score

        all_kps = self.evaluator.kps_eval
        all_kps_evals = all_kps.reduce()
        kps_prec = {label: eval.precision for label, eval in all_kps.items()}
        kps_prec["total"] = all_kps_evals.precision
        kps_rec = {label: eval.recall for label, eval in all_kps.items()}
        kps_rec["total"] = all_kps_evals.recall
        kps_f1 = {label: eval.f1_score for label, eval in all_kps.items()}
        kps_f1["total"] = all_kps_evals.f1_score

        f1_kp_reg = all_kps_evals.f1_score

        # Save best network
        if loss_stats.total_loss < self.best_loss:
            self.best_loss = loss_stats.total_loss
            self.net.save(self.save_dir / "model_best_loss.pth")
        if f1_csi["total"] > self.best_csi:
            self.best_csi = f1_csi["total"]
            self.net.save(self.save_dir / "model_best_csi.pth")
        if f1_classif["total"] > self.best_classif:
            self.best_classif = f1_classif["total"]
            self.net.save(self.save_dir / "model_best_classif.pth")
        if f1_kp_reg > self.best_kp_reg:
            self.best_kp_reg = f1_kp_reg
            self.net.save(self.save_dir / "model_best_kp_reg.pth")

        # Draw metrics to Tensorboard
        self.writer.add_scalars(
            "Loss/Validation", loss_stats.__dict__, self.global_step
        )
        self.writer.add_scalars("Metrics_AllKps/Precison", kps_prec, self.global_step)
        self.writer.add_scalars("Metrics_AllKps/Recall", kps_rec, self.global_step)
        self.writer.add_scalars("Metrics_AllKps/F1", kps_f1, self.global_step)
        self.writer.add_scalars(
            "Metrics_Anchor/Precision", anchor_prec, self.global_step
        )
        self.writer.add_scalars("Metrics_Anchor/Recall", anchor_rec, self.global_step)
        self.writer.add_scalars("Metrics_Anchor/f1", anchor_f1, self.global_step)
        self.writer.add_scalars("Metrics_Parts/Precision", part_prec, self.global_step)
        self.writer.add_scalars("Metrics_Parts/Recall", part_rec, self.global_step)
        self.writer.add_scalars("Metrics_Parts/f1", part_f1, self.global_step)
        self.writer.add_scalars("Metrics_CSI/f1", f1_csi, self.global_step)
        self.writer.add_scalars("Metrics_Classif/f1", f1_classif, self.global_step)

        # Draw ground truth annotation
        annotation_img = draw(batch["image"][0], annotation, self.args)
        self.writer.add_image(
            "Detections/Ground_Truth", F.to_tensor(annotation_img), self.global_step
        )

        # Draw network prediction
        prediction_img = draw(batch["image"][0], prediction, self.args)
        self.writer.add_image(
            "Detections/Prediction", F.to_tensor(prediction_img), self.global_step
        )

        # Draw ground truth anchor and part heatmaps
        anchor_hm_img, part_hm_img = draw_heatmaps(
            batch["anchor_hm"][0], batch["part_hm"][0], self.args
        )
        self.writer.add_image(
            "Heatmaps/Ground_Truth/Anchors", anchor_hm_img, self.global_step
        )
        self.writer.add_image(
            "Heatmaps/Ground_Truth/Parts", part_hm_img, self.global_step
        )

        # Draw predicted anchor and part heatmaps
        anchor_hm_img, part_hm_img = draw_heatmaps(
            data["anchor_hm_sig"][0], data["part_hm_sig"][0], self.args
        )
        self.writer.add_image(
            "Heatmaps/Predictions/Anchors", anchor_hm_img, self.global_step
        )
        self.writer.add_image(
            "Heatmaps/Predictions/Parts", part_hm_img, self.global_step
        )

        # Draw raw predictions
        pred_parts = draw_kp_and_emb(
            batch["image"][0],
            data["topk_anchor"],
            data["topk_kp"],
            data["embeddings"],
            self.args,
        )
        self.writer.add_image(
            "Other/Raw_Predictions", F.to_tensor(pred_parts), self.global_step
        )

        # Draw raw embeddings
        raw_embs_img = draw_embeddings(
            batch["image"][0], data["raw_embeddings"], self.args
        )
        self.writer.add_image(
            "Other/Raw_Embeddings", F.to_tensor(raw_embs_img), self.global_step
        )
