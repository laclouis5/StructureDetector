from ..utils import *
import numpy as np
import sys
from functools import reduce


class Evaluation:

    def __init__(self, tp=0, npos=0, ndet=0, acc=0.0):
        self.tp = tp
        self.npos = npos  
        self.ndet = ndet
        self.acc = acc

    def reset(self):
        self.tp = 0
        self.npos = 0  
        self.ndet = 0
        self.acc = 0.0

    def __add__(self, other):
        return Evaluation(
            self.tp + other.tp,
            self.npos + other.npos,
            self.ndet + other.ndet,
            self.acc + other.acc)

    def __iadd__(self, other):
        self.tp += other.tp
        self.npos += other.npos
        self.ndet += other.ndet
        self.acc += other.acc
        return self

    @property
    def fp(self):
        return self.ndet - self.tp

    @property
    def fn(self):
        return self.npos - self.tp

    @property
    def csi(self):
        denominator = self.npos + self.ndet - self.tp
        return self.tp / denominator if denominator != 0 else 1

    @property
    def precision(self):
        return self.tp / self.ndet if self.ndet != 0 else 1 if self.npos == 0 else 0

    @property
    def recall(self):
        return self.tp / self.npos if self.npos != 0 else 1 if self.ndet == 0 else 0

    @property
    def f1_score(self):
        p, r = self.precision, self.recall
        s = p + r
        return 2 * p * r / s if s != 0 else 0

    @property
    def avg_acc(self):
        return self.acc / self.tp if self.tp != 0 else float("nan")

    def __repr__(self):
        return f"f1: {self.f1_score:.2%}, rec: {self.recall:.2%}, prec: {self.precision:.2%}, npos: {self.npos}, ndet: {self.ndet}, tp/fp/fn: {self.tp}/{self.fp}/{self.fn}, avg_acc: {self.avg_acc:.2}"


class Evaluations:

    def __init__(self, labels):
        self.evals = {label: Evaluation() for label in labels}

    def reset(self):
        for label in self.evals.keys():
            self.evals[label].reset()

    @property
    def labels(self):
        return self.evals.keys()

    def items(self):
        return self.evals.items()

    def __getitem__(self, label):
        return self.evals[label]

    def __setitem__(self, index, item):
        self.evals[index] = item

    def __len__(self):
        return len(self.evals)

    def __add__(self, other):
        assert self.labels == other.labels, "The Evaluations should have the same labels"
        evaluations = Evaluations(self.labels)
        evaluations.evals = {label: self.evals[label] + evaluation for (label, evaluation) in other.items()}
        return evaluations

    def __iadd__(self, other):
        assert self.labels == other.labels, "The Evaluations should have the same labels"
        for (label, evaluation) in other.items():
            self.evals[label] += evaluation
        return self

    def reduce(self):
        return reduce(lambda e1, e2: e1 + e2, self.evals.values(), Evaluation())

    def __repr__(self):
        description = ""
        if len(self) > 1:
            description += f"total: {self.reduce()}\n"
        description += "\n".join((f"{label}: {evaluation}" for (label, evaluation) in self.items()))
        return description


class Evaluator:

    def __init__(self, args):
        self.args = args
        self.labels = args.labels.keys()
        self.kp_labels = self.args.parts.keys()

        self.reset()

    def reset(self):
        self.anchor_eval = Evaluations(self.labels)
        self.part_eval = Evaluations(self.kp_labels)
        self.csi_eval = Evaluations(self.labels)
        self.classification_eval = Evaluations(Evaluator.get_classification_labels())

    def accumulate(self, prediction, annotation, part_heatmap=None, eval_csi=False, eval_classif=False):
        self.anchor_eval += self.eval_anchor(prediction, annotation)

        if part_heatmap:
            self.part_eval += self.eval_part(annotation, part_heatmap)
        if eval_csi:
            self.csi_eval += self.eval_csi(prediction, annotation)
        if eval_classif:
            self.classification_eval += self.eval_classif(prediction, annotation)

    def eval_anchor(self, prediction, annotation):
        img_size = annotation.img_size
        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)
        prediction = prediction.resized(
            (self.args.width, self.args.height),
            img_size)

        (img_w, img_h) = img_size
        dist_thresh = min(img_w, img_h) * self.args.dist_threshold

        preds = dict_grouping(prediction.objects, key=lambda obj: obj.name)
        gts = dict_grouping(annotation.objects, key=lambda obj: obj.name)

        result = Evaluations(self.labels)

        for label in self.labels:
            res = result[label]
            preds_label = preds.get(label, [])
            gts_label = gts.get(label, [])

            res.ndet = len(preds_label)
            res.npos = len(gts_label)

            preds_label = sorted(preds_label, key=lambda obj: obj.anchor.score, reverse=True)
            visited = np.repeat(False, len(gts_label))

            for pred in preds_label:
                min_dist = sys.float_info.max
                j_min = None

                for (j, gt) in enumerate(gts_label):
                    dist = pred.distance(gt)
                    if dist < min_dist:
                        min_dist = dist
                        j_min = j

                if min_dist < dist_thresh and not visited[j_min]:
                    visited[j_min] = True
                    res.tp += 1
                    res.acc += min_dist / min(img_w, img_h)

        return result

    def eval_part(self, annotation, part_heatmap):
        img_size = annotation.img_size

        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)

        part_heatmap = (kp.resized((self.args.width, self.args.height), img_size) for kp in part_heatmap)

        (img_w, img_h) = img_size
        dist_thresh = min(img_w, img_h) * self.args.dist_threshold

        preds = dict_grouping(part_heatmap, key=lambda kp: kp.kind)
        gts = (kp for obj in annotation.objects for kp in obj.parts)
        gts = dict_grouping(gts, key=lambda kp: kp.kind)

        kp_result = Evaluations(self.kp_labels)

        for kp_label in self.kp_labels:
            res_kp = kp_result[kp_label]

            preds_kp_label = preds.get(kp_label, [])
            gts_kp_label = gts.get(kp_label, [])

            res_kp.ndet += len(preds_kp_label)
            res_kp.npos += len(gts_kp_label)

            preds_kp_label = sorted(preds_kp_label,
                key=lambda kp: kp.score,
                reverse=True)
            visited_kp = np.repeat(False, len(gts_kp_label))

            for pred_kp in preds_kp_label:
                min_dist_kp = sys.float_info.max
                j_min_kp = None

                for (l, gt_kp) in enumerate(gts_kp_label):
                    dist_kp = pred_kp.distance(gt_kp)

                    if dist_kp < min_dist_kp:
                        min_dist_kp = dist_kp
                        j_min_kp = l

                if min_dist_kp < dist_thresh and not visited_kp[j_min_kp]:
                    visited_kp[j_min_kp] = True
                    res_kp.tp += 1
                    res_kp.acc += min_dist_kp / min(img_w, img_h)

        return kp_result

    def eval_csi(self, prediction, annotation):
        img_size = annotation.img_size
        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)
        prediction = prediction.resized(
            (self.args.width, self.args.height),
            img_size)

        (img_w, img_h) = img_size
        dist_thresh = min(img_w, img_h) * self.args.dist_threshold

        preds = dict_grouping(prediction.objects, key=lambda obj: obj.name)
        gts = dict_grouping(annotation.objects, key=lambda obj: obj.name)

        result = Evaluations(self.labels)

        for label in self.labels:
            res = result[label]
            preds_label = preds.get(label, [])
            gts_label = gts.get(label, [])

            res.ndet = len(preds_label)
            res.npos = len(gts_label)

            preds_label = sorted(preds_label, key=lambda obj: obj.anchor.score, reverse=True)
            visited = np.repeat(False, len(gts_label))

            for pred in preds_label:
                best_csi = 0.0
                idx_best = None

                for (j, gt) in enumerate(gts_label):
                    csi = Evaluator.compute_csi(pred, gt, dist_thresh)
                    if csi > best_csi:
                        best_csi = csi
                        idx_best = j

                if best_csi >= self.args.csi_threshold and not visited[idx_best]:
                    visited[idx_best] = True
                    res.tp += 1
                    res.acc += best_csi

        return result

    @staticmethod
    def get_classification_labels():
        """WARNING: Hardcoded"""
        labels = [f"maize_{index}" for index in range(10)]
        labels += [f"bean_{index}" for index in range(10)]
        return labels

    def eval_classif(self, prediction, annotation):
        img_size = annotation.img_size
        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)
        prediction = prediction.resized(
            (self.args.width, self.args.height),
            img_size)

        (img_w, img_h) = img_size
        dist_thresh = min(img_w, img_h) * self.args.dist_threshold

        preds = dict_grouping(prediction.objects, key=lambda obj: f"{obj.name}_{obj.nb_parts}")
        gts = dict_grouping(annotation.objects, key=lambda obj: f"{obj.name}_{obj.nb_parts}")
        labels = Evaluator.get_classification_labels()
        result = Evaluations(labels)

        for label in labels:
            res = result[label]
            preds_label = preds.get(label, [])
            gts_label = gts.get(label, [])

            res.ndet = len(preds_label)
            res.npos = len(gts_label)

            preds_label = sorted(preds_label, key=lambda obj: obj.anchor.score, reverse=True)
            visited = np.repeat(False, len(gts_label))

            for pred in preds_label:
                best_dist = sys.float_info.max
                idx_best = None

                for (i, gt) in enumerate(gts_label):
                    dist = pred.distance(gt)
                    if dist < best_dist:
                        best_dist = dist
                        idx_best = i

                if best_dist <= dist_thresh and not visited[idx_best]:
                    visited[idx_best] = True
                    res.tp += 1
                    res.acc += best_dist / min(img_w, img_h)

        return result

    @staticmethod
    def compute_csi(prediction, target, dist_thresh): 
        preds_kp = dict_grouping(prediction.parts, key=lambda kp: kp.kind)
        gts_kp = dict_grouping(target.parts, key=lambda kp: kp.kind)

        if prediction.name != target.name: return 0.0

        evaluation = Evaluation()
        evaluation.npos += 1
        evaluation.ndet += 1

        evaluation.tp += prediction.distance(target) < dist_thresh and prediction.name == target.name

        for kp_label in gts_kp.keys() | preds_kp.keys():
            preds_kp_label = preds_kp.get(kp_label, [])
            gts_kp_label = gts_kp.get(kp_label, [])

            evaluation.npos += len(gts_kp_label)
            evaluation.ndet += len(preds_kp_label)

            preds_kp_label = sorted(preds_kp_label,
                key=lambda kp: kp.score,
                reverse=True)
            visited_kp = np.repeat(False, len(gts_kp_label))

            for pred_kp in preds_kp_label:
                min_dist_kp = sys.float_info.max
                j_min_kp = None

                for (j, gt_kp) in enumerate(gts_kp_label):
                    dist_kp = pred_kp.distance(gt_kp)

                    if dist_kp < min_dist_kp:
                        min_dist_kp = dist_kp
                        j_min_kp = j

                if min_dist_kp < dist_thresh and not visited_kp[j_min_kp]:
                    visited_kp[j_min_kp] = True
                    evaluation.tp += 1

        return evaluation.csi

    def __repr__(self):
        results = {
            "Anchor Location": self.anchor_eval, "Part Location": self.part_eval, 
            "CSI": self.csi_eval, "Classification": self.classification_eval}

        description = ""
        for (metric_name, evaluations) in results.items():
            description += f"{metric_name}\n"
            if len(evaluations) > 1:
                description += f"  total: {evaluations.reduce()}\n"
            evaluations = sorted(evaluations.items(), key=lambda tuple: tuple[0])
            for (label, evaluation) in evaluations:
                description += f"  {label}: {evaluation}\n"

        return description