from ..utils import *
import numpy as np
import sys
from functools import reduce
from rich import print as rprint
from rich.table import Table, Column
from copy import copy


class Evaluation:

    def __init__(self, tp=0, npos=0, ndet=0, acc=None, counts=None):
        Evaluation._precondition(tp, npos, ndet)

        self.tp = tp
        self.npos = npos  
        self.ndet = ndet
        self.acc = acc or []
        self.count_errors = counts or []

    def reset(self):
        self.__init__()

    def __iadd__(self, other):
        self.tp += other.tp
        self.npos += other.npos
        self.ndet += other.ndet
        self.acc += other.acc
        self.count_errors += other.count_errors
        return self

    def __add__(self, other):
        copy_ = copy(self)
        copy_ += other
        return copy_

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
        s = self.npos + self.ndet
        return 2 * self.tp / s if s != 0 else 1

    @property
    def avg_acc(self):
        return np.mean(self.acc) if len(self.acc) != 0 else float("nan")

    @property
    def acc_err(self):
        return np.std(self.acc) / np.sqrt(len(self.acc)) if len(self.acc) != 0 else float("nan")
        
    def stats(self):
        return f"{self.npos}", f"{self.ndet}", f"{self.recall:.2%}", f"{self.precision:.2%}", f"{self.f1_score:.2%}", f"{self.avg_acc:.4%}", f"{self.acc_err:.4%}"

    @staticmethod
    def columns():
        return (
            Column("Gts.", justify="right"), 
            Column("Preds.", justify="right"), 
            Column("Rec.", justify="right"), 
            Column("Prec.", justify="right"), 
            Column("F1 Score", justify="right", style="green"), 
            Column("L. Acc.", justify="right"), 
            Column("L. Err.", justify="right"))

    def __repr__(self):
        return f"f1: {self.f1_score:.2%}, rec: {self.recall:.2%}, prec: {self.precision:.2%}, npos: {self.npos}, ndet: {self.ndet}, tp/fp/fn: {self.tp}/{self.fp}/{self.fn}, avg_acc: {self.avg_acc:.2}"

    def pretty_print(self):
        table = Table(*Evaluation.columns())
        table.add_row(*self.stats())
        rprint(table)

    def save_conf_matrix(self):
        c_err_by_label = dict_grouping(self.count_errors, lambda t: t[0])
        for label, count_errors in c_err_by_label.items():
            conf_mat = np.zeros((10, 10))
            for _, p, e in count_errors:
                conf_mat[e, p] += 1
            np.save(f"conf_mat_{label}.npy", conf_mat)

    @staticmethod
    def _precondition(tp, npos, ndet):
        assert tp >= 0 and ndet >= 0 and npos >= 0, "tp, npos and ndet should be positive"
        assert tp <= ndet, "tp must be lower than or equal to ndet"
        assert tp <= npos, "tp must be lower than or equal to npos"


class Evaluations:

    def __init__(self, labels=None):
        self.evals = {label: Evaluation() for label in labels} if labels else {}

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
        evaluations = Evaluations()
        evaluations.evals = {label: self.evals[label] + evaluation 
            for label, evaluation in other.items()}
        return evaluations

    def __iadd__(self, other):
        assert self.labels == other.labels, "The Evaluations should have the same labels"
        for label, evaluation in other.items():
            self.evals[label] += evaluation
        return self

    def __or__(self, other):
        output = Evaluations()
        output.evals = {label: self[label] + other[label] 
            for label in self.labels & other.labels}
        output.evals.update({label: self[label] 
            for label in self.labels - other.labels})
        output.evals.update({label: other[label] 
            for label in other.labels - self.labels})
        return output

    def __ior__(self, other):
        self |= {label: other[label] 
            for label in other.labels - self.labels}
        self |= {label: self[label] + other[label] 
            for label in self.labels & other.labels}
        return self

    def reduce(self):
        return reduce(Evaluation.__iadd__, self.evals.values(), Evaluation())

    def pretty_print(self, table_name=None):
        table = Table("Label", *Evaluation.columns(), title=table_name)
        for label, evaluation in self.items():
            table.add_row(label, *evaluation.stats())
        if len(self) > 1:
            table.add_row("Total", *self.reduce().stats(), style="bold")
        rprint(table)

    def __repr__(self):
        description = ""
        if len(self) > 1:
            description += f"total: {self.reduce()}\n"
        description += "\n".join(
            f"{label}: {evaluation}" for (label, evaluation) in self.items())
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

    @property
    def kps_eval(self):
        return self.anchor_eval | self.part_eval

    def accumulate(self, prediction, annotation, part_heatmap=None, eval_csi=False, eval_classif=False):
        self.anchor_eval += self.eval_anchor(prediction, annotation)

        if part_heatmap is not None:
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

        dist_thresh = min(img_size) * self.args.dist_threshold
        preds = dict_grouping(prediction.objects, key=lambda obj: obj.name)
        gts = dict_grouping(annotation.objects, key=lambda obj: obj.name)

        result = Evaluations(self.labels)

        for label in self.labels:
            res = result[label]
            preds_label = preds.get(label, [])
            gts_label = gts.get(label, [])

            res.ndet = len(preds_label)
            res.npos = len(gts_label)

            preds_label = sorted(preds_label, 
                key=lambda obj: obj.anchor.score, reverse=True)
            visited = np.repeat(False, len(gts_label))

            for pred in preds_label:
                min_dist = sys.float_info.max
                j_min = None

                for j, gt in enumerate(gts_label):
                    dist = pred.distance(gt)
                    if dist < min_dist:
                        min_dist = dist
                        j_min = j

                if min_dist < dist_thresh and not visited[j_min]:
                    visited[j_min] = True
                    res.tp += 1
                    res.acc.append(min_dist / min(img_size))

        return result

    def eval_part(self, annotation, part_heatmap):
        img_size = annotation.img_size

        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)

        part_heatmap = (kp.resized((self.args.width, self.args.height), img_size) 
            for kp in part_heatmap)

        dist_thresh = min(img_size) * self.args.dist_threshold

        preds = dict_grouping(part_heatmap, key=lambda kp: kp.kind)
        gts = (kp for obj in annotation.objects for kp in obj.parts)
        gts = dict_grouping(gts, key=lambda kp: kp.kind)

        kp_result = Evaluations(self.kp_labels)

        for kp_label in self.kp_labels:
            res_kp = kp_result[kp_label]

            preds_kp_label = preds.get(kp_label, [])
            gts_kp_label = gts.get(kp_label, [])

            res_kp.ndet = len(preds_kp_label)
            res_kp.npos = len(gts_kp_label)

            preds_kp_label = sorted(preds_kp_label,
                key=lambda kp: kp.score,
                reverse=True)
            visited_kp = np.repeat(False, len(gts_kp_label))

            for pred_kp in preds_kp_label:
                min_dist_kp = sys.float_info.max
                j_min_kp = None

                for l, gt_kp in enumerate(gts_kp_label):
                    dist_kp = pred_kp.distance(gt_kp)

                    if dist_kp < min_dist_kp:
                        min_dist_kp = dist_kp
                        j_min_kp = l

                if min_dist_kp < dist_thresh and not visited_kp[j_min_kp]:
                    visited_kp[j_min_kp] = True
                    res_kp.tp += 1
                    res_kp.acc.append(min_dist_kp / min(img_size))

        return kp_result

    def eval_csi(self, prediction, annotation):
        img_size = annotation.img_size
        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)
        prediction = prediction.resized(
            (self.args.width, self.args.height),
            img_size)

        dist_thresh = min(img_size) * self.args.dist_threshold

        preds = dict_grouping(prediction.objects, key=lambda obj: obj.name)
        gts = dict_grouping(annotation.objects, key=lambda obj: obj.name)

        result = Evaluations(self.labels)

        for label in self.labels:
            res = result[label]
            preds_label = preds.get(label, [])
            gts_label = gts.get(label, [])

            res.ndet = len(preds_label)
            res.npos = len(gts_label)

            preds_label = sorted(preds_label,
                key=lambda obj: obj.anchor.score, reverse=True)
            visited = np.repeat(False, len(gts_label))

            for pred in preds_label:
                best_csi = 0.0
                idx_best = None

                for j, gt in enumerate(gts_label):
                    csi = Evaluator.compute_csi(pred, gt, dist_thresh)
                    if csi > best_csi:
                        best_csi = csi
                        idx_best = j

                if best_csi >= self.args.csi_threshold and not visited[idx_best]:
                    visited[idx_best] = True
                    res.tp += 1
                    res.acc.append(best_csi)

        return result

    @staticmethod
    def get_classification_labels():
        """WARNING: Hardcoded"""
        return [f"bean_{index}" for index in range(10)] + [f"maize_{index}" for index in range(10)]

    def eval_classif(self, prediction, annotation):
        img_size = annotation.img_size
        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)
        prediction = prediction.resized(
            (self.args.width, self.args.height),
            img_size)

        img_w, img_h = img_size
        dist_thresh = min(img_w, img_h) * self.args.dist_threshold

        preds = dict_grouping(prediction.objects, 
            key=lambda obj: f"{obj.name}_{obj.nb_parts}")
        gts = dict_grouping(annotation.objects, 
            key=lambda obj: f"{obj.name}_{obj.nb_parts}")
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

                for i, gt in enumerate(gts_label):
                    dist = pred.distance(gt)
                    if dist < best_dist:
                        best_dist = dist
                        idx_best = i

                if best_dist <= dist_thresh and not visited[idx_best]:
                    visited[idx_best] = True
                    res.tp += 1
                    res.acc.append(best_dist / min(img_w, img_h))

        return result

    def eval_classif_2(self, prediction, annotation):
        img_size = annotation.img_size
        annotation = annotation.resized(
            (self.args.width, self.args.height),
            img_size)
        prediction = prediction.resized(
            (self.args.width, self.args.height),
            img_size)

        dist_thresh = min(img_size) * self.args.dist_threshold

        preds = dict_grouping(prediction.objects, key=lambda obj: f"{obj.name}_{obj.nb_parts}")
        _gts = dict_grouping(annotation.objects, key=lambda obj: f"{obj.name}_{obj.nb_parts}")
        gts = annotation.objects
        visited = [False] * len(gts)
        labels = Evaluator.get_classification_labels()
        result = Evaluations(labels)

        for label in labels:
            res = result[label]
            preds_label = preds.get(label, [])
            gts_label = _gts.get(label, [])

            res.ndet = len(preds_label)
            res.npos = len(gts_label)

            preds_label = sorted(preds_label, key=lambda obj: obj.anchor.score, reverse=True)

            for pred in preds_label:
                best_dist = sys.float_info.max
                idx_best = None

                for i, gt in enumerate(gts):
                    dist = pred.distance(gt)
                    if dist < best_dist:
                        best_dist = dist
                        idx_best = i

                if best_dist > dist_thresh or \
                    visited[idx_best] or \
                    (pred.name not in gts[idx_best].name):
                    continue
                    
                # Same label from here
                label = pred.name
                if pred.nb_parts != gts[idx_best].nb_parts:
                    res.count_errors.append((label, pred.nb_parts, gts[idx_best].nb_parts))
                    continue

                visited[idx_best] = True
                res.tp += 1
                res.acc.append(best_dist / min(img_size))
                res.count_errors.append((label, pred.nb_parts, gts[idx_best].nb_parts))

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

                for j, gt_kp in enumerate(gts_kp_label):
                    dist_kp = pred_kp.distance(gt_kp)

                    if dist_kp < min_dist_kp:
                        min_dist_kp = dist_kp
                        j_min_kp = j

                if min_dist_kp < dist_thresh and not visited_kp[j_min_kp]:
                    visited_kp[j_min_kp] = True
                    evaluation.tp += 1

        return evaluation.csi

    def pretty_print(self):
        results = {
            "Anchor Location": self.anchor_eval, "Part Location": self.part_eval, 
            "All Kps Location": self.kps_eval, "CSI": self.csi_eval, 
            "Classification": self.classification_eval}

        for title, evals in results.items():
            table = Table(Column("Label", style="bold"), *Evaluation.columns(), title=title)

            for label, evaluation in evals.items():
                table.add_row(label, *evaluation.stats())

            if len(evals) > 1:
                total_eval = evals.reduce()
                table.add_row("Total", *total_eval.stats(), style="bold")
            
            rprint(table)

    def __repr__(self):
        results = {
            "Anchor Location": self.anchor_eval, "Part Location": self.part_eval,
            "All Kps Location": self.kps_eval, "CSI": self.csi_eval, 
            "Classification": self.classification_eval}

        description = ""
        for (metric_name, evaluations) in results.items():
            description += f"{metric_name}\n"
            if len(evaluations) > 1:
                description += f"  total: {evaluations.reduce()}\n"
            evaluations = sorted(evaluations.items(), key=lambda tuple: tuple[0])
            for (label, evaluation) in evaluations:
                description += f"  {label}: {evaluation}\n"

        return description