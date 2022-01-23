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
    def fp(self) -> int:
        return self.ndet - self.tp

    @property
    def fn(self) -> int:
        return self.npos - self.tp

    @property
    def precision(self) -> float:
        return self.tp / self.ndet if self.ndet != 0 else 1 if self.npos == 0 else 0

    @property
    def recall(self) -> float:
        return self.tp / self.npos if self.npos != 0 else 1 if self.ndet == 0 else 0

    @property
    def f1_score(self) -> float:
        s = self.npos + self.ndet
        return 2 * self.tp / s if s != 0 else 1

    @property
    def avg_acc(self) -> float:
        return np.mean(self.acc) if len(self.acc) != 0 else float("nan")

    @property
    def acc_err(self) -> float:
        return np.std(self.acc) / np.sqrt(len(self.acc)) if len(self.acc) != 0 else float("nan")
        
    def stats(self) -> str:
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

    def __repr__(self) -> str:
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
        self.evals: dict[str, Evaluation] = {label: Evaluation() for label in labels} if labels else {}

    def reset(self):
        for label in self.evals.keys():
            self.evals[label].reset()

    @property
    def labels(self):
        return self.evals.keys()

    def items(self):
        return self.evals.items()

    def __getitem__(self, label: str) -> Evaluation:
        return self.evals[label]

    def __setitem__(self, index: str, item: Evaluation):
        self.evals[index] = item

    def __len__(self) -> int:
        return len(self.evals)

    def __add__(self, other: "Evaluations") -> "Evaluations":
        assert self.labels == other.labels, "The Evaluations should have the same labels"
        evaluations = Evaluations()
        evaluations.evals = {label: self.evals[label] + evaluation 
            for label, evaluation in other.items()}
        return evaluations

    def __iadd__(self, other: "Evaluations") -> "Evaluations":
        assert self.labels == other.labels, "The Evaluations should have the same labels"
        for label, evaluation in other.items():
            self.evals[label] += evaluation
        return self

    def __or__(self, other: "Evaluations") -> "Evaluations":
        output = Evaluations()
        output.evals = {label: self[label] + other[label] 
            for label in self.labels & other.labels}
        output.evals.update({label: self[label] 
            for label in self.labels - other.labels})
        output.evals.update({label: other[label] 
            for label in other.labels - self.labels})
        return output

    def __ior__(self, other: "Evaluations") -> "Evaluations":
        self |= {label: other[label] 
            for label in other.labels - self.labels}
        self |= {label: self[label] + other[label] 
            for label in self.labels & other.labels}
        return self

    def reduce(self) -> Evaluation:
        return reduce(Evaluation.__iadd__, self.evals.values(), Evaluation())

    def pretty_print(self, table_name: str = None):
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
        self.reset()

    def reset(self):
        self.keypoint_evaluation = Evaluations(self.labels)

    def evaluate(self, prediction: Graph, annotation: Graph):
        """Assumes 'prediction' and 'annotation' are in the original input image resolution."""
        self._evaluate_keypoints(prediction, annotation)

    def _evaluate_keypoints(self, prediction: GraphAnnotation, annotation: GraphAnnotation):
        img_size = annotation.image_size
        pred_graph = prediction.graph
        gt_graph = annotation.graph

        dist_thresh = min(img_size) * self.args.dist_threshold
        preds = dict_grouping(pred_graph.keypoints, key=lambda kp: kp.kind)
        gts = dict_grouping(gt_graph.keypoints, key=lambda kp: kp.kind)

        kp_eval = self.keypoint_evaluation

        for label in self.labels:
            res = kp_eval[label]
            preds_label = preds.get(label, [])
            gts_label = gts.get(label, [])

            res.ndet = len(preds_label)
            res.npos = len(gts_label)

            preds_label = sorted(preds_label, 
                key=lambda kp: kp.score, reverse=True)
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

    def pretty_print(self):
        table = Table(Column("Label", style="bold"), *Evaluation.columns(), title="Keypoints Location")
        kp_eval = self.keypoint_evaluation

        for label, evaluation in kp_eval.items():
            table.add_row(label, *evaluation.stats())

        if len(kp_eval) > 1:
            total_eval = kp_eval.reduce()
            table.add_row("Total", *total_eval.stats(), style="bold")
        
        rprint(table)