from scipy.stats import spearmanr
from sklearn import metrics


def micro_accuracy(preds, labels, desc='') -> dict:
    preds = preds.tolist()
    labels = labels.tolist()
    return {f'val/acc{desc}': metrics.accuracy_score(labels, preds)}


def cls_micro_metrics(preds, labels, desc='') -> dict:
    preds = preds.tolist()
    labels = labels.tolist()
    return {
        f'val/acc{desc}': metrics.accuracy_score(labels, preds),
        f'val/precision{desc}': metrics.precision_score(labels, preds, average='micro', zero_division=0),
        f'val/recall{desc}': metrics.recall_score(labels, preds, average='micro', zero_division=0),
        f'val/f1{desc}': metrics.f1_score(labels, preds, average='micro', zero_division=0)
    }


def cls_macro_metrics(preds, labels, desc='') -> dict:
    preds = preds.tolist()
    labels = labels.tolist()
    return {
        f'val/acc{desc}': metrics.accuracy_score(labels, preds),
        f'val/precision{desc}': metrics.precision_score(labels, preds, average='macro', zero_division=0),
        f'val/recall{desc}': metrics.recall_score(labels, preds, average='macro', zero_division=0),
        f'val/f1{desc}': metrics.f1_score(labels, preds, average='macro', zero_division=0)
    }


def top_k_accuracy_score(preds, labels, k):
    preds = preds.tolist()
    labels = labels.tolist()
    return {f'val/accuracy_top_{k}': metrics.top_k_accuracy_score(labels, preds, k=k)}


def top_k_accuracy(preds, labels, k):
    if not isinstance(preds, list):
        preds = preds.tolist()
        labels = labels.tolist()
    count = 0
    for p, l in zip(preds, labels):
        if l in p[:k]:
            count += 1
    return {f'val/accuracy_top_{k}': count / len(preds)}


def spearman_corr(preds, labels):
    preds = preds.tolist()
    labels = labels.tolist()
    spearman_corr = spearmanr(preds, labels).correlation
    return {'val/spearman_corr': spearman_corr}


def cls_report(preds, labels):
    return metrics.classification_report(labels, preds, zero_division=0)
