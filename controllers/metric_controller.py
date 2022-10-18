from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2
from utils.conlleval import evaluate


def compute_F1(prec, rec):
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

def metric_fn(items, config):
    metrics = [item["metrics"] for item in items]
    if config["display_metric"] == "accuracy":
        correct_predictions = sum([metric["correct_predictions"] for metric in metrics])
        total = sum([metric["total"] for metric in metrics])
        accuracy = correct_predictions/total if total > 0 else 0
        loss = sum([metric["loss"] for metric in metrics])/len(metrics) if len(metrics) > 0 else 0

        composed_metric = {"loss": loss,
                           "accuracy": accuracy*100}

    elif config["display_metric"] == "F1" and config["model_type"] == "seq_label":
        loss = sum([metric["loss"] for metric in metrics]) / len(metrics) if len(metrics) > 0 else 0
        display_items = [item["display_items"] for item in items]
        all_predictions = []
        all_labels = []
        for item in display_items:
            all_predictions += item["predictions"]
            all_labels += item["labels"]

        composed_metric = {"loss": loss,
                           "F1": seqeval_f1_score(all_labels, all_predictions, scheme=IOB2)}
    elif config["model_type"] == "seq2seq" or config["model_type"] == "seq2set":
        composed_metric = {}
        total_data = sum([metric["total_data"] for metric in metrics])
        for beam in [False]:
            if beam:
                beam_tag = "_beam"
            else:
                beam_tag = ""

            if "total_present_precision_beam" not in metrics[0] and beam:
                continue

            if isinstance(metrics[0]["total_present_precision"], int):
                continue

            for topk in metrics[0]["total_present_precision"]:
                total_present_precision = sum(
                    [metric["total_present_precision" + beam_tag][topk] for metric in metrics])
                total_present_recall = sum([metric["total_present_recall" + beam_tag][topk] for metric in metrics])
                avg_present_precision = total_present_precision / total_data
                avg_present_recall = total_present_recall / total_data
                macro_present_F1 = compute_F1(avg_present_precision, avg_present_recall)

                composed_metric["present_precision_" + topk + beam_tag] = avg_present_precision
                composed_metric["present_recall_" + topk + beam_tag] = avg_present_recall
                composed_metric["macro_present_F1_" + topk + beam_tag] = macro_present_F1

                total_absent_precision = sum([metric["total_absent_precision" + beam_tag][topk] for metric in metrics])
                total_absent_recall = sum([metric["total_absent_recall" + beam_tag][topk] for metric in metrics])
                avg_absent_precision = total_absent_precision / total_data
                avg_absent_recall = total_absent_recall / total_data
                macro_absent_F1 = compute_F1(avg_absent_precision, avg_absent_recall)

                composed_metric["absent_precision_" + topk + beam_tag] = avg_absent_precision
                composed_metric["absent_recall_" + topk + beam_tag] = avg_absent_recall
                composed_metric["macro_absent_F1_" + topk + beam_tag] = macro_absent_F1

        loss = sum([metric["loss"] * metric["total_data"] for metric in metrics]) / total_data
        composed_metric["loss"] = loss

    return composed_metric


def compose_dev_metric(metrics, config):
    total_metric = 0
    n = len(metrics)
    for key in metrics:
        total_metric += metrics[key][config["save_by"]]
    return config["metric_direction"] * total_metric / n
