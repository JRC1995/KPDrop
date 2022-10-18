import random


def step_display_fn(epoch, iter, item, args, config):
    display_string = "Model: {}, Dataset: {}, Current LR: {}, Epoch {}, Step: {}, Loss: {:.3f}, {}: {:.3f}".format(
        config["model_name"],
        args.dataset,
        config["current_lr"],
        epoch,
        iter,
        item["metrics"]["loss"],
        config["display_metric"],
        item["metrics"][config["display_metric"]])

    return display_string


def example_display_fn(epoch, iter, item, args, config):
    idx2labels = config["idx2labels"]
    item_len = len(item["display_items"]["predictions"])
    chosen_id = random.choice([id for id in range(item_len)])

    if args.model_type == "sentence_pair":
        display_string = "Example:\nSequence1: {}\nSequence2: {}\nPrediction: {}\nGround Truth: {}\n".format(
            " ".join(item["display_items"]["sequences1"][chosen_id]),
            " ".join(item["display_items"]["sequences2"][chosen_id]),
            idx2labels[item["display_items"]["predictions"][chosen_id]],
            idx2labels[item["display_items"]["labels"][chosen_id]])
    elif args.model_type == "classifier":
        display_string = "Example:\nSequence: {}\nPrediction: {}\nGround Truth: {}\n".format(
            " ".join(item["display_items"]["sequences"][chosen_id]),
            idx2labels[item["display_items"]["predictions"][chosen_id]],
            idx2labels[item["display_items"]["labels"][chosen_id]])
    elif args.model_type == "seq_label":
        label = " ".join(item["display_items"]["labels"][chosen_id])
        seq_label_len = len(item["display_items"]["labels"][chosen_id])
        prediction = item["display_items"]["predictions"][chosen_id]
        assert len(prediction) == seq_label_len
        prediction = " ".join(prediction)

        display_string = "Example:\nSequence: {}\nPrediction: {}\nGround Truth: {}\n".format(
            " ".join(item["display_items"]["sequences"][chosen_id]),
            prediction, label)
    elif args.model_type == "seq2seq" or args.model_type == "seq2set":
        item_len = len(item["display_items"]["predictions"])
        chosen_id = random.choice([id for id in range(item_len)])
        display_string = "Example:\nSource: {}\nTarget: {}\nPrediction: {}\n".format(
            " ".join(item["display_items"]["source"][chosen_id]),
            " ".join(item["display_items"]["target"][chosen_id]),
            item["display_items"]["predictions"][chosen_id])

    return display_string


def display(display_string, log_paths):
    with open(log_paths["log_path"], "a") as fp:
        fp.write(display_string)
    with open(log_paths["verbose_log_path"], "a") as fp:
        fp.write(display_string)
    print(display_string)
