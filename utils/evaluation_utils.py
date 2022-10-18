import random

import nltk
from nltk.stem import PorterStemmer
import copy


def process_predictions(predictions, beam=False):
    stemmer = PorterStemmer()

    processed_predictions = []
    for beam_prediction in predictions:
        if beam:
            prediction_ = ""
            for prediction in beam_prediction:
                prediction = prediction.replace(";", "<sep>")
                prediction = prediction.split("<eos>")[0]
                if not prediction_:
                    prediction_ += prediction
                else:
                    prediction_ += ' <sep> ' + prediction
            prediction = prediction_
        else:
            beam_prediction = beam_prediction.replace(";", "<sep>")
            prediction = beam_prediction.split("<eos>")[0]

        prediction = prediction.split("<sep>")

        stemed_prediction = []
        for kp in prediction:
            kp = kp.lower().strip()
            if kp != "" and kp != "<peos>" and kp!="," and kp != "." and kp != "<unk>":  # and "." not in kp and "," not in kp
                tokenized_kp = kp.split(" ")  # nltk.word_tokenize(kp)
                tokenized_stemed_kp = [stemmer.stem(kw).strip() for kw in tokenized_kp]
                stemed_kp = " ".join(tokenized_stemed_kp).replace("< digit >", "<digit>")
                if stemed_kp.strip() != "":
                    stemed_prediction.append(stemed_kp.strip())

        # make prediction duplicates free but preserve order for @topk

        prediction_dict = {}
        stemed_prediction_ = []
        for kp in stemed_prediction:
            if kp not in prediction_dict:
                prediction_dict[kp] = 1
                stemed_prediction_.append(kp)
        stemed_prediction = stemed_prediction_

        processed_predictions.append(stemed_prediction)

    return processed_predictions


def process_ground_truths(trgs, key="none"):
    stemmer = PorterStemmer()
    processed_trgs = []
    for trg in trgs:
        trg = " ".join(trg)
        trg = trg.replace(";", "<sep>")
        trg_split = trg.split("<eos>")[0]
        trg_split = trg_split.split(" <sep> ")
        stemed_trg = []
        for kp in trg_split:
            kp = kp.lower().strip()
            if kp.strip() != "<peos>":
                tokenized_kp = kp.split(" ")
                if key != "semeval":
                    tokenized_stemed_kp = [stemmer.stem(kw).strip() for kw in tokenized_kp]
                else:
                    tokenized_stemed_kp = [kw.lower().strip() for kw in tokenized_kp]
                stemed_kp = " ".join(tokenized_stemed_kp).replace("< digit >", "<digit>")
                if stemed_kp.strip() != "":
                    stemed_trg.append(stemed_kp.strip())
        stemed_trg = list(set(stemed_trg))
        processed_trgs.append(stemed_trg)

    return processed_trgs


def process_srcs(srcs):
    stemmer = PorterStemmer()
    processed_srcs = []
    for src in srcs:
        tokenized_src = src
        tokenized_stemed_src = [stemmer.stem(token.strip().lower()).strip() for token in tokenized_src]
        stemed_src = " ".join(tokenized_stemed_src).strip().replace("< digit >", "<digit>")
        processed_srcs.append(stemed_src)
    return processed_srcs


def evaluate(srcs, trgs, predictions, beam=False, key="none"):
    assert len(srcs) == len(trgs)
    assert len(predictions) == len(trgs)

    chosen_id = random.choice([i for i in range(len(predictions))])

    # print("Before processing Predictions: ", predictions[chosen_id])
    predictions = process_predictions(predictions, beam)
    # print("After processing Predictions: ", predictions[chosen_id])

    # print("Before processing trgs: ", trgs[chosen_id])
    trgs = process_ground_truths(trgs, key)
    # print("After processing trgs: ", trgs[chosen_id])
    # print("Before processing srcs: ", srcs[chosen_id])
    srcs = process_srcs(srcs)
    # print("After processing srcs: ", srcs[chosen_id])

    total_present_precision = {}
    total_present_recall = {}
    total_absent_precision = {}
    total_absent_recall = {}

    i = 0
    total_data = 0
    for src, trg, prediction in zip(srcs, trgs, predictions):

        present_trg = []
        absent_trg = []
        for kp in trg:
            if kp in src:
                present_trg.append(kp)
            else:
                absent_trg.append(kp)

        present_prediction = []
        absent_prediction = []
        for kp in prediction:
            if kp in src:
                present_prediction.append(kp)
            else:
                absent_prediction.append(kp)

        present_precision = {}
        present_recall = {}

        absent_precision = {}
        absent_recall = {}

        """
        if i==chosen_id:
            print("present_trg: ", present_trg)
            print("absent_trg: ", absent_trg) 
        """

        # ["5", "10", "50", "5R", "10R", "50R", "M", "O"]

        original_present_prediction = copy.deepcopy(present_prediction)
        original_absent_prediction = copy.deepcopy(absent_prediction)

        for topk in ["5R", "M", "5", "10", "50"]:
            present_prediction = copy.deepcopy(original_present_prediction)
            absent_prediction = copy.deepcopy(original_absent_prediction)
            if topk == "M":
                pass
            elif topk == "O":
                if not present_trg:
                    present_prediction = []
                elif len(present_prediction) > len(present_trg):
                    present_prediction = present_prediction[0:len(present_trg)]
                if not absent_trg:
                    absent_prediction = []
                elif len(absent_prediction) > len(absent_trg):
                    absent_prediction = absent_prediction[0:len(absent_trg)]
            else:
                if "R" in topk:
                    R = True
                    topk = int(topk[0:-1])
                else:
                    R = False
                    topk = int(topk)
                if len(present_prediction) > topk:
                    present_prediction = present_prediction[0:topk]
                elif R:
                    while len(present_prediction) < topk:
                        present_prediction.append("<fake keyphrase>")
                if len(absent_prediction) > topk:
                    absent_prediction = absent_prediction[0:topk]
                elif R:
                    while len(absent_prediction) < topk:
                        absent_prediction.append("<fake keyphrase>")
                topk = str(topk)
                if R:
                    topk = topk + "R"

            """
            if i == chosen_id:
                print("{} present prediction: ".format(topk), present_prediction)
                print("{} absent prediction: ".format(topk), absent_prediction)
            """

            present_tp = 0
            for kp in present_prediction:
                if kp in present_trg:
                    present_tp += 1

            present_precision[topk] = present_tp / len(present_prediction) if len(present_prediction) > 0 else 0
            present_recall[topk] = present_tp / len(present_trg) if len(present_trg) > 0 else 0

            absent_tp = 0
            for kp in absent_prediction:
                if kp in absent_trg:
                    absent_tp += 1

            absent_precision[topk] = absent_tp / len(absent_prediction) if len(absent_prediction) > 0 else 0
            absent_recall[topk] = absent_tp / len(absent_trg) if len(absent_trg) > 0 else 0

            """
            if i == chosen_id:
                print("{} present precision: ".format(topk), present_precision[topk])
                print("{} present recall: ".format(topk), present_recall[topk])
                print("{} absent precision: ".format(topk), absent_precision[topk])
                print("{} absent recall: ".format(topk), absent_recall[topk])
            """

            if topk in total_present_precision:
                total_present_precision[topk] += present_precision[topk]
            else:
                total_present_precision[topk] = present_precision[topk]

            if topk in total_present_recall:
                total_present_recall[topk] += present_recall[topk]
            else:
                total_present_recall[topk] = present_recall[topk]

            if topk in total_absent_precision:
                total_absent_precision[topk] += absent_precision[topk]
            else:
                total_absent_precision[topk] = absent_precision[topk]

            if topk in total_absent_recall:
                total_absent_recall[topk] += absent_recall[topk]
            else:
                total_absent_recall[topk] = absent_recall[topk]

        total_data += 1
        i += 1

    return {"total_data": total_data,
            "total_present_precision": total_present_precision,
            "total_present_recall": total_present_recall,
            "total_absent_recall": total_absent_recall,
            "total_absent_precision": total_absent_precision}
