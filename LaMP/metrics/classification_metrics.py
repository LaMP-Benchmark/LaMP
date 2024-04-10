import numpy as np
import evaluate

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

def create_metric_f1_accuracy(tokenizer, all_labels):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            print(x)
            return -1
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, labels=list(range(len(all_labels))), average = "macro")
        result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
        return result
    return compute_metrics

def create_metric_f1_accuracy_bert(all_labels):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds, axis=1)
        result_acc = accuracy_metric.compute(predictions=preds, references=labels)
        result_f1 = f1_metric.compute(predictions=preds, references=labels, labels=list(range(len(all_labels))), average = "macro")
        result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
        return result
    return compute_metrics

def create_metric_mae_rmse_bert(all_labels):
    mse_metric = evaluate.load("mse")
    mae_metric = evaluate.load("mae")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds, axis=1)
        result_mae = mae_metric.compute(predictions=preds, references=labels)
        result_rmse = mse_metric.compute(predictions=preds, references=labels, squared = False)
        result = {"mae" : result_mae["mae"], "rmse" : result_rmse["mse"]}
        return result
    return compute_metrics

def create_metric_mae_rmse(tokenizer, all_labels):
    mse_metric = evaluate.load("mse")
    mae_metric = evaluate.load("mae")
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared = False)
        result = {"mae" : result_mae["mae"], "rmse" : result_rmse["mse"]}
        return result
    return compute_metrics


def create_metric_f1_accuracy_chatgpt(all_labels):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            print(x)
            return -1
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, labels=list(range(len(all_labels))), average = "macro")
        result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
        return result
    return compute_metrics

def create_metric_mae_rmse_chatgpt(all_labels):
    mse_metric = evaluate.load("mse")
    mae_metric = evaluate.load("mae")
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared = False)
        result = {"mae" : result_mae["mae"], "rmse" : result_rmse["mse"]}
        return result
    return compute_metrics