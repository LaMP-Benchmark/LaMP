import json
import zipfile
import glob
import os
import shutil
import evaluate

def postprocess_text_classification(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    return preds, labels

def postprocess_text_generation(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def create_metric_f1_accuracy(all_labels):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        result_acc = accuracy_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, labels=list(range(len(all_labels))), average = "macro")
        result = {"accuracy" : result_acc["accuracy"], "f1" : result_f1["f1"]}
        return result
    return compute_metrics

def create_metric_f1_accuracy_sigtest(all_labels):
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        results_acc = []
        results_f1 = []
        for pred, gold in zip(decoded_preds, decoded_labels):
            result_acc = accuracy_metric.compute(predictions=[pred], references=[gold])
            result_f1 = f1_metric.compute(predictions=[pred], references=[gold], labels=list(range(len(all_labels))), average = "macro", pos_label = gold)
            results_acc.append(result_acc["accuracy"])
            results_f1.append(result_f1["f1"])
        result = {"accuracy" : results_acc, "f1" : results_f1}
        return result
    return compute_metrics

def create_metric_mae_rmse():
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
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]
        result_mae = mae_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_rmse = mse_metric.compute(predictions=decoded_preds, references=decoded_labels, squared = False)
        result = {"MAE" : result_mae["mae"], "RMSE" : result_rmse["mse"]}
        return result
    return compute_metrics

def create_metric_mae_rmse_sigtest():
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
        decoded_preds, decoded_labels = postprocess_text_classification(decoded_preds, decoded_labels)
        decoded_preds = [create_mapping(x,y) for x,y in zip(decoded_preds, decoded_labels)]
        decoded_labels = [create_mapping(x,x) for x in decoded_labels]
        results_mae = []
        results_rmse = []
        for pred, gold in zip(decoded_preds, decoded_labels):
            result_mae = mae_metric.compute(predictions=[pred], references=[gold])
            result_rmse = mse_metric.compute(predictions=[pred], references=[gold], squared = False)
            results_mae.append(result_mae["mae"])
            results_rmse.append(result_rmse["mse"])
        result = {"MAE" : results_mae, "RMSE" : results_rmse}
        return result
    return compute_metrics

def create_metric_rouge():
    rouge_metric = evaluate.load('rouge')
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
        return result
    return compute_metrics

def create_metric_rouge_sigtest():
    rouge_metric = evaluate.load('rouge')
    def compute_metrics(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = postprocess_text_generation(decoded_preds, decoded_labels)
        result_rouge = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator = False)
        result = {"rouge-1" : result_rouge["rouge1"], "rouge-L" : result_rouge["rougeL"]}
        return result
    return compute_metrics

class LaMPEvaluation(object):
    
    def __init__(self, all_golds_zip_file_addr = None, single_gold_json_file_addr = None, extract_addr = "./tmp") -> None:
        assert all_golds_zip_file_addr or single_gold_json_file_addr, "The golds should be provided for all datasets or at least one."
        assert not (all_golds_zip_file_addr and single_gold_json_file_addr), "The golds should be provided using zip file or json file not both."
        self.tasks_golds = dict()
        self.extract_addr = extract_addr
        self.evaluate_all_is_possible = False
        if all_golds_zip_file_addr:
            os.makedirs(self.extract_addr, exist_ok=True)
            with zipfile.ZipFile(all_golds_zip_file_addr, 'r') as zobj:
                zobj.extractall(path = extract_addr)
            for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
                with open(file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
            self._empty_dir(self.extract_addr)
            self.evaluate_all_is_possible = True
        if single_gold_json_file_addr:
            with open(single_gold_json_file_addr) as file:
                    task = json.load(file)
                    self.tasks_golds[task['task']] = task['golds']
    
    def _empty_dir(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def _get_all_gold_ids(self, task_name):
        return set([sample['id'] for sample in self.tasks_golds[task_name]])
    
    def _get_all_ids(self, input):
        return set([sample['id'] for sample in input])
    
    def evaluate_all(self, predicts_zipfile_addr):
        assert self.evaluate_all_is_possible, "You did not provide golds for all tasks."
        with zipfile.ZipFile(predicts_zipfile_addr, 'r') as zobj:
            zobj.extractall(path = self.extract_addr)
        results_raw = dict()
        all_task_names = set()
        for file_addr in glob.glob(os.path.join(self.extract_addr, "**/*.json"), recursive=True):
            with open(file_addr) as file:
                preds = json.load(file)
            all_task_names.add(preds['task'])
            results_raw[preds['task']] = self._evaluate_task(preds['golds'], preds['task'])
        self._empty_dir(self.extract_addr)
        assert len(all_task_names) == 7, "The provided results do not cover all the tasks in the benchmark."
        return results_raw

    def evaluate_task(self, predicts_json_addr, task_name):
        with open(predicts_json_addr) as file:
            preds = json.load(file)
        assert preds['task'] == task_name or preds['task'].replace("-","_") == task_name, "The provided task_name and the results do not match."
        assert preds['task'] in self.tasks_golds.keys() or preds['task'].replace("-","_") in self.tasks_golds.keys(), "The provided golds cannot be used to evaluate this task."
        return self._evaluate_task(preds['golds'], task_name)
        
    def _evaluate_task(self, predictions, task_name):
        golds_dict = {y['id']:y['output'] for y in self.tasks_golds[task_name]}
        preds_dict = {x['id']:x['output'] for x in predictions}
        
        gold_ids = self._get_all_gold_ids(task_name)
        pred_ids = self._get_all_ids(predictions)
        print(gold_ids - pred_ids)
        assert gold_ids == pred_ids, "Predictions ids and gold ids do not match."

        if task_name in ["LaMP_1", "LaMP_2"]:
            metric = create_metric_f1_accuracy(self._get_labels(task_name))
        elif task_name == "LaMP_3":
            metric = create_metric_mae_rmse()
        else:
            metric = create_metric_rouge()
        
        gold_ids = list(gold_ids)
        golds = [golds_dict[id] for id in gold_ids]
        preds = [preds_dict[id] for id in gold_ids]
        return metric(preds, golds)
    
    def _evaluate_task_per_sample(self, predictions, task_name):
        golds_dict = {y['id']:y['output'] for y in self.tasks_golds[task_name]}
        preds_dict = {x['id']:x['output'] for x in predictions}
        
        gold_ids = self._get_all_gold_ids(task_name)
        pred_ids = self._get_all_ids(predictions)
        print(gold_ids - pred_ids)
        assert gold_ids == pred_ids, "Predictions ids and gold ids do not match."

        if task_name in ["LaMP_1", "LaMP_2"]:
            metric = create_metric_f1_accuracy_sigtest(self._get_labels(task_name))
        elif task_name == "LaMP_3":
            metric = create_metric_mae_rmse_sigtest()
        else:
            metric = create_metric_rouge_sigtest()
        
        gold_ids = list(gold_ids)
        golds = [golds_dict[id] for id in gold_ids]
        preds = [preds_dict[id] for id in gold_ids]
        return metric(preds, golds)
    
    def _get_labels(self, task_name):
        if task_name == "LaMP_1":
            return ["[1]", "[2]"]
        elif task_name == "LaMP_2":
            return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
        elif task_name == "LaMP_3":
            return ["1", "2", "3", "4", "5"]
        else:
            raise ValueError("Invalid task_name")