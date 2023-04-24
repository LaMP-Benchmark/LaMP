from evaluation import LaMPEvaluation
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--golds_json", required=True, help="Address to all gold labels for the task as a json file")
parser.add_argument("--preds_json", required=True, help="Address to all predictions for the task as a json file")
parser.add_argument("--task_name", required=True, help="[LaMP_1, LaMP_2, LaMP_3, LaMP_4, LaMP_5, LaMP_6, LaMP_7]")
parser.add_argument("--output_file", required=True, help="Address to the results file")

if __name__ == "__main__":

    opts = parser.parse_args()

    evaluator = LaMPEvaluation(single_gold_json_file_addr=opts.golds_json)
    results = evaluator.evaluate_task(opts.preds_json, opts.task_name)
    with open(opts.output_file, "w") as file:
        json.dump(results, file)
