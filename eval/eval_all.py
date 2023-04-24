from evaluation import LaMPEvaluation
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--golds_zip", required=True, help="Address to all gold labels for all tasks zipped in a file")
parser.add_argument("--preds_zip", required=True, help="Address to all predictions for all tasks zipped in a file")
parser.add_argument("--temp_dir", required=False, help="Address to a temp dir for extracting files", default="./tmp")
parser.add_argument("--output_file", required=True, help="Address to the results file")

if __name__ == "__main__":

    opts = parser.parse_args()

    evaluator = LaMPEvaluation(all_golds_zip_file_addr=opts.golds_zip, extract_addr=opts.temp_dir)
    results = evaluator.evaluate_all(opts.preds_zip)
    with open(opts.output_file, "w") as file:
        json.dump(results, file)
