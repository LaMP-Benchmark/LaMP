import argparse
import json
import os

def merge(inps, outs, label_files, input_files, score_name):
    for inp in inps:
        del inp['profile']
        for o in outs:
            if o['id'] == inp['id']:
                output = o['output']
                break
        inp['gold'] = output
        labels = []
        outputs = []
        inputs = []
        for k, label_file in enumerate(label_files):
            print(k)
            labels.append(label_file[inp['id']]['metric'][score_name])
            outputs.append(label_file[inp['id']]['output'])
        for input_file in input_files:
            inputs.append(input_file[inp['id']]['input'])
        inp['labels'] = labels
        inp['outputs'] = outputs
        inp['inputs'] = inputs
    return inps

parser = argparse.ArgumentParser()

parser.add_argument("--retrivers_data_addr", '--names-list', nargs='+', required=True)
parser.add_argument("--task_inputs_addr", required=True)
parser.add_argument("--task_outputs_addr", required=True)
parser.add_argument("--output_dataset_addr", required=True)
parser.add_argument("--metric", required=True)

if __name__ == "__main__":
    opts = parser.parse_args()

    score_name = opts.metric

    q_addr = opts.task_inputs_addr
    o_addr = opts.task_outputs_addr
    res_addr = opts.output_dataset_addr
    retrivers_data_addrs = opts.retrivers_data_addr

    with open(q_addr) as qfile, open(o_addr) as oflie, open(res_addr, "w") as resfile:
        inp = json.load(qfile)
        out = json.load(oflie)
        scores_file = []
        input_file = []
        for x in retrivers_data_addrs:
            with open(os.path.join(x, "scores.json")) as sfile:
                scores_file.append(json.load(sfile))
            with open(os.path.join(x, "data.json")) as sfile:
                input_file.append(json.load(sfile))
        res = merge(inp, out['golds'], scores_file, input_file, score_name)
        json.dump(res, resfile, indent=4)
