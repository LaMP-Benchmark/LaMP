import json 
import argparse

def merge(inps, outs, ranks):
    for inp in inps:
        for o in outs:
            if o['id'] == inp['id']:
                output = o['output']
                break
        new_profile = []
        for x in ranks[inp['id']]:
            for y in inp['profile']:
                if y['id'] == x:
                    new_profile.append(y)
                    break
        inp['profile'] = new_profile
        inp['output'] = output
    return inps

parser = argparse.ArgumentParser()

parser.add_argument("--lamp_questions_addr", required = True)
parser.add_argument("--lamp_output_addr", required = True)
parser.add_argument("--merged_output_addr", required = True)
parser.add_argument("--profile_ranking_addr", default="")

if __name__ == "__main__":
    opts = parser.parse_args()
    q_addr = opts.lamp_questions_addr
    o_addr = opts.lamp_output_addr
    rank_addr = opts.profile_ranking_addr
    res_addr = opts.merged_output_addr

    with open(q_addr) as qfile:
        inp = json.load(qfile)
    with open(o_addr) as oflie:
        out = json.load(oflie)
    if rank_addr:
        with open(rank_addr) as rflie:
            rank = json.load(rflie)
    else:
        rank = dict()
        for data in inp:
            rank[data['id']] = []
            for item in data['profile']:
                rank[data['id']].append(item['id'])

    with open(res_addr, "w") as resfile:
        res = merge(inp, out, rank)
        json.dump(res, resfile, indent=4)


