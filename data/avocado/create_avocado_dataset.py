import zipfile
import glob
import os
import shutil
import json
import tqdm
import mailparser
import argparse

def empty_dir(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            # if the current item is a file, remove it
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # if the current item is a directory, remove it recursively using shutil.rmtree()
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def process_file(file_addr):
    message = ""
    id = os.path.basename(file_addr)
    mail = mailparser.parse_from_file(file_addr)
    subject = mail.subject
    message = mail.body
    return id, {"subject" : subject, "content" : message.strip()}

parser = argparse.ArgumentParser()

parser.add_argument("--avocado_files_dir", required=True, help="Address to the directory containing zip files for avocado dataset 'avocado-1.0.2/data/text'")
parser.add_argument("--extract_addr", required=True, help="A temp dir to extract the files for creating dataset")
parser.add_argument("--output_dir", required=True, help="The directory to generate the final dataset")
parser.add_argument("--input_question_file_train", required=True, help="The address to the train_questions.json file")
parser.add_argument("--input_question_file_dev", required=True, help="The address to the dev_questions.json file")
parser.add_argument("--input_question_file_test", required=True, help="The address to the test_questions.json file")

if __name__ == "__main__":
    opts = parser.parse_args()

    with open(opts.input_question_file_train) as file:
        input_questions_train = json.load(file)
    with open(opts.input_question_file_dev) as file:
        input_questions_dev = json.load(file)
    with open(opts.input_question_file_test) as file:
        input_questions_test = json.load(file)
    
    all_required_files = set()
    for sample in input_questions_train + input_questions_dev + input_questions_test:
        all_required_files.add(sample['input'])
        for p in sample['profile']:
            all_required_files.add(p['text'])
    
    zip_addrs = glob.glob(os.path.join(opts.avocado_files_dir, "*"))
    os.makedirs(opts.extract_addr, exist_ok=True)
    database = dict()
    for zip_addr in tqdm.tqdm(zip_addrs):
        with zipfile.ZipFile(zip_addr, 'r') as zobj:
            zobj.extractall(path = opts.extract_addr)
            extracted_files_addrs = glob.glob(os.path.join(opts.extract_addr, "*/*"))
            for file_addr in extracted_files_addrs:
                if os.path.basename(file_addr) in all_required_files:
                    id, obj = process_file(file_addr)
                    database[id] = obj
        empty_dir(opts.extract_addr)
    
    os.makedirs(opts.output_dir, exist_ok=True)

    inps_train, outs_train = [], []
    for sample in input_questions_train:
        id = sample['input']
        sample['input'] = f"Generate a subject for the following email: {database[id]['content']}"
        sample['output'] = database[id]['subject']
        for p in sample['profile']:
            pid = p['text']
            p['text'] = database[pid]['content']
            p['title'] = database[pid]['subject']
        inps_train.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile']})
        outs_train.append({"id" : sample['id'], "output" : sample['output']})

    inps_dev, outs_dev = [], []
    for sample in input_questions_dev:
        id = sample['input']
        sample['input'] = f"Generate a subject for the following email: {database[id]['content']}"
        sample['output'] = database[id]['subject']
        for p in sample['profile']:
            pid = p['text']
            p['text'] = database[pid]['content']
            p['title'] = database[pid]['subject']
        inps_dev.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile']})
        outs_dev.append({"id" : sample['id'], "output" : sample['output']})

    
    inps_test= []
    for sample in input_questions_test:
        id = sample['input']
        sample['input'] = f"Generate a subject for the following email: {database[id]['content']}"
        for p in sample['profile']:
            pid = p['text']
            p['text'] = database[pid]['content']
            p['title'] = database[pid]['subject']
        inps_test.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile']})
        
    with open(os.path.join(opts.output_dir, "train_questions.json"), "w") as file:
        json.dump(inps_train, file)
    with open(os.path.join(opts.output_dir, "train_outputs.json"), "w") as file:
        json.dump({"task":"LaMP_6","golds":outs_train}, file)

    with open(os.path.join(opts.output_dir, "dev_questions.json"), "w") as file:
        json.dump(inps_dev, file)
    with open(os.path.join(opts.output_dir, "dev_outputs.json"), "w") as file:
        json.dump({"task":"LaMP_6","golds":outs_dev}, file)

    with open(os.path.join(opts.output_dir, "test_questions.json"), "w") as file:
        json.dump({"task":"LaMP_6","golds":inps_test}, file)

        

