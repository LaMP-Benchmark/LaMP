# Codes for papers on Large Language Models Personalization (LaMP)

[LaMP: When Large Language Models Meet Personalization](https://arxiv.org/abs/2304.11406)

```
@misc{salemi2023lamp,
      title={La{MP}: When Large Language Models Meet Personalization}, 
      author={Alireza Salemi and Sheshera Mysore and Michael Bendersky and Hamed Zamani},
      year={2023},
      eprint={2304.11406},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

This paper highlights the importance of personalization in the current state of natural language understanding and generation and introduces the LaMP benchmark --- a novel benchmark for training and evaluating language models for producing personalized outputs. LaMP offers a comprehensive evaluation framework with diverse language tasks and multiple entries for each user profile. It consists of seven personalized tasks, spanning across three classification and four text generation tasks. We further propose a retrieval augmentation approach that retrieves personalized items from user profiles to construct personalized prompts for large language models. The experiments conducted to establish fine-tuned and zero-shot baseline results for the benchmark conclude that LMs utilizing profile augmentation outperform their counterparts that do not factor in profile information.

[Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation](https://arxiv.org/abs/2404.05970)

This paper studies retrieval-augmented approaches for personalizing large language models (LLMs), which potentially have a substantial impact on various applications and domains. We propose the first attempt to optimize the retrieval models that deliver a limited number of personal documents to large language models for the purpose of personalized generation. We develop two optimization algorithms that solicit feedback from the downstream personalized generation tasks for retrieval optimization--one based on reinforcement learning whose reward function is defined using any arbitrary metric for personalized generation and another based on knowledge distillation from the downstream LLM to the retrieval model. This paper also introduces a pre- and post-generation retriever selection model that decides what retriever to choose for each LLM input. Extensive experiments on diverse tasks from the language model personalization (LaMP) benchmark reveal statistically significant improvements in six out of seven datasets.

```
@misc{salemi2024optimization,
      title={Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation}, 
      author={Alireza Salemi and Surya Kallumadi and Hamed Zamani},
      year={2024},
      eprint={2404.05970},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Data

You can download all the datasets from the links provided [here](https://lamp-benchmark.github.io/download). However, we provided the minimal ids to generate the dataset using our codes for the Personalized Email Subject Generation because this dataset is not publicly accessible. Follow the following section to generate that dataset.

### LaMP 6: Personalized Email Subject Generation (Avocado dataset)

The [Avocado](https://catalog.ldc.upenn.edu/LDC2015T03) dataset is not publicly accessible. However, we provided the samples' id and the code we used to generate our dataset. Therefore, if you get access to the dataset, you can quickly generate the dataset with the same format as the other datasets in LaMP using the following code:

```
python data/avocado/create_avocado_dataset.py \
    --avocado_files_dir \*Address to the directory containing zip files for avocado dataset 'avocado-1.0.2/data/text'*\ \
    --extract_addr \*A temp dir to extract the files for creating dataset*\ \
    --output_dir \*The directory to generate the final dataset*\ \
    --input_question_file_train \*The address to the train_questions.json file we provided in LaMP*\ \
    --input_question_file_dev \*The address to the dev_questions.json file we provided in LaMP*\ \
    --input_question_file_test \*The address to the test_questions.json file we provided in LaMP*\
```

## Evaluation

The instructions for evaluating your results on the test set are provided [here](https://lamp-benchmark.github.io/leaderboard). In order to evaluate your results on the dev set, we provided an evaluation script that can be found here:


Evaluate all tasks together:

```
python eval/eval_all.py \
    --golds_zip /*Address to all gold labels for all tasks zipped in a file*/ \
    --preds_zip /*Address to all predictions for all tasks zipped in a file*/ \
    --temp_dir /*Address to a temp dir for extracting files*/ \
    --output_file /*Address to the results file*/ \
```

Evaluate one task:

```
python eval/eval_task.py \
    --golds_json /*Address to gold labels for the task as a json file*/ \
    --preds_json /*Address to predictions for the task as a json file*/ \
    --task_name /*Name of the task [LaMP_1, LaMP_2, LaMP_3, LaMP_4, LaMP_5, LaMP_6, LaMP_7]*/
    --output_file /*Address to the results file*/ \
```

The pred files should follow the exact same format as the gold files:

```
{
    "task" : "/*task name*/",
    "golds" : [
        {
            "id" : "/*sample 1 id*/",
            "output" : "/*output of the model for the first sample*/"
        },
        ...,
        {
            "id" : "/*sample n id*/",
            "output" : "/*output of the model for the n'th sample*/"
        }
    ]
}
```

## Personalizing LLMs with RAG (LaMP)

You first need to create an environment for this using the following script:

```
python3 -m venv lamp_venv
source lamp_venv/bin/activate
pip install -r LaMP/requirements.txt
```

### Ranking Profiles based on the Input

The first step is to sort items in each user profile based on the input for the task:

```
cd LaMP
python rank_profiles.py \
    --input_data_addr /*input questions for one of the LaMP tasks*/ \
    --output_ranking_addr /*output address for the generated ranking file*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --ranker /*the ranking model to be used [bm25, contriever, recency]*/ \
    [optional] --use_date /*the batch size for ranking*/ \
    [optional] --use_date \ /*if used, it adds time to the text of each profile item*/
    [optional] --contriever_checkpoint /*address to the Contriever checkpoint to be used*/ \

```

After that, use the following script to sort the profiles in the dataset based on the ranking file:

```
cd LaMP
python utils/merge_with_rank.py \
    --lamp_questions_addr /*address to the LaMP task inputs file*/ \
    --lamp_output_addr /*address to the LaMP task outputs file*/ \
    --profile_ranking_addr /*address to the generated ranking file from the previous script*/
    --merged_output_addr /*address to the sorted dataset using the provided ranking file*/ \

```

### Training LLM with RAG

The next step is to train the LLM on a LaMP task:

```
cd LaMP
python train_llm.py \
    --train_data /*address to sorted training data using the previous step*/ \
    --validation_data /*address to sorted validation data using the previous step*/ \
    [optional] --test_data /*address to sorted test data using the previous step*/ \
    --model_name /*address to the model that should be used for initialization of the LLM*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --output_dir /*output directory to save results and checkpoints*/ \
    --retriever /*the ranking model to be used [bm25, contriever, recency]*/ \
    --use_profile \ /*used to perfrom personalization with RAG */
    --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
    --num_retrieved /*number of items to be retrieved from the user profile*/ \ 
```

### Zero-shot Evaluation of LLM with RAG

You can also evaluate the LLMs with the following script:

```
cd LaMP
python evaluate_llm.py \
    --validation_data /*address to sorted validation data using the previous step*/ \
    --model_addr /*address to the model that should be used for initialization of the LLM*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --output_dir /*output directory to save results */ \
    --use_profile \ /*used to perfrom personalization with RAG */
    --retriever /*the ranking model to be used [bm25, contriever, recency]*/ \
    --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
    --num_retrieved /*number of items to be retrieved from the user profile*/ \ 
```

## Optimizing Retrieval Model for Personalizing LLMs (ROPG)

This code uses the feedback from LLM to train a retrieval model for personalizing the LLM. You first need to create an environment for this using the following script:

```
python3 -m venv ropg_venv
source ropg_venv/bin/activate
pip install -r ROPG/requirements.txt
```

### Feedback Generation using LLM for Items in the User Profile

The first step is to collect feedback from the LLM using the following script:

```
cd LaMP
python profile_item_utilization_scorer.py \
    --train_data /*address to sorted training data using the previous steps*/ \
    --model_name /*address to the model that should be used for feedback generation*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --output_dir /*output directory to save results */ \
    --profile_size /*number of top k items from user profile to get feedback for them*/
```

### Optimizing Retrieval Model

You can use the following code to train a retrieval model based on the feedback generated from the previous step.

For training with ROPG-KD, which uses knowledge distillation, use the following script:

```
cd ROPG
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ train_kd.py \
    --train_data /*address to sorted training data using the previous steps*/ \
    --do_train \
    --scores_path /*address to the feedback file generated in the previous step*/
    --name /*output directory*/ \
    --ctx_size /*number of documents to be used for training the retrieval model for each query*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --temperature /*temperature for distillation*/
```

For training with ROPG-RL, which uses reinforcement learning, use the following script:

```
cd ROPG
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ train_rl.py \
    --train_data /*address to sorted training data using the previous steps*/ \
    --do_train \
    --scores_path /*address to the feedback file generated in the previous step*/
    --name /*output directory*/ \
    --ctx_size /*number of documents to be used for training the retrieval model for each query*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
```

## Retrieval Model Selection for Personalizing LLMs (RSPG)

This section uses the feedback from the LLM based on the performance of different retrieval models to train a retrieval model selector. You first need to create an environment for this using the following script:

```
python3 -m venv rspg_venv
source rspg_venv/bin/activate
pip install -r RSPG/requirements.txt
```


### Feedback Generation using LLM for each Retrieval Model

use the following code to get the feedback for each retrieval model in the retrieval model pool:

```
cd LaMP
python retriever_utilization_scorer.py \
    --data_addr /**address to sorted task data using the previous steps**/
    --model_name /*address to the model that should be used for feedback generation*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --output_dir /*output directory to save results */ \
    --use_profile \ /*use only in the case you want the feedback from RAG approach, shouldn't be used when getting feedback from an LLM without RAG*/
    --num_retrieved /*number of items to be retrieved from the user profile*/ \ 
    --retriever /*the retriever model that should be used to get feedback for*/ \
    --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
```

You should use the following script with all the retrieval models in your retrieval model pool. In our paper we used Contriever, ROPG-RL, ROPG-KD, Recency, BM25, and no retrieval (no RAG).

### Optimizing Retrieval Model Selector

The first step is to combine all the feedbacks got from the previous step and make a training and validation set:

```
cd RSPG
python utils/create_data.py \
    --retrivers_data_addr "/*address to feedback 1*/" "/*address to feedback 2*/" ... "/*address to feedback n*/" \
    --task_inputs_addr /*input questions for one of the LaMP tasks*/ \
    --task_outputs_addr /*outputs for one of the LaMP tasks*/ \
    --output_dataset_addr /*address to save the created dataset*/
    --metric /*the metric name that should be used as feedback [accuracy, rouge-1, rouge-l]*/ \
```

After this, you can use the following script to train the retrieval selector model (RSPG):

```
cd RSPG
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ rspg.py \
    --train_data /*address to the training data created in the previous step*/ \
    --val_data /*address to the validation data created in the previous step*/ \
    --rspg_type /*retrieval selection mode: [Pre, Post]*/
    --val_lamp_golds /*address to the a LaMP task output file for validation set*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --do_train \
    --name /*output directory*/ \
    --temperature /*distillation temperature*/ \
```

### Inference with Retrieval Model Selector

In order to do inference with RSPG, you can use the following script:

```
cd RSPG
NGPU=/*Number of GPUs*/ python -m torch.distributed.launch --nproc_per_node=/*Number of GPUs*/ rspg.py \
    --train_data /*address to the training data created in the previous step*/ \
    --val_data /*address to the validation data created in the previous step*/ \
    --rspg_type /*retrieval selection mode: [Pre, Post]*/
    --val_lamp_golds /*address to the a LaMP task output file for validation set*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --do_validation \
    --name /*output directory*/ \
    --model_path /*address to the checkpoint to be evaluated*/ \
```

## Reference

If you find this repository helpful, please cite the following works!

[LaMP: When Large Language Models Meet Personalization](https://arxiv.org/abs/2304.11406)

```
@misc{salemi2023lamp,
      title={La{MP}: When Large Language Models Meet Personalization}, 
      author={Alireza Salemi and Sheshera Mysore and Michael Bendersky and Hamed Zamani},
      year={2023},
      eprint={2304.11406},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation](https://arxiv.org/abs/2404.05970)

```
@misc{salemi2024optimization,
      title={Optimization Methods for Personalizing Large Language Models through Retrieval Augmentation}, 
      author={Alireza Salemi and Surya Kallumadi and Hamed Zamani},
      year={2024},
      eprint={2404.05970},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
LaMP (codes and data creation methods) is licensed by Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). See the [CC-BY-NC-SA-4.0.txt](CC-BY-NC-SA-4.0.txt) file for details. For the datasets in this benchmark, you should follow their license.

## Acknowledgments

This work was supported in part by the Center for Intelligent Information Retrieval, in part by NSF grant #2143434, in part by the Office of Naval Research contract number N000142212688, and in part by Lowe's, in part by an Amazon Research Award, Fall 2022 CFP, in part by an award from Google, and in part by an award from Microsoft. Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsors.
