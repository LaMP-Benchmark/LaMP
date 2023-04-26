# LaMP: When Large Language Models Meet Personalization

This paper highlights the importance of personalization in the current state of natural language understanding and generation and introduces the LaMP benchmark --- a novel benchmark for training and evaluating language models for producing personalized outputs. LaMP offers a comprehensive evaluation framework with diverse language tasks and multiple entries for each user profile. It consists of seven personalized tasks, spanning across three classification and four text generation tasks. We further propose a retrieval augmentation approach that retrieves personalized items from user profiles to construct personalized prompts for large language models. The experiments conducted to establish fine-tuned and zero-shot baseline results for the benchmark conclude that LMs utilizing profile augmentation outperform their counterparts that do not factor in profile information.

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

## Reference

[LaMP: When Large Language Models Meet Personalization](https://arxiv.org/abs/2304.11406)

```
@misc{salemi2023lamp,
      title={LaMP: When Large Language Models Meet Personalization}, 
      author={Alireza Salemi and Sheshera Mysore and Michael Bendersky and Hamed Zamani},
      year={2023},
      eprint={2304.11406},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
LaMP (codes and data creation methods) is licensed by Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). See the [CC-BY-NC-SA-4.0.txt](CC-BY-NC-SA-4.0.txt) file for details. For the datasets in this benchmark, you should follow their license.
