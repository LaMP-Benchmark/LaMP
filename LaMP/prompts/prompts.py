from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
from prompts.utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_after_colon, extract_after_description, extract_after_abstract
from prompts.contriever_retriever import retrieve_top_k_with_contriever
import random

def classification_citation_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}'
    return corpus, query

def classification_news_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    query = extract_after_article(inp)
    return corpus, query

def classification_movies_query_corpus_maker(inp, profile):
    corpus = [f'{x["description"]}' for x in profile]
    query = extract_after_description(inp)
    return corpus, query

def classification_review_query_corpus_maker(inp, profile):
    corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_review(inp)
    return corpus, query

def generation_news_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    query = extract_after_article(inp)
    return corpus, query

def generation_paper_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    query = extract_after_paper(inp)
    return corpus, query

def generation_paper_long_query_corpus_maker(inp, profile):
    corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    query = extract_after_abstract(inp)
    return corpus, query


def parphrase_tweet_query_corpus_maker(inp, profile):
    corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_colon(inp)
    return corpus, query

def generation_avocado_query_corpus_maker(inp, profile):
    corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_colon(inp)
    return corpus, query

def generation_avocado_long_query_corpus_maker(inp, profile):
    corpus = [f'{x["text"]} {x["title"]}' for x in profile]
    query = extract_after_colon(inp)
    return corpus, query

def create_classification_citation_prompt(inp, profile, max_length, tokenizer):
    prompts = []
    per_p_max_length = (max_length - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0
    for p in profile:
        tokens = tokenizer(p["title"], max_length=per_p_max_length + saved_tokens - 2, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - 2
        new_title = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{new_title}"'
        prompts.append(prompt)
    return add_string_after_title(inp, ", and ".join(prompts))

def create_classification_news_prompt(inp, profile, max_length, tokenizer): # good
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'the category for the article: " " is "{p["category"]}" ')['input_ids'])
        tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'the category for the article: "{new_text}" is "{p["category"]}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'

def create_classification_movies_prompt(inp, profile, max_length, tokenizer): # good
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'the tag for the movie: " " is "{p["tag"]}" ')['input_ids'])
        tokens = tokenizer(p["description"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'the tag for the movie: "{new_text}" is "{p["tag"]}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'

def create_classification_review_prompt(inp, profile, max_length, tokenizer):
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'{p["score"]} is the score for " " ')['input_ids'])
        tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'{p["score"]} is the score for "{new_text}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'

def create_generation_news_prompt(inp, profile, max_length, tokenizer):
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'"{p["title"]}" is the title for " " ')['input_ids'])
        tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["title"]}" is the title for "{new_text}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'

def create_generation_paper_prompt(inp, profile, max_length, tokenizer):
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1) - len(tokenizer("Following the given patterns")['input_ids'])) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'"{p["title"]}" is a title " " ')['input_ids'])
        tokens = tokenizer(p["abstract"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_asbtract = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["title"]}" is a title for "{new_asbtract}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns {inp}'

def create_generation_paper_long_prompt(inp, profile, max_length, tokenizer):
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1) - len(tokenizer("Following the given patterns")['input_ids'])) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'"{p["title"]}" is the title " " ')['input_ids'])
        tokens = tokenizer(p["abstract"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_asbtract = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["title"]}" is the title for "{new_asbtract}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns {inp}'


def create_parphrase_tweet_prompt(inp, profile, max_length, tokenizer):
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1) - len(tokenizer("are written by user. Following the given patterns")['input_ids'])) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'"" ')['input_ids'])
        tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_asbtract = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{new_asbtract}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)} are written by a person. Following the given patterns {inp}'

def create_generation_avocado_prompt(inp, profile, max_length, tokenizer):
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1)) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'"{p["title"]}" is the title for " " ')['input_ids'])
        tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["title"]}" is the title for "{new_text}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. {inp}'

def create_generation_avocado_long_prompt(inp, profile, max_length, tokenizer):
    per_p_max_length = (max_length - 1 - 2 * (len(profile) - 1) - len(tokenizer("are written by user. Following the given patterns")['input_ids'])) // len(profile)
    saved_tokens = 0
    prompts = []
    for p in profile:
        needed_part_len = len(tokenizer(f'"{p["title"]}" is the title for " " ')['input_ids'])
        tokens = tokenizer(p["text"], max_length=per_p_max_length + saved_tokens - needed_part_len, truncation=True)
        saved_tokens += per_p_max_length - len(tokens['input_ids']) - needed_part_len
        new_text = tokenizer.batch_decode([tokens['input_ids']], skip_special_tokens=True)[0]
        prompt = f'"{p["title"]}" is the title for "{new_text}" '
        prompts.append(prompt)
    return f'{", and ".join(prompts)}. Following the given patterns {inp}'

def create_prompt_generator(num_retrieve, ret_type = "bm25", is_ranked = False, max_length = 512, tokenizer = None):
    contriever = None
    if ret_type == "contriever" and not is_ranked:
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        contriever = AutoModel.from_pretrained('facebook/contriever').to("cuda:0")
        contriever.eval()

    def prompt(inp, profile, task):
        if task == "LaMP-1":
            corpus, query = classification_citation_query_corpus_maker(inp, profile)
        elif task == "LaMP-2-old":
            corpus, query = classification_news_query_corpus_maker(inp, profile)
        elif task == "LaMP-2":
            corpus, query = classification_movies_query_corpus_maker(inp, profile)
        elif task == "LaMP-3":
            corpus, query = classification_review_query_corpus_maker(inp, profile)
        elif task == "LaMP-4":
            corpus, query = generation_news_query_corpus_maker(inp, profile)
        elif task == "LaMP-5":
            corpus, query = generation_paper_query_corpus_maker(inp, profile)
        elif task == "LaMP-7":
            corpus, query = parphrase_tweet_query_corpus_maker(inp, profile)
        elif task == "LaMP-6":
            corpus, query = generation_avocado_query_corpus_maker(inp, profile)
        
        if not is_ranked:
            if ret_type == "bm25":
                tokenized_corpus = [x.split() for x in corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.split()
                selected_profs = bm25.get_top_n(tokenized_query, profile, n=num_retrieve)
            elif ret_type == "contriever":
                selected_profs = retrieve_top_k_with_contriever(contriever, tokenizer, corpus, profile, query, num_retrieve)
            elif ret_type == "random":
                selected_profs = random.choices(profile, k = num_retrieve)
            elif ret_type == "recency":
                profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
                selected_profs = profile[-num_retrieve:][::-1]
        else:
            if ret_type == "recency_contriever":
                selected_profs_cont = profile[:num_retrieve // 2]
                profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
                selected_profs_rec = profile[-(num_retrieve // 2):][::-1]
                selected_profs = selected_profs_cont + selected_profs_rec
            else:
                selected_profs_cont = profile[:num_retrieve]
                selected_profs = selected_profs_cont
        factor = 0.6
        while True:
            try:
                max_len_prompt = max_length - min(len(tokenizer(inp)['input_ids']), int(factor * max_length))
                if task == "LaMP-1":
                    return create_classification_citation_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP-2-old":
                    return create_classification_news_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP-2":
                    return create_classification_movies_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP-3":
                    return create_classification_review_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP-4":
                    return create_generation_news_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP-5":
                    return create_generation_paper_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP-7":
                    return create_parphrase_tweet_prompt(inp, selected_profs, max_len_prompt, tokenizer)
                elif task == "LaMP-6":
                    return create_generation_avocado_prompt(inp, selected_profs, max_len_prompt, tokenizer)
            except:
                factor -= 0.1
                if factor < 0:
                    print("not possible")
                    return inp
    return prompt, contriever