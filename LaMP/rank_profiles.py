import torch
from prompts.utils import batchify
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
from prompts.utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_after_colon, extract_after_abstract, extract_after_description
from rank_bm25 import BM25Okapi
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_data_addr", required = True)
parser.add_argument("--output_ranking_addr", required = True)
parser.add_argument("--task", required = True)
parser.add_argument("--ranker", required = True)
parser.add_argument("--batch_size", type = int, default=16)
parser.add_argument("--use_date", action='store_true')
parser.add_argument("--contriever_checkpoint", default="facebook/contriever")


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, k, batch_size = 16):
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda:0")
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    for batch in batched_corpus:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        outputs_batch = contriver(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    return [profile[m] for m in topk_indices.tolist()]

def retrieve_top_k_with_bm25(corpus, profile, query, k):
    tokenized_corpus = [x.split() for x in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    selected_profs = bm25.get_top_n(tokenized_query, profile, n=k)
    return selected_profs

def classification_citation_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x['id'] for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f'{extracted[1]} {extracted[2]}'
    return corpus, query, ids

def classification_review_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_review(inp)
    return corpus, query, ids

def generation_news_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_article(inp)
    return corpus, query, ids

def generation_paper_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids

def parphrase_tweet_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_colon(inp)
    ids = [x['id'] for x in profile]
    return corpus, query, ids

def generation_avocado_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x['id'] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids

def classification_movies_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["description"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["description"]}' for x in profile]
    query = extract_after_description(inp)
    ids = [x['id'] for x in profile]
    return corpus, query, ids


if __name__ == "__main__":

    opts = parser.parse_args()
    task = opts.task
    ranker = opts.ranker

    with open(opts.input_data_addr) as file:
        dataset = json.load(file)
    
    rank_dict = dict()

    for data in tqdm.tqdm(dataset):
        inp = data['input']
        profile = data['profile']
        if task == "LaMP-1":
            corpus, query, ids = classification_citation_query_corpus_maker(inp, profile, opts.use_date)
        elif task == "LaMP-3":
            corpus, query, ids = classification_review_query_corpus_maker(inp, profile, opts.use_date)
        elif task == "LaMP-2":
            corpus, query = classification_movies_query_corpus_maker(inp, profile, opts.use_date)
        elif task == "LaMP-4":
            corpus, query, ids = generation_news_query_corpus_maker(inp, profile, opts.use_date)
        elif task == "LaMP-5":
            corpus, query, ids = generation_paper_query_corpus_maker(inp, profile, opts.use_date)
        elif task == "LaMP-7":
            corpus, query, ids = parphrase_tweet_query_corpus_maker(inp, profile, opts.use_date)
        elif task == "LaMP-6":
            corpus, query, ids = generation_avocado_query_corpus_maker(inp, profile, opts.use_date)
        
        if ranker == "contriever":
            tokenizer = AutoTokenizer.from_pretrained(opts.contriever_checkpoint)
            contriver = AutoModel.from_pretrained(opts.contriever_checkpoint).to("cuda:0")
            contriver.eval()
            randked_profile = retrieve_top_k_with_contriver(contriver, tokenizer, corpus, profile, query, len(profile), opts.batch_size)
        elif ranker == "bm25":
            randked_profile = retrieve_top_k_with_bm25(corpus, profile, query, len(profile))
        elif ranker == "recency":
            profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
            randked_profile = profile[::-1]

        data['profile'] = randked_profile

        rank_dict[data['id']] = [x['id'] for x in randked_profile]

    
    with open(opts.output_ranking_addr, "w") as file:
        json.dump(rank_dict, file)