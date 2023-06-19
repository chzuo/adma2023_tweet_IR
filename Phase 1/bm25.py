from rank_bm25 import BM25Okapi,BM25Plus
import pandas as pd
import json,os,glob,time,re,string
import spacy
nlp = spacy.load('en_core_web_sm')

def get_claim(title):
    title=title.replace(',','')
    doc = nlp(title)
    claim = ' '.join([token.lemma_ for token in doc if token.is_stop == False and token.text.isalnum() == True])
    claim = claim.lower()
    #print(title,claim)
    return claim
 
# open .tsv file
df = pd.read_csv('test.tsv', sep='\t',names=['claim','news_id'])
tweet_newsid_dict=dict(zip(df['claim'],df['news_id']))

#create data
tweet_claim_dict = dict()
claim_newsid_dict = dict()
for tweet in tweet_newsid_dict:
    claim = get_claim(tweet)
    tweet_claim_dict[tweet] = claim
    claim_newsid_dict[claim] = tweet_newsid_dict[tweet]

news_df = pd.read_csv('newsid_title.tsv', sep='\t',names=['newsid','title'])
newsid_title_dict=dict(zip(df['news_id'],df['title']))

for newsid in list(newsid_title_dict.keys()):
    clean_title = get_claim(newsid_title_dict[newsid])
    newsid_title_token_dict[newsid]=clean_title

# start bm25 search
tokenized_corpus_t = [doc.split(" ") for doc in list(newsid_title_token_dict.values())]
bm25 = BM25Okapi(tokenized_corpus_t)
tweet_title_ir_results_dict=dict()
for query in list(tweet_newsid_dict.keys())[:]:
    claim = tweet_claim_dict[query]
    tokenized_query = claim.split()
    doc_scores = bm25.get_scores(tokenized_query)
    candidate_set = [list(newsid_title_token_dict.keys())[i] for i in np.argsort(doc_scores)[::-1][:1000]]
    tweet_title_ir_results_dict[query]=candidate_set

# start evaluation
def get_results(results_dict):
    result = [0 for i in range(4)]
    mrr = 0
    for tweet in results_dict:
        candidate_set = results_dict[tweet]
        newsid = tweet_newsid_dict[tweet]
        if str(newsid) in candidate_set:
            mrr += 1/(candidate_set.index(str(newsid))+1)
            for i,index in zip([1,10,100,500],range(4)):
                if str(newsid) in candidate_set[:i]:           
                    result[index] += 1
        #else:
         #   print(tweet)

    result_score = [round(s/float(len(results_dict))*100,ndigits=1) for s in result]
    mrr = round(mrr/len(results_dict)*100,ndigits=1)
    print(result_score)
    print(mrr)
get_results(tweet_title_ir_results_dict)