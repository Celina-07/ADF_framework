import pandas as pd
import json
from tqdm import tqdm
import random
import numpy as np
import time
import sys
import os
import getopt
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import math
import ast


import warnings

warnings.filterwarnings("ignore")

### Initial values
dataset, algorithm, alpha_value = None, None, None

try:
    opts, args = getopt.getopt(
        sys.argv[1:], "d:a:v:", # options that require argument are followed by a colon
        ["dataset=", "algorithm=", "alpha_value="])

except getopt.GetoptError:
    print("Invalid command-line arguments.")
    print("Usage: ADF_script.py [-w] [-d <dataset>] [-a <algorithm>] [-v <alpha-value>]")
    sys.exit(2)


for opt, arg in opts:
    if opt in ("-d", "--dataset"):
        dataset = str(arg)
        # intensity =  ast.literal_eval(arg)
        
    elif opt in ("-a", "--algorithm"):
        algorithm = str(arg)

    elif opt in ("-v", "--alpha-value"):
        alpha_value = int(arg)/10 #Division by 10 to have a value between 0 and 1 

        
print(f'\ADF parameters:')
print(f"\tDataset: {dataset}")
print(f"\tAlgorithm: {algorithm}")
print(f"\tAlpha value: {alpha_value}")


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = []
    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_scores.append(normalized_score)
    return normalized_scores

def embeddings_to_df(e):
    e_list = e.tolist()
    df_e = pd.DataFrame(e_list)
    return df_e


if algorithm=='centroidVector': #CSV file
    datapath='reco_scores_baseline/{}/recommendations_{}_{}.csv'.format(dataset,algorithm,dataset)
    results = pd.read_csv(datapath, index_col=0) 
    results = results.rename(columns={'item_id':'news_id'})     
else: #JSON file
    datapath='reco_scores_baseline/{}/recommendations_{}_{}.json'.format(dataset,algorithm,dataset)
    with open(datapath, 'r') as f:
        results_json = json.loads(f.read())

    # Initialize lists to store extracted data
    user_ids = []
    news_ids = []
    scores = []

    # Iterate through the JSON data to extract user IDs, news IDs, and scores
    for user_id, news_data in tqdm(results_json.items()):
        for news_id, score in news_data.items():
            user_ids.append(user_id)
            news_ids.append(news_id)
            scores.append(score)

    # Construct a DataFrame from the extracted data
    results = pd.DataFrame({
        'user_id': user_ids,
        'news_id': news_ids,
        'score': scores
    })
    # Normalize recommendation scores
    results['score'] = normalize_scores(results['score'].tolist())


print('Getting news and embeddings...')
if dataset == 'MIND':
    news = pd.read_pickle('data/{}/news_info.pkl'.format(dataset))
    news = news[['NewsID_small', 'Category', 'Title', 'Embedding']].rename(
        columns={'NewsID_small': 'news_id', 'Category': 'category_name'})
    news = news[~news['news_id'].isna()]
    news = news[~news['Embedding'].isna()]
    #Define list of categories
    categories_list = ['lifestyle', 'health', 'news', 'sports', 'weather','entertainment','foodanddrink','autos','travel','video','tv','finance','movies','music','kids']
    #Only keep news from these categories
    news = news[news['category_name'].isin(categories_list)]
    #Create a dictionary that maps each unique category_name to a unique integer
    category_mapping = {category: i for i, category in enumerate(news['category_name'].unique())}
    # Apply the mapping to the 'category_name' column
    news['category'] = news['category_name'].map(category_mapping)

    news['Embedding'] = news['Embedding'].apply(ast.literal_eval)
    news_embeddings = embeddings_to_df(news['Embedding'])
    news_embeddings.index = news['news_id']

elif dataset =='ADRESSA':
    news = pd.read_csv('data/{}/news_adressa_emb.csv'.format(dataset), index_col=0)
    news = news[['nid','category','title','embeddings']].rename(columns={'nid':'news_id','category':'category_name','title':'Title','embeddings':'Embedding'})
    news = news[~news['news_id'].isna()]
    news = news[~news['Embedding'].isna()]
    #Rename the "100sport" category in "sport" to have one unique category corresponding to sport
    news['category_name'] = news['category_name'].replace('100sport', 'sport')
    #Define list of categories
    categories_list = ['nyheter', 'sport', 'forbruker', 'kultur', 'meninger','bolig','tema','tjenester','bil','migration catalog']
    #Only keep news from these categories
    news = news[news['category_name'].isin(categories_list)]
    # Create a dictionary that maps each unique category_name to a unique integer
    category_mapping = {category: i for i, category in enumerate(news['category_name'].unique())}
    # Apply the mapping to the 'category_name' column
    news['category'] = news['category_name'].map(category_mapping)
    news['Embedding'] = news['Embedding'].apply(ast.literal_eval)
    news_embeddings = embeddings_to_df(news['Embedding'])
    news_embeddings.index = news['news_id']


#Filter the news that are recommended
news_embeddings_filtered = news_embeddings[news_embeddings.index.isin(list(set(results['news_id'].unique().tolist()) & set(news['news_id'].unique().tolist())))]
news_embeddings_filtered = news_embeddings_filtered.reset_index().drop_duplicates(subset='news_id').set_index('news_id')
print('Done !')

results = results[results['news_id'].isin(list(set(results['news_id'].unique().tolist()) & set(news['news_id'].unique().tolist())))]
results = results.reset_index(drop=True)

print('Getting diversity matrix...')
similarity_matrix = pd.DataFrame(cosine_similarity(news_embeddings_filtered))
similarity_matrix.index = news_embeddings_filtered.index.tolist()
similarity_matrix.columns = news_embeddings_filtered.index.tolist()

diversity_matrix = 1 - similarity_matrix
print('Done !')
# Import behaviors data (train set)
behaviors = pd.read_csv('data/{}/behaviors.csv'.format(dataset), index_col=0)
behaviors = behaviors.rename(columns={'UserID':'user_id', 'NewsID':'news_id','Score':'score'})
#Merge behaviors and news data to have the aspect(s) of news in the interactions data (the aspect on which ADF brings diversity)
train = behaviors.merge(news[['news_id','category']], on='news_id')


# #Example if aspect = news category, must be adapted to the research context
# categories = train['category'].unique().tolist()
# categories.sort()

#Get the list of users from the results dataframe
#list_users = train['user_id'].unique().tolist()
#only keeping users having at least 20 interactions

# Original code : bug (keyerror : user_id)
# list_users = pd.DataFrame(behaviors['user_id'].value_counts())[pd.DataFrame(behaviors['user_id'].value_counts())['user_id']>=20].index.unique().tolist()
# Fix :
user_counts = behaviors['user_id'].value_counts()

list_users = pd.read_csv('list_users_{}.csv'.format(dataset))['0'].tolist()
print('Nb of users:',len(list_users))


results = results[results['user_id'].isin(list_users)]
results = results.reset_index(drop=True)

def get_df_by_user(df, onlyPosScores=False) :
    dict = df.to_dict()
    res = {}
    for i in tqdm(range(len(dict['user_id']))) :
        userid = dict['user_id'][i]
        if not userid in res :
            res[userid] = {'user_id' : [], 'news_id' : [], 'score' : []}
            #res[userid] = pd.DataFrame(columns=['news_id','score','category'])
        #res[userid].loc[len(res)] = {'news_id' : dict['news_id'][i], 'score' : dict['score'][i], 'category' : dict['category'][i]}
        if (onlyPosScores and dict['score'][i]) or not onlyPosScores :
            res[userid]['user_id'].append(userid)
            res[userid]['news_id'].append(dict['news_id'][i])
            res[userid]['score'].append(dict['score'][i])

    for u in tqdm(res) :
        res[u] = pd.DataFrame(res[u])
    return res


results_df = get_df_by_user(results)
list_users = results['user_id'].unique().tolist()



def triangular_matrix(m):
    m_tri = m.where(np.triu(np.ones(m.shape),k=1).astype(bool))
    return m_tri

def ILD(m):
    m_tri = triangular_matrix(m).stack().reset_index()
    m_tri.columns = ['i','j','dissimilarity']
    ild = (m_tri['dissimilarity'].sum())/len(m_tri)
    return ild

def greedy_reranking(recos, users_list, diversity_matrix, alpha, k=20):
    final_results = pd.DataFrame()
    for u in tqdm(users_list):
        recos_user = recos[u]
        recos_user = recos_user[:100]
        recos_list_user = recos_user['news_id'].tolist()
        # pertinence_scores = recos_user['score'].tolist()
        diversity_matrix_user = diversity_matrix[diversity_matrix.index.isin(recos_list_user)][recos_list_user]
        
        selected_items = []
        remaining_items = recos_user['news_id'].tolist()

        while len(selected_items) < k and remaining_items:
            best_item = None
            best_score = -1
            for candidate in remaining_items:
                accuracy = recos_user[recos_user['news_id']==candidate]['score'].values[0]
                temporary_list = selected_items.copy()
                temporary_list.append(candidate)
                temporary_list_sorted = temporary_list.copy()
                temporary_list_sorted.sort()
                # div_score = np.mean([diversity_matrix_user[candidate, s] for s in selected_items]) if selected_items else 0
                div_score = ILD(diversity_matrix_user[diversity_matrix_user.index.isin(temporary_list)][temporary_list_sorted]) if selected_items else 0 
                score = ((1-alpha)*accuracy)+(alpha*div_score) 
                # print('taille liste:', len(selected_items), 'taille liste temp',len(temporary_list), 'candidate news', candidate, 'accuracy',accuracy, 'diversity',div_score, 'score total',score)
                if score > best_score:
                    best_score=score
                    best_item=candidate
            if best_score >= 0:
                selected_items.append(best_item)
                remaining_items.remove(best_item)
            else:
                break
        recos_user_final = recos_user[recos_user['news_id'].isin(selected_items)]
        recos_user_final = recos_user_final.drop_duplicates(subset=['news_id'])
        recos_user_final.set_index('news_id', inplace=True)
        recos_user_final = recos_user_final.reindex(selected_items)
        recos_user_final.reset_index(inplace=True)
        recos_user_final = recos_user_final[['user_id','news_id','score']]
        final_results = pd.concat([final_results, recos_user_final])
    # final_results['news_id'] = final_results['news_id'].astype(int)
    return final_results


if not os.path.exists('reco_scores_greedy/{}/{}/'.format(dataset, algorithm)) :
    os.mkdir('reco_scores_greedy/{}/{}/'.format(dataset, algorithm))


print('Applying greedy...')
results = greedy_reranking(results_df, list_users, diversity_matrix, alpha_value, k=20)
print('greedy done!')
# path = results/dkn_MIND_RESULTS_a_01_
# algorithm_name = datapath.split("/")[-1].split("_")[1]+"_"+datapath.split("/")[-1].split("_")[2].split(".")[0]
# data_path = 'data_results/'+algorithm_name+'_RESULTS_a_'+str(a)+'_ADF.csv'
print('Saving results...')
results.to_csv('reco_scores_greedy/{}/{}/{}_{}_RESULTS_lambda_{}_ADF.csv'.format(dataset, algorithm, algorithm, dataset, str(alpha_value).replace('.','')))
print('Results saved!')
print('DONE ! Greedy re-ranking applied on {} for the {} algorithm, with lambda={}.'.format(dataset, algorithm, alpha_value))



