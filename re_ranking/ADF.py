import warnings
warnings.filterwarnings("ignore")

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
        alpha_value = int(arg)/10


print(f'\ADF parameters:')
print(f"\tDataset: {dataset}")
print(f"\tAlgorithm: {algorithm}")
print(f"\tAlpha value: {alpha_value}")

print(os.getcwd())


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = []
    for score in scores:
        normalized_score = (score - min_score) / (max_score - min_score)
        normalized_scores.append(normalized_score)
    return normalized_scores


if algorithm=='centroidVector': #CSV file
    datapath='reco_scores_baseline/{}/recommendations_{}_{}.csv'.format(dataset,algorithm,dataset)
    results = pd.read_csv(datapath, index_col=0)    
else: #JSON file
    datapath = 'reco_scores_baseline/{}/recommendations_{}_{}.json'.format(dataset,algorithm,dataset)
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


# Import behaviors data (train set)
behaviors = pd.read_csv('data/{}/behaviors.csv'.format(dataset), index_col=0)

behaviors = behaviors.rename(columns={'UserID':'user_id', 'NewsID':'news_id','Score':'score'})

#Merge behaviors and news data to have the aspect(s) of news in the interactions data (the aspect on which ADF brings diversity)
train = behaviors.merge(news[['news_id','category']], on='news_id')


#Example if aspect = news category, must be adapted to the research context
categories = train['category'].unique().tolist()
categories.sort()

# Get the list of users from the results dataframe
# list_users = train['user_id'].unique().tolist()




def get_df_by_user(df, onlyPosScores=False) :
    dict = df.to_dict()
    res = {}
    for i in tqdm(range(len(dict['user_id']))) :
        userid = dict['user_id'][i]
        if not userid in res :
            res[userid] = {'user_id' : [], 'news_id' : [], 'score' : [], 'category' : []}
            #res[userid] = pd.DataFrame(columns=['news_id','score','category'])
        #res[userid].loc[len(res)] = {'news_id' : dict['news_id'][i], 'score' : dict['score'][i], 'category' : dict['category'][i]}
        if (onlyPosScores and dict['score'][i]) or not onlyPosScores :
            res[userid]['user_id'].append(userid)
            res[userid]['news_id'].append(dict['news_id'][i])
            res[userid]['score'].append(dict['score'][i])
            res[userid]['category'].append(dict['category'][i])

    for u in tqdm(res) :
        res[u] = pd.DataFrame(res[u])
    return res

# behaviors = dataframe containing users' behaviors (interactions) in the train set, with the details about news' aspect
# list_aspects = list of unique aspects
# list_users = list of users' ids
# should consider the probability of each news for its category. This depends on the way categories are defined (by default = False).

def get_selection_distribution(list_aspects, list_users, train_per_user):
    categories_distribution = pd.DataFrame(columns=list_aspects, index=list_users)
    for u in tqdm(list_users):
        user_df = train_per_user[u]
        for c in list_aspects:
            nbElements = user_df['category'].value_counts()[c] if c in user_df['category'].value_counts() else 0
            categories_distribution.loc[u,c] = nbElements/len(user_df['category'])
    return categories_distribution


list_users = pd.read_csv('list_users_{}.csv'.format(dataset))['0'].tolist()
print('Nb of users:',len(list_users))



print('trainDict & train_per_user initialization')
start_time = time.time()
train_per_user = get_df_by_user(train, True)
print("--- took a total of %s seconds ---" % (time.time() - start_time))

try :
    print("Trying to read data/categories_distribution.csv...")
    categories_distribution = pd.read_csv("data/{}/categories_distribution.csv".format(dataset), header=[0], index_col=[0])
    print("Successful. If you changed anything related to the dataset, "
    + "you should delete data/{}/categories_distribution.csv to generate a new one".format(dataset))
except FileNotFoundError:
    print("Couldn't read data/{}/categories_distribution.csv, creating it".format(dataset))
    categories_distribution = get_selection_distribution(categories, list_users, train_per_user)
    categories_distribution.to_csv("data/{}/categories_distribution.csv".format(dataset))



def normalized_entropy(distribution):
    return entropy(distribution, base=2) / np.log2(len(distribution))

def compute_entropy_users(list_users, categories_distribution):
    u_values = pd.DataFrame(index=list_users)
    for u in tqdm(list_users):
        distrib_user = categories_distribution.loc[u].values.tolist()
        entropy_value = normalized_entropy(distrib_user)
        u_values.loc[u, 'entropy'] = entropy_value
    return u_values


user_entropy_values = compute_entropy_users(list_users, categories_distribution)


# Function allowing to estimate the target diversity based on a and b parameters
def div(x, a, b):
    return (1 - b) * (x ** (1 - a)) + b


# Function allowing to get the homogenized distribution (target distribution)
def homogeneization(distrib, param=0.5):
    n = len(distrib)
    new_distrib = [((1 - param) * p) + (param / n) for p in distrib]
    return new_distrib


# Function to have the results dataframe with corresponding news categories
def get_results_categories(initial_results):
    results_categories = initial_results.copy()
    results_categories = results_categories.merge(news[['news_id', 'category']], on='news_id')
    return results_categories


results_categories = get_results_categories(results)

results_df = get_df_by_user(results_categories)


# PARAMETERS:
# df = dataframe with results
# entropy_df = dataframe with entropy values for each user
# list_users = list of users ids
# proportions_df = distribution of users interactions over news categories
# a,b = values of parameters a and b of ADF
# news = dataframe containing news information (especially the categories)
# behaviors = dataframe contaning users' behaviors
# k = number of recommendations
def ADF(df, entropy_df, list_users, proportions_df, a, b, news, train_per_user, k=20):
    results_entropy = entropy_df.copy()
    final_results = pd.DataFrame()
    for u in tqdm(list_users):
        ##GET NECESSARY DATA
        # get user's interactions
        accessed_news = train_per_user[u]['news_id'].tolist()
        # get the recommendations results for the user
        data = df[u]
        # get list of categories recommended to a user
        list_recommended_categories = data['category'].unique().tolist()
        # get the distribution of user's interest over news categories
        proportions_user = proportions_df.loc[u].tolist()
        # get user's entropy value
        entropy_user = results_entropy.loc[u].values[0]

        ##DEFINE THE TARGET ENTROPY (diversification)
        target_entropy = div(entropy_user, a, b)
        # Save the target entropy in a specific dataframe
        results_entropy.loc[u, 'target_entropy'] = target_entropy
        # Create an empty dict to save the results
        dict_lambda_user = {}
        # Here the distribution of users' preferences is homogenized using varying value of the lambda parameter
        # It allows to find the distribution allowing to have the entropy value as close as possible to the target entropy
        for l in np.arange(0, 1.05, 0.05):
            # get the homogenized distribution with a specific lambda value
            new_distrib = homogeneization(proportions_user, param=round(l, 2))
            # compute associated entropy
            new_entropy = normalized_entropy(new_distrib)
            # get the error between the computed entropy and the target entropy
            delta = target_entropy - new_entropy
            # save this value in the dict
            dict_key = {round(l, 2): abs(delta)}
            dict_lambda_user.update(dict_key)
        # identify the optimal lambda value by searching for the one allowing to have the smallest error
        optimal_lambda = min(dict_lambda_user, key=dict_lambda_user.get)
        # save this optimal lambsa value in the associated dataframe
        results_entropy.loc[u, 'optimal_lambda'] = optimal_lambda

        ##GET THE RECOMMENDATIONS
        # Once the optimal lambda value is knows, the target distribution can be computed from user's initial distribution
        new_proportion = homogeneization(proportions_user, param=optimal_lambda)
        # Recommendations are then retrieved for each news category
        for c in range(len(new_proportion)):
            prop = new_proportion[c]
            # Compute the number of news to recommend in category c (according to the target distribution)
            nb_recos = int(round(prop * k, 0))
            # If the category is recommended to user u, select the news having higher recommendation scores
            if c in list_recommended_categories:
                data_sub = data[data['category'] == c].sort_values(by='score', ascending=False)[:nb_recos]
            # else, choose random news from the category
            else:
                data_sub = pd.DataFrame(columns=['user_id', 'news_id', 'score', 'category'])
                # here the news that were already accessed by a user are filtered
                news_to_select = news[~news['news_id'].isin(accessed_news)]
                cat_news = news_to_select[news_to_select['category'] == c]['news_id'].tolist()
                if len(cat_news) >= nb_recos:
                    selected_news = random.sample(cat_news, k=nb_recos)
                else:
                    selected_news = cat_news
                for n in selected_news:
                    new_row = {'user_id': u, 'news_id': n, 'score': 0, 'category': c}
                    # data_sub = data_sub.append(new_row, ignore_index=True)
                    data_sub.loc[len(data_sub)] = new_row
            data_sub['category'] = c
            # the recommendations are stored in the dataframe
            final_results = pd.concat([final_results, data_sub])
            final_results = final_results.reset_index(drop=True)
    # the results are returned: final_results contains the list of k recommendations per user, results_entropy contains the information about the target entropy and associated optimal lambda value for each user
    return final_results, results_entropy


if not os.path.exists('reco_scores_ADF/{}/{}/'.format(dataset, algorithm)) :
    os.mkdir('reco_scores_ADF/{}/{}/'.format(dataset, algorithm))


print('Applying ADF...')
results, entropy_results = ADF(results_df, user_entropy_values, list_users, categories_distribution, alpha_value, 0, news, train_per_user, k=20)
print('ADF done!')
# path = results/dkn_MIND_RESULTS_a_01_
# algorithm_name = datapath.split("/")[-1].split("_")[1]+"_"+datapath.split("/")[-1].split("_")[2].split(".")[0]
# data_path = 'data_results/'+algorithm_name+'_RESULTS_a_'+str(a)+'_ADF.csv'
print('Saving results...')
results.to_csv('reco_scores_ADF/{}/{}/{}_{}_RESULTS_a_{}_ADF.csv'.format(dataset, algorithm, algorithm, dataset, str(alpha_value).replace('.','')))
entropy_results.to_csv('reco_scores_ADF/{}/{}/{}_{}_RESULTS_indiv_a_{}_ADF.csv'.format(dataset, algorithm, algorithm, dataset, str(alpha_value).replace('.','')))
print('Results saved!')
print('DONE ! ADF applied on {} for the {} algorithm, with alpha={}.'.format(dataset, algorithm, alpha_value))
