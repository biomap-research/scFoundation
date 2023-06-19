import os.path
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('--emb', type=int, default=0, help='use emb')
parser.add_argument('--ckpt_name', type=str, default='50M-0.1B-res', help='ckpt path')
parser.add_argument('--drug', type=str, default='Etoposide', help='drug name')

args = parser.parse_args()

ckpt_name = args.ckpt_name

#Afatinib AR-42 Cetuximab Gefitinib NVP-TAE684 Sorafenib Vorinostat Etoposide PLX4720_451Lu

DRUG = args.drug  # Set to the name of the drug being split
os.chdir('../../')

if args.emb == 1:
    
    print(f'Using Embedding & Drug is {DRUG}')
    geneset=''  ### chose all genes, ppi genes, or top 4k most variant genes
    #geneset="_ppi"
    #geneset="_tp4k"

    Solid="False"  ###use GDSC solid tumor only

    SAVE_RESULT_TO_STRAT = './data/split_norm'+geneset+'/' + DRUG + '/stratified_emb/'
    SOURCE_DIR = 'source_5_folds'
    
    if Solid=="True":
        SOURCE_DIR="source_solid_5_folds"

    TARGET_DIR = 'target_5_folds'

    #######################################################################################################
    ## code refers to https://github.com/hosseinshn/AITL/blob/master/AITL_submit/data/split_data_aitl.py ##
    ## process and save files for model training                                                         ## 
    #######################################################################################################
    if Solid=="False":
        raw_source_exprs_resp_z = pd.read_csv('./data/split_norm/Source_exprs_resp_z.'+DRUG+geneset+'.tsv',sep='\t', index_col=0)
        source_exprs_resp_z = np.load('./data/split_norm/Source_exprs_resp_19264.'+DRUG+"_"+ckpt_name+'_embedding.npy')
        source_exprs_resp_z = pd.DataFrame(source_exprs_resp_z,index = raw_source_exprs_resp_z.index)
        source_exprs_resp_z = pd.concat([raw_source_exprs_resp_z.iloc[:,:2],source_exprs_resp_z],axis=1)
        
    ## if use solid tumor only
    if Solid=="True":
        
        raw_source_exprs_resp_z = pd.read_csv('./data/split_norm/Source_solid_exprs_resp_z.' + DRUG + geneset + '.tsv', sep='\t', index_col=0)
        source_exprs_resp_z = np.load('./data/split_norm/Source_solid_exprs_resp_19264.'+DRUG+"_"+ckpt_name+'_embedding.npy')
        source_exprs_resp_z = pd.DataFrame(source_exprs_resp_z,index = raw_source_exprs_resp_z.index)

    # Target data #
    raw_target_combined_exprs_resp_z = pd.read_csv('./data/split_norm/Target_expr_resp_z.'+DRUG+geneset+'.tsv', sep='\t', index_col=0)
    target_combined_exprs_resp_z = np.load('./data/split_norm/Target_expr_resp_19264.'+DRUG+"_"+ckpt_name+'_tgthighres4_embedding.npy')
    target_combined_exprs_resp_z = pd.DataFrame(target_combined_exprs_resp_z,index = raw_target_combined_exprs_resp_z.index)
    target_combined_exprs_resp_z = pd.concat([raw_target_combined_exprs_resp_z.iloc[:,0],target_combined_exprs_resp_z],axis=1)
    
    
else:
    geneset=''  ### chose all genes, ppi genes, or top 4k most variant genes
    #geneset="_ppi"
    #geneset="_tp4k"

    Solid="False"  ###use GDSC solid tumor only

    SAVE_RESULT_TO_STRAT = './data/split_norm'+geneset+'/' + DRUG + '/stratified/'
    SOURCE_DIR = 'source_5_folds'

    if Solid=="True":
        SOURCE_DIR="source_solid_5_folds"

    TARGET_DIR = 'target_5_folds'

    #######################################################################################################
    ## code refers to https://github.com/hosseinshn/AITL/blob/master/AITL_submit/data/split_data_aitl.py ##
    ## process and save files for model training                                                         ## 
    #######################################################################################################
    if Solid=="False":
        source_exprs_resp_z = pd.read_csv('./data/split_norm/Source_exprs_resp_z.'+DRUG+geneset+'.tsv',
                                    sep='\t', index_col=0)
    ## if use solid tumor only
    if Solid=="True":
        source_exprs_resp_z = pd.read_csv('./data/split_norm/Source_solid_exprs_resp_z.' + DRUG + geneset + '.tsv',
                                          sep='\t', index_col=0)

    # Target data #
    target_combined_exprs_resp_z = pd.read_csv('./data/split_norm/Target_expr_resp_z.'+DRUG+geneset+'.tsv',
                                    sep='\t', index_col=0)
    
def create_splits_stratified(orig_df, skf, splits_dict, dataset):
    """
    == For creating stratified splits for datasets ==
    :param orig_df - original dataframe (dataset) to be split
    :param skf - stratified kfold split function
    :param splits_dict - dictionary that holds the splits
    :param numfolds - number of folds
    :param dataset - source or target
    :param splitstype - indicates what kind of splits to perform (i.e. traintest or trainvaltest)
    """


    if dataset == "target":
        x_expression = orig_df.iloc[:, 1:]  # gene expressions (features)
        y_response = orig_df.iloc[:, 0]  # binary class labels (0 or 1)
        splitstypeds = ["traintest", "train", "test"]
        counter = 1
        for train_index, test_index in skf.split(x_expression, y_response):
            x_train, x_test = x_expression.iloc[train_index], x_expression.iloc[test_index]
            y_train, y_test = y_response.iloc[train_index], y_response.iloc[test_index]
            splits_dict["split" + str(counter)] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[1]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[1]]["X"] = x_train
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[1]]["Y"] = y_train
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[2]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[2]]["X"] = x_test
            splits_dict["split" + str(counter)][splitstypeds[0]][splitstypeds[2]]["Y"] = y_test
            counter += 1

    elif dataset == "source":
        x_expression = orig_df.iloc[:, 2:]  # gene expressions (features) 
        y_logIC50 = orig_df.iloc[:, 1] # col index 1 of the source df is logIC50
        y_response = orig_df.iloc[:, 0]  # binary class labels (0 or 1)
        print("# of class 0 examples in original (unsplit) source data: {}".format(len(y_response[y_response == 0])))
        print("# of class 1 examples in original (unsplit) source data: {}".format(len(y_response[y_response == 1])))
        splitstypeds = ["train", "val"]
        counter = 1
        for train_index, val_index in skf.split(x_expression, y_response):
            x_train, x_val = x_expression.iloc[train_index], x_expression.iloc[val_index]
            y_trainB, y_valB = y_response.iloc[train_index], y_response.iloc[val_index]
            y_trainIC50, y_valIC50 = y_logIC50.iloc[train_index], y_logIC50.iloc[val_index]
            splits_dict["split" + str(counter)] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]] = {}
            splits_dict["split" + str(counter)][splitstypeds[0]]["X"] = x_train
            splits_dict["split" + str(counter)][splitstypeds[0]]["Y_response"] = y_trainB
            splits_dict["split" + str(counter)][splitstypeds[0]]["Y_logIC50"] = y_trainIC50           
            splits_dict["split" + str(counter)][splitstypeds[1]] = {}
            splits_dict["split" + str(counter)][splitstypeds[1]]["X"] = x_val
            splits_dict["split" + str(counter)][splitstypeds[1]]["Y_response"] = y_valB
            splits_dict["split" + str(counter)][splitstypeds[1]]["Y_logIC50"] = y_valIC50
            counter += 1

def create_splits_sourceIC50(orig_df, kf, splits_dict):
    """
    == For creating numfolds-fold splits for source data with logIC50 as labels ==
    :param orig_df - original dataframe (dataset) to be split
    :param kf - KFold split function
    :param splits_dict - dictionary that holds the splits
    """
    # NOTE: for preprocessed source,
    #   - col index 0 is the binary class response (labels)
    #   - col index 1 is logIC50 (labels)
    #   - col index 2 and beyond is the gene expression (features)
    counter = 1
    for train_index, val_index in kf.split(orig_df):
        x_train, x_val = orig_df.iloc[train_index, 2:], orig_df.iloc[val_index, 2:] # gene expressions (features)
        y_train, y_val = orig_df.iloc[train_index, 1], orig_df.iloc[val_index, 1] # logIC50
        splits_dict["split" + str(counter)] = {}
        splits_dict["split" + str(counter)]["train"] = {}
        splits_dict["split" + str(counter)]["train"]["X"] = x_train
        splits_dict["split" + str(counter)]["train"]["Y"] = y_train
        splits_dict["split" + str(counter)]["val"] = {}
        splits_dict["split" + str(counter)]["val"]["X"] = x_val
        splits_dict["split" + str(counter)]["val"]["Y"] = y_val
        counter += 1


skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

splits_dict_target = {}
splits_dict_source = {} # perform regular kfold split on source (do not use binary labels)
create_splits_stratified(target_combined_exprs_resp_z, skf, splits_dict_target, "target")
create_splits_stratified(source_exprs_resp_z, skf, splits_dict_source, "source")

print("\n-- Counting the number of samples for each class from each fold of the stratified kfold split (TARGET)-- \n")
for split in splits_dict_target:
    print("\n{}".format(split))

    print("# training examples (class 0): {}".format(len(splits_dict_target[split]["traintest"]["train"]["Y"][splits_dict_target[split]["traintest"]["train"]["Y"] == 0])))
    print("# training examples (class 1): {}".format(len(splits_dict_target[split]["traintest"]["train"]["Y"][splits_dict_target[split]["traintest"]["train"]["Y"] == 1])))

    print("# test examples (class 0): {}".format(len(splits_dict_target[split]["traintest"]["test"]["Y"][splits_dict_target[split]["traintest"]["test"]["Y"] == 0])))
    print("# test examples (class 1): {}".format(len(splits_dict_target[split]["traintest"]["test"]["Y"][splits_dict_target[split]["traintest"]["test"]["Y"] == 1])))


print("\n-- Counting the number of samples from each fold of the 5-fold split (SOURCE)-- \n")
for split in splits_dict_source:
    print("\n{}".format(split))
    print("# training examples: {}, # features: {}".format(splits_dict_source[split]["train"]["X"].shape[0],
                                                                splits_dict_source[split]["train"]["X"].shape[1]))
    print("# class 0 training examples: {}".format(len(splits_dict_source[split]["train"]["Y_response"][splits_dict_source[split]["train"]["Y_response"] == 0])))
    print("# class 1 training examples: {}".format(len(splits_dict_source[split]["train"]["Y_response"][splits_dict_source[split]["train"]["Y_response"] == 1])))

    print("# validation examples: {}, # features: {}".format(splits_dict_source[split]["val"]["X"].shape[0],
                                                                    splits_dict_source[split]["val"]["X"].shape[1]))
    print("# class 0 validation examples: {}".format(len(splits_dict_source[split]["val"]["Y_response"][splits_dict_source[split]["val"]["Y_response"] == 0])))
    print("# class 1 validation examples: {}".format(len(splits_dict_source[split]["val"]["Y_response"][splits_dict_source[split]["val"]["Y_response"] == 1])))
                                                            


# Saving Source splits #
print("Saving source splits ...")
for split in splits_dict_source:
    # splits_dict_source[split]["train"]["X"]
    dirName = SAVE_RESULT_TO_STRAT + SOURCE_DIR + '/' + split + '/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    for train_val in splits_dict_source[split]:
        splits_dict_source[split][train_val]['X'].to_csv(path_or_buf=os.path.join(dirName, 'X_' + train_val + '_source.tsv'),sep='\t', index=True),
        splits_dict_source[split][train_val]['Y_response'].to_csv(path_or_buf=os.path.join(dirName, 'Y_' + train_val + '_source.tsv'),
                                                                sep='\t', index=True, header=True) # note: single col. pandas df is treated as Series
        splits_dict_source[split][train_val]['Y_logIC50'].to_csv(path_or_buf=os.path.join(dirName, 'Y_logIC50' + train_val + '_source.tsv'),
                                                                sep='\t', index=True, header=True) # note: single col. pandas df is treated as Series#

print("Successfully saved source splits.\n")

print("Saving target splits ...")
# Saving stratified Target splits #
for split in splits_dict_target:
    dirName = SAVE_RESULT_TO_STRAT + TARGET_DIR + '/' + split + '/'
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")
    for tvt in splits_dict_target[split]:

        if tvt == 'traintest':
            for traintest in splits_dict_target[split][tvt]:
                splits_dict_target[split][tvt][traintest]["X"].to_csv(path_or_buf=os.path.join(dirName, 'X_' + traintest + '_target.tsv'),
                                                                            sep='\t', index=True)
                splits_dict_target[split][tvt][traintest]["Y"].to_csv(path_or_buf=os.path.join(dirName, 'Y_' + traintest + '_target.tsv'),
                                                                            sep='\t', index=True, header=True)
print("Successfully saved target splits.\n")

