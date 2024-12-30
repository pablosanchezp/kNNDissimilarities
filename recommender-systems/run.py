################################## IMPORTS ##################################
from Recommenders.RP3betaRecommender import RP3betaRecommender
from Utils.Evaluator import EvaluatorHoldout
from Utils.DataSplitter import DataSplitter
from Utils.DataReader import DataReader
from hybrid import Hybrid
from tqdm import tqdm

################################# READ DATA #################################
reader = DataReader()
splitter = DataSplitter()
urm = reader.load_urm()
ICM = reader.load_icm()
targets = reader.load_target()

URM_train, URM_val, URM_test = splitter.split(urm, validation=0.2, testing=0.2)

####################### ISTANTIATE AND FIT THE HYBRID #######################

recommender = RP3betaRecommender(URM_train)
recommender.fit()

################################ PRODUCE CSV ################################

f = open("submission.csv", "w+")
f.write("user_id,item_list\n")
for t in tqdm(targets):
    recommended_items = recommender.recommend(t, cutoff=10, remove_seen_flag=True, return_scores=True)
    for item, score in zip(recommended_items[0], recommended_items[1]):
        f.write(f"{t} \t {item} \t {score}\n")  # Formato: 'item: score'
