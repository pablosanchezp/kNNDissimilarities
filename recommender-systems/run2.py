################################## IMPORTS ##################################

from Utils.Evaluator import EvaluatorHoldout
from Utils.DataSplitter import DataSplitter
from Utils.DataReader import DataReader
from hybrid import Hybrid
from Recommenders import RP3betaRecommender
from tqdm import tqdm
import scipy.sparse as sps
import argparse
import pandas as pd

################################# READ DATA #################################


if __name__ == "__main__":
    # Crear el objeto ArgumentParser
    parser = argparse.ArgumentParser(description="Process some files and parameters.")

    # Agregar los argumentos que se van a recibir
    parser.add_argument("--training", type=str, help="Path of the training file.")
    parser.add_argument("--test", type=str, help="Path of the test file.")
    parser.add_argument("--nI", type=int, help="Number of items.")
    parser.add_argument("--result", type=str, help="Path of the result file.")
    parser.add_argument("--implicit", type=bool, help="Boolean for implicit.")
    parser.add_argument("--alpha", type=float, help="Alpha parameter.", required=True)
    parser.add_argument("--beta", type=float, help="Beta parameter.", required=True)

    # Parsear los argumentos
    args = parser.parse_args()

    # Imprimir los argumentos para verificar que se recibieron correctamente
    print(f"Training file: {args.training}")
    print(f"Test file: {args.test}")
    print(f"Number of items: {args.nI}")
    print(f"Result file: {args.result}")
    print(f"Implicit: {args.implicit}")
    print(f"Alpha: {args.alpha}")
    print(f"Beta: {args.beta}")

    reader = DataReader()
    # splitter = DataSplitter()

    urm = reader.load_urm(filepath=args.training, sep="\t")
    #targets = reader.load_target()

    user_map = {user_id: i for i, user_id in enumerate(urm['user_id'].unique())}
    item_map = {item_id: i for i, item_id in enumerate(urm['item_id'].unique())}

    reverse_user_map = {v: k for k, v in user_map.items()}
    reverse_item_map = {v: k for k, v in item_map.items()}

    # Paso 2: Mapear los IDs originales a los índices consecutivos
    urm['user_idx'] = urm['user_id'].map(user_map)
    urm['item_idx'] = urm['item_id'].map(item_map)

    # Paso 3: Crear la csr_matrix
    ratings_matrix = sps.csr_matrix((urm['rating'], (urm['user_idx'], urm['item_idx'])))

    df_test = pd.read_csv(args.test, header=None, sep="\t")

    # Obtener los valores únicos de la primera columna (indexada como 0)
    unique_users_test = df_test.iloc[:, 0].unique()
    users_test_appearing_training = [val for val in unique_users_test if val in user_map]

    # URM_train, URM_val, URM_test = splitter.split(urm, validation=0.1, testing=0.1)

    ####################### ISTANTIATE AND FIT THE HYBRID #######################

    recommender = RP3betaRecommender.RP3betaRecommender(ratings_matrix)

    recommender.fit(alpha=args.alpha, beta=args.beta, min_rating=urm['rating'].min(), topK=args.nI, implicit=args.implicit,
                    normalize_similarity=True)

    ################################ PRODUCE CSV ################################

    f = open(args.result, "w")
    for t in tqdm(users_test_appearing_training):
        recommended_items = recommender.recommend(user_map[t], cutoff=args.nI, remove_seen_flag=True, return_scores=True)
        for item, score in zip(recommended_items[0], recommended_items[1]):
            f.write(f"{t}\t{reverse_item_map[item]}\t{score}\n")  # Formato: 'item: score'
