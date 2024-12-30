import pandas as pd
import argparse
from model import EASE


def main():
    parser = argparse.ArgumentParser(
        description="launch ease")
    parser.add_argument("--training", type=str, help="Path of the training file.")
    parser.add_argument("--test", type=str, help="Path of the test file.")
    parser.add_argument("--implicit", type=bool, help="Boolean for implicit.")
    parser.add_argument("--result", type=str, help="Path of the result file.")
    parser.add_argument("--lamb", type=float, help="lamda.")
    parser.add_argument("--nI", type=int, help="numberItems.")


    args = parser.parse_args()

    training_df = pd.read_csv(args.training, header=None, sep="\t")

    columns = training_df.columns.tolist()
    training_df.columns = ['user_id', 'item_id', 'rating'] + columns[3:]

    test_df = pd.read_csv(args.test, header=None, sep="\t")
    unique_users = test_df[0].unique()

    ease_rec = EASE()
    ease_rec.fit(training_df, args.lamb, args.implicit)
    df_result = ease_rec.predict(training_df, unique_users, training_df['item_id'].unique(), args.nI)
    df_result.to_csv(args.result, index=False, header=None, sep="\t")


if __name__ == "__main__":
    main()
