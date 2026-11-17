import os
import yaml
import torch
import argparse
import pandas as pd

from preprocess.preprocess import (
    preprocess_df,
    train_val_test_split,
    get_fraud_features,
    text2emb_bertbase,
    get_graphs
)

from methods.evaluate import evaluate


def main(args_main):

    # ============================================================
    # 1. LOAD CONFIG
    # ============================================================

    with open(args_main.config) as file:
        args = yaml.safe_load(file)

    if args_main.device != 'config':
        args['device'] = args_main.device

    print(
        'train_ratio:',
        (1 - args['test_size']) *
        (1 - args['val_size'])
    )

    print(
        'val_ratio:',
        (1 - args['test_size']) *
        args['val_size']
    )

    print(
        'test_ratio:',
        args['test_size']
    )

    print('Training arguments:')

    for k, v in args.items():
        print(f'{k}: {v}')

    print(
        f'Genuine review file: '
        f'{args_main.genuine_review_file}'
    )

    print(
        f'Spam review file: '
        f'{args_main.spam_review_file}'
    )

    # ============================================================
    # 2. LOAD DATA
    # ============================================================

    df_genuine = pd.read_csv(
        os.path.join(
            args_main.data_dir,
            args_main.genuine_review_file
        )
    )

    # Genuine = 0
    df_genuine['label'] = 0

    df_spam = pd.read_csv(
        os.path.join(
            args_main.data_dir,
            args_main.spam_review_file
        )
    )

    # Spam dataset already contains label 1
    df = pd.concat(
        [
            df_genuine,
            df_spam
        ],
        axis=0
    )

    df.index = range(df.shape[0])

    # ============================================================
    # 3. PREPROCESS
    # ============================================================

    df = preprocess_df(df)

    print(
        'Dataset shapes:',
        df.shape,
        df_genuine.shape,
        df_spam.shape
    )

    # ============================================================
    # 4. PAPER-INSPIRED WRITING-EFFORT FEATURE
    #
    # Review length is used as a proxy for writing effort.
    # Reviews are divided into four length quartiles.
    # ============================================================

    print(
        '\nCreating review-length feature...'
    )

    df['review_length'] = (
        df['reviewText']
        .astype(str)
        .str.split()
        .str.len()
    )

    df['length_quartile'] = pd.qcut(
        df['review_length'].rank(
            method='first'
        ),
        q=4,
        labels=[
            'Q1',
            'Q2',
            'Q3',
            'Q4'
        ]
    )

    print(
        '\nLength quartile distribution:'
    )

    print(
        df[
            'length_quartile'
        ]
        .value_counts()
        .sort_index()
    )

    # ============================================================
    # 5. CREATE SAFE PRODUCT + LENGTH RELATION
    #
    # IMPORTANT:
    # We DO NOT connect every review globally by length quartile.
    #
    # That created enormous groups (~22k reviews each).
    #
    # Instead:
    #
    # product + length quartile
    #
    # Example:
    #
    # product123_Q1
    # product123_Q2
    # product456_Q1
    #
    # This creates much smaller graph groups.
    # ============================================================

    print(
        '\nCreating product-length relationship...'
    )

    df['product_length'] = (
        df['product'].astype(str)
        + '_'
        + df[
            'length_quartile'
        ].astype(str)
    )

    print(
        'Number of product-length groups:',
        df['product_length'].nunique()
    )

    # ============================================================
    # 6. BERT EMBEDDINGS
    # ============================================================

    review_texts = (
        df['reviewText'].values
    )

    review_ids, review_embs = (
        text2emb_bertbase(
            review_texts,
            args['device']
        )
    )

    # Keep successfully processed reviews
    df = df.loc[
        review_ids,
        :
    ].copy()

    review_texts = (
        df['reviewText'].values
    )

    review_n = df.shape[0]

    df.index = range(
        review_n
    )

    print(
        '\nReviews after BERT:',
        df.shape
    )

    # ============================================================
    # 7. LABELS
    # ============================================================

    labels = (
        df.label.values
    )

    print(
        '\nLabel counts:'
    )

    print(
        df.label.value_counts()
    )

    # ============================================================
    # 8. TRAIN / VALIDATION / TEST SPLIT
    # ============================================================

    train_mask, val_mask, test_mask = (
        train_val_test_split(
            labels,
            args
        )
    )

    # ============================================================
    # 9. GRAPH CONSTRUCTION
    #
    # Original FraudSquad relations:
    #
    # user
    # product_star
    # product_month
    #
    # Added relation:
    #
    # product_length
    #
    # ============================================================

    print(
        '\nConstructing graphs...'
    )

    graphs, adjs, graph_homo = (
        get_graphs(
            df,
            connect_columns=[
                'user',
                'product_star',
                'product_month',
                'product_length'
            ],
            self_loop=True
        )
    )

    # ============================================================
    # 10. EXISTING FRAUD FEATURES
    # ============================================================

    other_features = (
        get_fraud_features(
            df,
            review_texts
        )
    )

    # ============================================================
    # 11. DEVICE
    # ============================================================

    graph_homo = (
        graph_homo.to(
            args['device']
        )
    )

    labels = torch.tensor(
        labels.copy(),
        dtype=torch.long,
        device=args['device']
    )

    print(
        '\nLabels: '
        'num 0, num 1, ratio 1, total'
    )

    print(
        (labels == 0).sum(),
        (labels == 1).sum(),
        (labels == 1).sum()
        / review_n,
        labels.shape
    )

    # ============================================================
    # 12. FRAUDSQUAD MODEL
    # ============================================================

    from methods.fraudsquad_src.model_handler import (
        fraudsquad_handler
    )

    print(
        '\nStarting FraudSquad...'
    )

    (
        test_score,
        test_auc,
        f1,
        ap,
        val_auc
    ) = fraudsquad_handler(
        review_texts.tolist(),
        graph_homo,
        train_mask,
        val_mask,
        test_mask,
        labels.cpu().numpy(),
        args
    )

    # ============================================================
    # 13. FINAL EVALUATION
    # ============================================================

    (
        auc,
        ap,
        precision,
        recall,
        f1
    ) = evaluate(
        test_score,
        labels.cpu().numpy()[
            test_mask
        ],
        args[
            'top_ratio'
        ]
    )

    print()
    print(
        '=================================='
    )

    print(
        'FINAL FRAUDSQUAD RESULTS'
    )

    print(
        '=================================='
    )

    print(
        'Fraudsquad AUC:',
        auc
    )

    print(
        'Fraudsquad AP:',
        ap
    )

    print(
        'Fraudsquad Precision:',
        precision
    )

    print(
        'Fraudsquad Recall:',
        recall
    )

    print(
        'Fraudsquad F1:',
        f1
    )


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':

    arg_parser = (
        argparse.ArgumentParser()
    )

    arg_parser.add_argument(
        '--config',
        type=str,
        default=(
            'config/'
            'amazon_cfg_0503.yml'
        )
    )

    arg_parser.add_argument(
        '--data_dir',
        type=str,
        default='data/Amazon'
    )

    arg_parser.add_argument(
        '--genuine_review_file',
        type=str,
        default=(
            'amazon_2022_5core.csv'
        )
    )

    arg_parser.add_argument(
        '--spam_review_file',
        type=str,
        default=(
            'reviews_from_qwen_dsr1.csv'
        )
    )

    arg_parser.add_argument(
        '--device',
        type=str,
        default='config'
    )

    args_main = (
        arg_parser.parse_args()
    )

    main(args_main)