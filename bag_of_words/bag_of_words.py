import sys
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from utils.logger import get_console_logger
from utils.config import load_config

"""
Authors:
    Jinfeng Xiao (jxiao13@illinois.edu)
    Matthew Davis (matthew.davis@invitae.com)

Summary:
    Read papers and convert them to bag-of-words representations.
"""


def read_corpus(input_dir, **kwargs):
    """
    Read papers into a list.

    :param input_dir:
    :return:
    """
    corpus = {x: '' for x in os.listdir(input_dir)}
    logger.info("Reading corpus ...")
    for pmid in corpus:
        for field in os.listdir(input_dir + pmid + '/'):
            with open(input_dir + pmid + '/' + field, 'r') as fin:
                text = fin.read().replace('\n', ' ')
                corpus[pmid] += text
    return corpus


def tokenize_corpus(corpus, output_dir, tokenizer="tfidf", output_format="pkl", max_df=0.5, min_df=0.01, **kwargs):
    """
    Tokenize a corpus into a doc-by-token sparse matrix

    :param corpus:
    :param output_dir:
    :param tokenizer:
    :param output_format:
    :param max_df:
    :param min_df:
    """

    logger.info("Building bag-of-words representation ...")

    if tokenizer == "tf":
        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df)

    elif tokenizer == "tfidf":
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)

    else:
        logger.error("Invalid tokenizer.")
        sys.exit()

    # bow = vectorizer.fit_transform(list(corpus.values()))
    bow = vectorizer.fit_transform([corpus[x] for x in sorted(corpus)])

    if output_format == "pkl":
        out_fn = "%s/%s_%s.%s" % (output_dir, "bag_of_words", tokenizer, output_format)

        logger.info("Writing to %s ..." % out_fn)

        with open(out_fn, 'wb') as fout:
            pickle.dump(bow, fout)
            # bow_rownames = list(corpus.keys())
            bow_rownames = sorted(corpus)
            pickle.dump(bow_rownames, fout)
            bow_colnames = vectorizer.get_feature_names()
            pickle.dump(bow_colnames, fout)
    # elif output == "csv":
    #
    #TODO: add support for CSV files
    else:
        logger.error("Invalid output format.")
    logger.info("Done!")


if __name__ == "__main__":
    logger = get_console_logger(name="tokenize_corpus")
    config_d = load_config(os.path.abspath("config.yml"), 'bag_of_words')

    corpus = read_corpus(**config_d)
    tokenize_corpus(corpus, **config_d)
