from __future__ import print_function

import logging
import os
import json

from dgmvae import evaluators, utt_utils
from dgmvae import main as main_train
from dgmvae import main_aggresive as main_train_agg
from dgmvae.dataset import corpora
from dgmvae.dataset import data_loaders
from dgmvae.models.sup_models import *
from dgmvae.utils import prepare_dirs_loggers, get_time
from dgmvae.multi_bleu import multi_bleu_perl
from dgmvae.options import get_parser

logger = logging.getLogger()

def get_corpus_client(config):
    if config.data.lower() == "ptb":
        corpus_client = corpora.PTBCorpus(config)
    elif config.data.lower() == "daily_dialog":
        corpus_client = corpora.DailyDialogCorpus(config)
    elif config.data.lower() == "stanford":
        corpus_client = corpora.StanfordCorpus(config)
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")
    return corpus_client

def get_dataloader(config, corpus):
    if config.data.lower() == "ptb":
        dataloader = data_loaders.PTBDataLoader
    elif config.data.lower() == "daily_dialog":
        dataloader = data_loaders.DailyDialogSkipLoaderLabel
    elif config.data.lower() == "stanford":
        dataloader = data_loaders.SMDDataLoader
    else:
        raise ValueError("Only support three corpus: ptb, daily_dialog and stanford.")

    train_dial, valid_dial, test_dial = corpus['train'], \
                                        corpus['valid'], \
                                        corpus['test']

    train_feed = dataloader("Train", train_dial, config)
    valid_feed = dataloader("Valid", valid_dial, config)
    test_feed = dataloader("Test", test_dial, config)

    return train_feed, valid_feed, test_feed

def get_model(corpus_client, config):
    try:
        model = eval(config.model)(corpus_client, config)
    except Exception as e:
        raise NotImplementedError("Fail to build model %s" % (config.model))
    if config.use_gpu:
        model.cuda()
    return model

def evaluation(model, test_feed, train_feed, evaluator):
    if config.aggressive:
        engine = main_train_agg
    else:
        engine = main_train
        
    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
        sampling_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-sampling.txt".format(get_time()))
    else:
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.session_dir, "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")
        sampling_file = os.path.join(config.session_dir, "{}-sampling.txt".format(get_time()))

    config.batch_size = 50
    model.load_state_dict(torch.load(model_file))

    engine.validate(model, test_feed, config)
    # engine.validate(model, valid_feed, config)

    if hasattr(model, "sampling_for_likelihood"):
        nll = utt_utils.calculate_likelihood(model, test_feed, 500, config)  # must
        if config.forward_only:
            logger_file = open(os.path.join(config.log_dir, config.load_sess, "session.log"), "a")
            logger_file.write("Log-likehood %lf" % nll)

    print("--test homogeneity--")
    utt_utils.find_mi(model, test_feed, config)  # homogeneity_score

    # with open(os.path.join(sampling_file), "w") as f:
    #     print("Saving test to {}".format(sampling_file))
    #     utt_utils.exact_sampling(model, 46000, config, dest_f=f)

    # selected_clusters = utt_utils.latent_cluster(model, train_feed, config, num_batch=None)

    # with open(os.path.join(dump_file + '.json'), 'w') as f:
    #    json.dump(selected_clusters, f, indent=2)

    with open(os.path.join(dump_file), "wb") as f:
        print("Dumping test to {}".format(dump_file))
        utt_utils.dump_latent(model, test_feed, config, f, num_batch=None)

    with open(os.path.join(test_file), "w") as f:
        print("Saving test to {}".format(test_file))
        utt_utils.generate(model, test_feed, config, evaluator, num_batch=None, dest_f=f)

    multi_bleu_perl(config.session_dir if not config.forward_only else os.path.join(config.log_dir, config.load_sess))

def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))

    corpus_client = get_corpus_client(config)
    dial_corpus = corpus_client.get_corpus()
    evaluator = evaluators.BleuEvaluator("CornellMovie")
    train_feed, valid_feed, test_feed = get_dataloader(config, dial_corpus)
    model = get_model(corpus_client, config)

    if config.forward_only is False:
        try:
            engine = main_train
            engine.train(model, train_feed, valid_feed,
                         test_feed, config, evaluator, gen=utt_utils.generate)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")
    evaluation(model, test_feed, train_feed, evaluator)

if __name__ == "__main__":
    config = get_parser("sup_models")
    with torch.cuda.device(config.gpu_idx):
        main(config)
