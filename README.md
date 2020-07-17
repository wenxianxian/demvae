# Dispersed Exponential Family Mixture VAEs for Interpretable Text Generation
Codebase for [Dispersed Exponential Family Mixture VAEs for Interpretable Text Generation](https://arxiv.org/abs/1906.06719).

This codebase is built based on [NeuralDialog-LAED](https://github.com/snakeztc/NeuralDialog-LAED) from Tiancheng Zhao.

## Requirements
    python 2.7
    pytorch >= 0.3.0.post4
    sklearn
    nltk

## Datasets
The *data* folder contains three datasets:
- [PennTree Bank](https://github.com/townie/PTB-dataset-from-Tomas-Mikolov-s-webpage/tree/master/data): sentence data
- [Daily Dialog](https://arxiv.org/abs/1710.03957): human-human open domain chatting.
- [Stanford Multi-domain Dialog](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/): human-woz task-oriented dialogs.

The *data/word2vec* includes [GloVe](https://nlp.stanford.edu/projects/glove/.) word embeddings filtered by the words in training sets.


## Training

### Language Generation

You can run the following command to train a dispersed GM-VAE model on PTB:
    
    LOG_DIR="logs/ptb/dgmvae"
    python main_lm.py --model GMVAE --log_dir $LOG_DIR --beta 0.2

You can use `--use_mutual True` to add the mutual information term in objective.

### Interpretable Text Generation

#### Unsupervised text generation by dispersed Gaussian Mixture VAE (DGM-VAE)

You can run the following command to train a dispersed GM-VAE model on DD and evaluate the interpretability by homogeneity:
    
    LOG_DIR="logs/dd/gmvae"
    python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --latent_size 5 --model GMVAE --log_dir $LOG_DIR --beta 0.3 --use_mutual True --post_sample_num 1 --sel_metric obj --lr_decay False

#### Supervised text generation by dispersed Categorical Mixture VAE (DCM-VAE)

You can run the following command to train a supervised dispersed CM-VAE model on DD and evaluate the interpretability by accuracy:
    
    LOG_DIR="logs/dd_sup/bmvae"
    python main_supervised.py --data daily_dialog --data_dir data/daily_dialog --model BMVAE --log_dir $LOG_DIR --beta 0.6


### Dialog Generation

You can run the following command to train a dispersed GM-VAE model on SMD for dialog generation:

    LOG_DIR="logs/smd/dgmvae"
    python main_stanford.py --data stanford --data_dir data/stanford --model AeED_GMM --log_dir $LOG_DIR --use_mutual True --beta 0.5 --freeze_step 7000


More examples of running baseline models could be found in `scripts/test.sh`.

## Evaluation

### Test a existing model

To run an existing model, you can:

- Set the `--forward_only` argument to be `True`
- Set the `--load_sess` argument to the path of the model folder in *LOG_DIR*
- Run the script 

Metrics such as BLEU and negative log-likelihood are calculated by running this script.


### Test reverse perplexity

To test the reverse perplexity, you need to train a third-party language model in the synthetic training set and test in the real test set. 

For example, you could use the [awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm) as the third-party language model:

Firstly, run the following scripts to split the generated sentences into training and validation sets, and copy the real test set.

    MODEL_DIR="logs/ptb/dgmvae/xxx-main_lm.py"
    python scripts/split_sampling_corpus.py --model_dir $MODEL_DIR

The training, validation and test sets are saved in the `reverse_PPL` directory under `MODEL_DIR`.

Secondly, train language model (for example, the awd-lstm-lm) in the synthetic dataset:

    output_data_dir=$MODEL_DIR"/reverse_PPL"
    python awd-lstm-lm/main.py --batch_size 20 --data $output_data_dir --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 20 --save PTB.pt > $ouput_result_path

You can use other language models, just replacing the training and validation sets by the synthetic data.

### Test word-level KL divergence

You can run the following script to evaluate the word-level KL divergence between the synthetic set and the real training set:


    MODEL_DIR="logs/ptb/dgmvae/xxx-main_lm.py"
    python scripts/test_wKL.py --model_dir $MODEL_DIR --data_dir data/ptb
    

