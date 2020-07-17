#!/usr/bin/env bash

# === Test PTB ===

# RNNLM:
python main_lm.py --model RNNLM --log_dir logs/ptb/rnnlm
# VAE and AE:
python main_lm.py --model VAE --log_dir logs/ptb/vae
python main_lm.py --model VAE --log_dir logs/ptb/ae --use_kl False --post_sample_num 1
# Discrete VAE:
python main_lm.py --model DiVAE --log_dir logs2/ptb/dvae --use_kl True --use_mutual False --post_sample_num 1
python main_lm.py --model DiVAE --log_dir logs2/ptb/divae --use_kl True --use_mutual True --post_sample_num 1
# Semi-VAE
python main_lm.py --model SVAE --log_dir logs2/ptb/semi_vae --use_mutual False
python main_lm.py --model SVAE --log_dir logs2/ptb/semi_vae --use_mutual True
# GMVAE
python main_lm.py --model GMVAE --log_dir logs2/ptb/gmvae --beta 1.0 --use_mutual False
python main_lm.py --model GMVAE --log_dir logs2/ptb/gmvae --beta 1.0 --use_mutual True
python main_lm.py --model GMVAE --log_dir logs2/ptb/dgmvae --beta 0.2 --use_mutual False
python main_lm.py --model GMVAE --log_dir logs2/ptb/dgmvae --beta 0.2 --use_mutual True


# === Test DD ===
# Di-VAE
python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --model DiVAE --log_dir logs2/dd/divae --use_kl True --use_mutual True --post_sample_num 1
# Semi-VAE
python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --latent_size 5 --model SVAE --log_dir logs2/dd/semi_vae --use_mutual True --post_sample_num 1
python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --latent_size 5 --model SVAE --log_dir logs2/dd/semi_vae --use_mutual False --post_sample_num 1
# GM-VAE
python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --latent_size 5 --model GMVAE --log_dir logs2/dd/gmvae --beta 1.0 --use_mutual False --post_sample_num 1 --sel_metric obj --lr_decay False
python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --latent_size 5 --model GMVAE --log_dir logs2/dd/gmvae --beta 1.0 --use_mutual True --post_sample_num 1 --sel_metric obj --lr_decay False
python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --latent_size 5 --model GMVAE --log_dir logs2/dd/gmvae --beta 0.3 --use_mutual False --post_sample_num 1 --sel_metric obj --lr_decay False
python main_inter.py --data daily_dialog --data_dir data/daily_dialog --mult_k 3 --k 5 --latent_size 5 --model GMVAE --log_dir logs2/dd/gmvae --beta 0.3 --use_mutual True --post_sample_num 1 --sel_metric obj --lr_decay False

# === Test DD (supervised) ===
python main_supervised.py --data daily_dialog --data_dir data/daily_dialog --model DiVAE --log_dir logs/dd_sup/divae
python main_supervised.py --data daily_dialog --data_dir data/daily_dialog --model BMVAE --log_dir logs/dd_sup/bmvae --beta 0.6

# === Test SMD ===
python main_stanford.py --data stanford --data_dir data/stanford --model AeED --log_dir logs/smd/divae --use_mutual True --anneal False
python main_stanford.py --data stanford --data_dir data/stanford --model AeED_GMM --log_dir logs/smd/dgmvae --use_mutual True --beta 0.5 --freeze_step 7000



