import torch
import torch.nn as nn
import torch.nn.functional as F
from dgmvae.dataset.corpora import PAD, BOS, EOS, UNK
from torch.autograd import Variable
from dgmvae import criterions
from dgmvae.enc2dec.decoders import DecoderRNN
from dgmvae.enc2dec.encoders import EncoderRNN
from dgmvae.utils import INT, FLOAT, LONG, cast_type
from dgmvae import nn_lib
import numpy as np
from dgmvae.models.model_bases import BaseModel
from dgmvae.models.sent_models import DiVAE as DiVAEBase
from dgmvae.enc2dec.decoders import GEN, TEACH_FORCE
from dgmvae.utils import Pack, kl_anneal_function, interpolate, idx2onehot
import math

class BMVAE(BaseModel):
    def __init__(self, corpus, config):
        super(BMVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.pad_id = self.rev_vocab[PAD]
        self.num_layer = config.num_layer
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell)


        self.q_z = nn.Sequential(
            nn.Linear(self.enc_out_size, self.enc_out_size),
            nn.Tanh(),
            nn.Linear(self.enc_out_size, config.latent_size * config.mult_k)
        )
        self.q_c = nn.Sequential(
            nn.Linear(self.enc_out_size, self.enc_out_size),
            nn.Tanh(),
            nn.Linear(self.enc_out_size, config.k)
        )

        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector((config.latent_size + config.k) * config.mult_k if config.feed_discrete_variable_into_decoder
                                                         else config.latent_size * config.mult_k,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size, self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=self.num_layer, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.ce_weight = config.ce_weight if "ce_weight" in config else 0.0
        self.concat_decoder_input = config.concat_decoder_input if "concat_decoder_input" in config else False

        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

        self.kl_w = 0.0

        self.init_prior()

        self.return_latent_key = ('log_qy', 'dec_init_state', 'y_ids', 'z')

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--latent_size', type=int, default=8, help="size of latent space for each discrete variables")
        parser.add_argument('--mult_k', type=int, default=30, help="number of discrete variables")
        parser.add_argument('--k', type=int, default=30, help="number of classes for each variable")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=200)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=False)
        parser.add_argument('--num_layer', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)
        parser.add_argument('--feed_discrete_variable_into_decoder', type=str2bool, default=False)

        # Dispersed GMVAE settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--beta', type=float, default=0.1)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=False)
        parser.add_argument('--gmm', type=str2bool, default=True)
        parser.add_argument('--bmm', type=str2bool, default=True)

        # supervised settings:
        parser.add_argument('--ce_weight', type=float, default=1.0)
        parser.add_argument('--mi_weight', type=float, default=0.0)

        return parser

    def eta2theta(self, eta):
        # eta: [?, mult_k, latent_size - 1]
        assert eta.dim() == 3
        _eta = torch.cat((eta, torch.zeros((eta.size(0), eta.size(1), 1), dtype=eta.dtype, device=eta.device)), dim=-1)
        _theta = torch.softmax(_eta, dim=-1)
        return _theta

    def init_prior(self, prior_theta=None):
        # Natural Parameters
        assert self.config.latent_size >= 2
        if prior_theta is not None:
            assert prior_theta.size(0) == self.config.k
            assert prior_theta.size(1) == self.config.mult_k
            assert prior_theta.size(2) == self.config.latent_size
            # theta_i = log(p_i / p_k)
            _eta = torch.log(prior_theta / prior_theta[:, :, -1])[:, :, :-1]
            self._eta = torch.nn.Parameter(_eta, requires_grad=True)  # change: False
        else:
            rand_eta = torch.randn(self.config.k, self.config.mult_k,
                                   self.config.latent_size - 1)  # k: number of Gaussian Components, mult_k: number of variable, latent_size: value of each variable
            self._eta = torch.nn.Parameter(rand_eta, requires_grad=True)

    def cluster_logits(self, query):
        # query: batch x (mult_k * latent_size)
        # return: batch x k.
        q_repeat = query.view(-1, self.config.mult_k, self.config.latent_size).unsqueeze(2).expand(-1, -1, self.config.k, -1)
        log_p = - ((q_repeat - self.gaussian_mus) / self.gaussian_vars.abs()).pow(
            2) / 2 - self.gaussian_vars.abs().log() - 0.5 * math.log(2 * math.pi)
        # log_p: [batch_size x mult_k x k x latent_size]
        log_p_sum = torch.sum(log_p, dim=-1)
        return log_p_sum

    def model_sel_loss(self, loss, batch_cnt): # return albo
        return loss.elbo + self.ce_weight * loss.ce_z

    def valid_loss(self, loss, batch_cnt=None, step = None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0

        if not self.config.anneal:
            vae_kl_weight = 1.0

        total_loss = loss.nll
        total_loss += vae_kl_weight * (loss.ckl + loss.zkl)  # TODO: reg_kl -> ckl here
        if self.config.use_mutual:
            total_loss -= self.config.mi_weight * loss.mi

        if loss.ce_z is not None:
            total_loss += (self.ce_weight * loss.ce_z)
        if loss.ce_c is not None and loss.klz_sup is not None:
            total_loss += self.ce_weight * (loss.ce_c + loss.klz_sup)

        return total_loss

    def _ave_eta(self, tgt_prob):
        # tgt_prob: [batch_size, k]
        # self.eta: self.config.k, self.config.mult_k, self.config.latent_size - 1)
        ave_eta = torch.mm(tgt_prob, self._eta.view(self.config.k, -1)).view(-1, self.config.mult_k, self.config.latent_size - 1)
        ave_theta = self.eta2theta(ave_eta) # batch_size x mult_k x latent_size
        return ave_theta

    def zkl_loss(self, tgt_probs, log_qz, mean_z=True):
        # log_qz: [batch_size, mult_k, latent_size]
        if mean_z == True:
            ave_p = self._ave_eta(tgt_probs)
            zkl = torch.exp(log_qz) * (log_qz - torch.log(ave_p + 1e-15))
            zkl = torch.mean(torch.sum(zkl, dim=(-1, -2)))
        else:
            log_qz = log_qz.unsqueeze(1).expand(-1, self.config.k, -1, -1) # [batch_size, k, mult_k, latent_size]
            zkl = torch.exp(log_qz) * (log_qz - torch.log(self.eta2theta(self._eta) + 1e-15)) # [batch_size, k, mult_k, latent_size]
            zkl = torch.sum(zkl, dim=(-1, -2))  # [batch_size, k]
            zkl = torch.mean(torch.sum(zkl * tgt_probs, dim=1))
        return zkl

    def dispersion(self, tgt_probs):
        # A = log(sum_i e^{eta_i})
        # mv = E_{q(c|x)} A(eta_c) - A(E_{q(c|x)}eta_c)
        # tgt_probs: [batch_size, k]
        epsilon = 1e-12
        _eta = torch.cat([self._eta, torch.zeros([self.config.k, self.config.mult_k, 1], dtype=self._eta.dtype,
                                                 device=self._eta.device)], dim=-1)  # [k, mult_k, latent_size]
        A_eta_c = torch.log(torch.sum(torch.exp(_eta), dim=-1) + epsilon) # [k, mult_k]
        EA = torch.sum(torch.mm(tgt_probs, A_eta_c), dim=-1)
        E_eta_c = torch.mm(tgt_probs, _eta.view(self.config.k, -1)).view(-1, self.config.mult_k, self.config.latent_size)
        AE = torch.sum(torch.log(torch.sum(torch.exp(E_eta_c), dim=-1) + epsilon), dim=-1)
        return torch.mean(EA - AE)

    def mean_of_params(self, tgt_probs):
        # A = log(sum_i e^{eta_i})
        # mv = E_{q(c|x)} A(eta_c) - A(E_{q(c|x)}eta_c)
        # tgt_probs: [batch_size, k]
        epsilon = 1e-12

        _eta = torch.cat([self._eta, torch.zeros([self.config.k, self.config.mult_k, 1], dtype=self._eta.dtype,
                                                 device=self._eta.device)], dim=-1)  # [k, mult_k, latent_size]

        res = torch.mm(tgt_probs, (_eta * _eta).view(self.config.k, -1)) - torch.mm(tgt_probs, _eta.view(self.config.k, -1)).pow(2)

        return torch.sum(res) / tgt_probs.size(0)

    def suploss_for_c(self, log_qc, labels_c, log_qz):
        # log_qc: [batch_size, k]
        # labels_c: [batch_size]
        # CE for C:
        ce_c = torch.nn.functional.nll_loss(log_qc, labels_c, reduction="mean")

        golden_c = self._eta[labels_c]  # [batch_size, mult_k, latent_size]
        kl_z = torch.exp(log_qz) * (log_qz - torch.log(self.eta2theta(golden_c)))
        return ce_c, kl_z

    def suploss_for_z(self, log_qz, labels_z):
        # log_qz: batch_size x mult_k x latent_size
        # labels_z: batch_size x mult_k'

        batch_size = log_qz.size(0)
        ce_loss = torch.nn.functional.nll_loss(log_qz[:, :labels_z.size(-1), :].contiguous().view(-1, self.config.latent_size), labels_z.view(-1),
                                               reduction="sum")
        return ce_loss / batch_size

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        z_labels = data_feed.get("z_labels", None)
        c_labels = data_feed.get("c_labels", None)
        
        if z_labels is not None:
            z_labels = self.np2var(z_labels, LONG)
        if c_labels is not None:
            c_labels = self.np2var(c_labels, LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # x_last = torch.mean(x_outs, dim=1)

        # posterior network
        qc_logits = self.q_c(x_last)  # batch_size x k
        qc = torch.softmax(qc_logits, dim=-1)  # batch_size x k
        qz_logits = self.q_z(x_last).view(-1, self.config.mult_k, self.config.latent_size)  # batch_size x mult_k x latent_size

        if mode == GEN and gen_type == "sample":
            sample_c = torch.randint(0, self.config.k, (batch_size,), dtype=torch.long)  # [sample_n, 1]
            pz = self.eta2theta(self._eta[sample_c])  # [k, mult_k, latent_size] -> [sample_n, mult_k, latent_size]

            sample_y, y_ids = self.cat_connector(torch.log(pz).view(-1, self.config.latent_size), 1.0, self.use_gpu,
                                                 hard=not self.training, return_max_id=True)
            sample_y = sample_y.view(-1, self.config.mult_k * self.config.latent_size)
            y_ids = y_ids.view(-1, self.config.mult_k)
        else:
            sample_y, y_ids = self.cat_connector(qz_logits.view(-1, self.config.latent_size), 1.0, self.use_gpu,
                                                 hard=True, return_max_id=True)
            # sample_y: [batch* mult_k, latent_size], y_ids: [batch* mult_k, 1]
            sample_y = sample_y.view(-1, self.config.mult_k * self.config.latent_size)
            y_ids = y_ids.view(-1, self.config.mult_k)

        # decode
        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(sample_y)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   mode=mode, gen_type="greedy",
                                                   beam_size=self.beam_size,
                                                   latent_variable=sample_y if self.concat_decoder_input else None)
        # compute loss or return results
        if mode == GEN:
            dec_ctx[DecoderRNN.KEY_LATENT] = y_ids
            if mode == GEN and gen_type == "sample":
                dec_ctx[DecoderRNN.KEY_CLASS] = sample_c
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels)
            ppl = self.ppl(dec_outs, labels)

            # regularization terms:
            # CKL:
            avg_log_qc = torch.log(torch.mean(qc, dim=0) + 1e-15)  # [k]
            # ckl = torch.sum(torch.exp(avg_log_qc) * (avg_log_qc - self.log_uniform_y))
            # CKL (original)
            log_qc = torch.log(qc + 1e-15)
            ckl = torch.mean(torch.sum(qc * (log_qc - self.log_uniform_y), dim=-1))  #

            # ZKL
            log_qz = torch.log_softmax(qz_logits, dim=-1)
            qz = torch.exp(log_qz)
            zkl = self.zkl_loss(qc, log_qz, mean_z=True)
            # ZKL (original)
            zkl_ori = self.zkl_loss(qc, log_qz, mean_z=False)

            # MI: in this model, the mutual information is calculated for z
            avg_log_qz = torch.log(torch.mean(qz, dim=0) + 1e-15) # mult_k x k
            mi = torch.mean(torch.sum(qz * log_qz, dim=(-1,-2))) - torch.sum(torch.exp(avg_log_qz) * avg_log_qz)
            mi_of_c = torch.mean(torch.sum(qc * log_qc, dim=-1)) - torch.sum(torch.exp(avg_log_qc) * avg_log_qc)

            # dispersion term
            dispersion = self.dispersion(qc)

            if self.config.beta > 0:
                zkl = zkl + self.config.beta * dispersion

            if c_labels is not None:
                ce_c, klz_sup = self.suploss_for_c(log_qc, c_labels, log_qz)
            else:
                ce_c, klz_sup = None, None
            ce_z = self.suploss_for_z(log_qz, z_labels) if z_labels is not None else None
            c_entropy = torch.mean(torch.sum(qc * log_qc,dim=-1))

            results = Pack(nll=nll, mi=mi, ckl=ckl, zkl=zkl, dispersion=dispersion, PPL=ppl,
                           real_zkl=zkl_ori, real_ckl=ckl,
                           ce_z=ce_z, ce_c=ce_c, klz_sup=klz_sup,
                           elbo=nll + zkl_ori + ckl,
                           c_entropy=c_entropy, mi_of_c=mi_of_c,
                           param_var=self.mean_of_params(tgt_probs=qc))

            if return_latent:
                results['log_qy'] = log_qz
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids
                results['z'] = sample_y

            return results

    def sampling_backward_acc(self, batch_size):
        sample_y = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        # print(sample_y)
        # print(sample_y.size())
        # exit()
        y_index = (self.torch2var(torch.arange(self.config.mult_k) * self.config.k) + sample_y).view(-1)
        # sample_y = model.np2var(sample_y)
        mean = self.gaussian_mus.view(-1, self.config.latent_size)[y_index].squeeze()
        sigma = self.gaussian_vars.view(-1, self.config.latent_size)[y_index].squeeze()
        zs = self.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        zs = zs.view(-1, self.config.mult_k * self.config.latent_size)
        cs = self.torch2var(idx2onehot(sample_y.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)
        dec_init_state = self.dec_init_connector(torch.cat((cs, zs), dim=1)
                                                  if self.config.feed_discrete_variable_into_decoder
                                                  else zs)

        _, _, outputs = self.decoder(cs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type=self.config.gen_type,
                                      beam_size=self.config.beam_size)
        return outputs

    def sampling(self, batch_size):
        sample_y = torch.randint(0, self.config.k, [batch_size, self.config.mult_k], dtype=torch.long).cuda()
        # print(sample_y)
        # print(sample_y.size())
        # exit()
        y_index = (self.torch2var(torch.arange(self.config.mult_k) * self.config.k) + sample_y).view(-1)
        # sample_y = model.np2var(sample_y)
        mean = self.gaussian_mus.view(-1, self.config.latent_size)[y_index].squeeze()
        sigma = self.gaussian_vars.view(-1, self.config.latent_size)[y_index].squeeze()
        zs = self.reparameterization(mean, 2 * torch.log(torch.abs(sigma) + 1e-15), sample=True)
        zs = zs.view(-1, self.config.mult_k * self.config.latent_size)
        cs = self.torch2var(idx2onehot(sample_y.view(-1), self.config.k)).view(-1, self.config.mult_k * self.config.k)
        dec_init_state = self.dec_init_connector(torch.cat((cs, zs), dim=1)
                                                  if self.config.feed_discrete_variable_into_decoder
                                                  else zs)

        _, _, outputs = self.decoder(cs.size(0),
                                      None, dec_init_state,
                                      mode=GEN, gen_type=self.config.gen_type,
                                      beam_size=self.config.beam_size)
        return outputs

    def sampling_for_likelihood(self, batch_size, data_feed, sample_num, sample_type="LL"):
        # Importance sampling...
        assert sample_type in ("LL", "ELBO")

        # just for calculating log-likelihood
        if type(data_feed) is tuple:
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)  # batch_size * seq_len
        out_utts = out_utts.repeat(sample_num, 1)

        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # qc_logits = self.q_c(x_last)  # batch_size x k
        # qc = torch.softmax(qc_logits, dim=-1)  # batch_size x k
        qz_logits = self.q_z(x_last).view(-1, self.config.mult_k, self.config.latent_size)  # batch_size x mult_k x latent_size
        log_qz = torch.log_softmax(qz_logits, dim=-1).view(-1, self.config.latent_size)  # [batch_size x mult_k, latent_size]
        sample_z = torch.multinomial(torch.exp(log_qz), 1)  # [batch_sizexmult_k, 1]
        # print(sample_z.size())
        log_qzx = torch.sum(torch.gather(log_qz, 1, sample_z).view(-1, self.config.mult_k), dim=-1)

        log_theta = torch.log(self.eta2theta(self._eta)) # k, mult_k, latent_size
        log_pz = torch.gather(log_theta.unsqueeze(0).repeat(log_qzx.size(0), 1, 1, 1).view(-1, self.config.latent_size), 1,
                     sample_z.view(-1, 1, self.config.mult_k).repeat(1, self.config.k, 1).view(-1, 1)).view(-1, self.config.k, self.config.mult_k)
        log_pz = log_pz.double()
        log_pz = torch.log(torch.mean(torch.exp(torch.sum(log_pz, dim=-1)), dim=-1))


        sample_z = self.torch2var(idx2onehot(sample_z.view(-1), self.config.latent_size)).view(-1, self.config.latent_size * self.config.mult_k)

        # Calculate p(x|z)
        dec_init_state = self.dec_init_connector(sample_z)
        dec_outs, dec_last, outputs = self.decoder(sample_z.size(0),
                                                   dec_inputs, dec_init_state,
                                                   mode=TEACH_FORCE,
                                                   gen_type=self.config.gen_type,
                                                   beam_size=self.config.beam_size,
                                                   latent_variable=sample_z if self.concat_decoder_input else None)

        # nll = self.nll_loss(dec_outs, labels)
        nll = F.nll_loss(dec_outs.view(-1, dec_outs.size(-1)), labels.view(-1), reduction="none").view(out_utts.size(0),
                                                                                                       -1)
        nll = torch.sum(nll, dim=-1)

        # print(nll, log_pz, log_qzx)

        ll = torch.exp(-nll.double() + log_pz - log_qzx.double())  # log (p(z)p(x|z) / q(z|x))
        return ll
        # ll = ll.view(-1, sample_num)
        # nll_per = torch.log(torch.mean(ll, dim=-1))  #
        # batch_size = nll_per.size(0)
        # nll_per = torch.sum(nll_per)

        return nll_per, batch_size

class DiVAE(BaseModel):

    def __init__(self, corpus, config):
        super(DiVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.unk_id = self.rev_vocab[UNK]
        self.num_layer = config.num_layer
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell)

        self.q_y = nn.Linear(self.enc_out_size, config.mult_k * config.k)
        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(config.mult_k * config.k,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size, self.dec_cell_size,
                                  self.go_id, self.eos_id, self.unk_id,
                                  n_layers=self.num_layer, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  embedding=self.embedding)

        if 'bow_loss' in self.config and self.config.bow_loss:
            self.bow_mlp = nn.Linear(config.mult_k * config.k, self.vocab_size)
            self.bow_loss = True
            self.bow_entropy = criterions.BowEntropy(self.rev_vocab[PAD], self.config)
        else:
            self.bow_loss = False

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.ppl = criterions.Perplexity(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()
        self.log_py = nn.Parameter(torch.log(torch.ones(self.config.mult_k,
                                                        self.config.k)/config.k),
                                   requires_grad=True)
        self.register_parameter('log_py', self.log_py)

        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

        self.kl_w = 0.0

        self.return_latent_key = ("dec_init_state", "log_qy", "y_ids")

    @staticmethod
    def add_args(parser):
        from dgmvae.utils import str2bool
        # Latent variable:
        parser.add_argument('--mult_k', type=int, default=30, help="number of discrete variables")
        parser.add_argument('--k', type=int, default=8, help="number of classes for each variable")

        # Network setting:
        parser.add_argument('--rnn_cell', type=str, default='gru')
        parser.add_argument('--embed_size', type=int, default=200)
        parser.add_argument('--utt_type', type=str, default='rnn')
        parser.add_argument('--enc_cell_size', type=int, default=512)
        parser.add_argument('--dec_cell_size', type=int, default=512)
        parser.add_argument('--bi_enc_cell', type=str2bool, default=False)
        parser.add_argument('--num_layer', type=int, default=1)
        parser.add_argument('--use_attn', type=str2bool, default=False)
        parser.add_argument('--attn_type', type=str, default='cat')
        parser.add_argument('--tie_output_embed', type=str2bool, default=True)
        parser.add_argument('--max_utt_len', type=int, default=40)
        parser.add_argument('--max_dec_len', type=int, default=40)
        parser.add_argument('--max_vocab_cnt', type=int, default=10000)

        # Di-VAE settings:
        parser.add_argument('--use_mutual', type=str2bool, default=False)
        parser.add_argument('--concat_decoder_input', type=str2bool, default=True)  # or True
        parser.add_argument('--gmm', type=str2bool, default=False)
        parser.add_argument('--bmm', type=str2bool, default=False)

        # supervised settings:
        parser.add_argument('--ce_weight', type=float, default=1.0)

        return parser

    def valid_loss(self, loss, batch_cnt=None, step=None):
        if batch_cnt is not None:
            step = batch_cnt

        if step is not None and 'anneal_function' in self.config:
            vae_kl_weight = kl_anneal_function(self.config.anneal_function, step,
                                               self.config.anneal_k, self.config.anneal_x0)
        else:
            vae_kl_weight = 1.0
        if self.config.use_mutual or self.config.anneal is not True:
            vae_kl_weight = 1.0

        total_loss = loss.nll + (vae_kl_weight * loss.reg_kl)

        if "ce_z" in loss:
            total_loss += self.config.ce_weight * loss.ce_z

        return total_loss

    def suploss_for_c(self, log_qc, labels_c, log_qz):
        # log_qc: [batch_size, k]
        # labels_c: [batch_size]
        # CE for C:
        ce_c = torch.nn.functional.nll_loss(log_qc, labels_c, reduction="mean")

        golden_c = self._eta[labels_c]  # [batch_size, mult_k, k]
        kl_z = torch.exp(log_qz) * (log_qz - torch.log(self.eta2theta(golden_c)))
        return ce_c, kl_z

    def suploss_for_z(self, log_qz, labels_z):
        # log_qz: batch_size x mult_k x k
        # labels_z: batch_size x mult_k
        batch_size = log_qz.size(0)
        ce_loss = torch.nn.functional.nll_loss(log_qz[:, :labels_z.size(-1), :].contiguous().view(-1, self.config.k), labels_z.view(-1),
                                               reduction="sum")
        return ce_loss / batch_size

    def model_sel_loss(self, loss, batch_cnt):  # return albo
        return loss.elbo + loss.ce_z

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        if isinstance(data_feed, tuple):
            data_feed = data_feed[0]
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        z_labels = data_feed.get("z_labels", None)
        c_labels = data_feed.get("c_labels", None)
        if z_labels is not None:
            z_labels = self.np2var(z_labels, LONG)
        if c_labels is not None:
            c_labels = self.np2var(c_labels, LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # posterior network
        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, qy_logits.dim()-1)

        # switch that controls the sampling
        sample_y, y_ids = self.cat_connector(qy_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        sample_y = sample_y.view(-1, self.config.k * self.config.mult_k)
        y_ids = y_ids.view(-1, self.config.mult_k)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(sample_y)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels)
            if self.config.avg_type == "seq":
                ppl = self.ppl(dec_outs, labels)

            # regularization qy to be uniform
            avg_log_qy = torch.exp(log_qy.view(-1, self.config.mult_k, self.config.k))
            avg_log_qy = torch.log(torch.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y,
                                          batch_size, unit_average=True)

            real_ckl = self.cat_kl_loss(log_qy, self.log_uniform_y, batch_size, average=False)
            real_ckl = torch.mean(torch.sum(real_ckl.view(-1, self.config.mult_k), dim=-1))
            
            if self.config.use_mutual:
                reg_kl = b_pr
            else:
                reg_kl = real_ckl

            # find out mutual information
            # H(Z) - H(Z|X)
            mi = self.entropy_loss(avg_log_qy, unit_average=True)\
                 - self.entropy_loss(log_qy, unit_average=True)

            ce_z = self.suploss_for_z(log_qy.view(-1, self.config.mult_k, self.config.k), z_labels) if z_labels is not None else None

            results = Pack(nll=nll, reg_kl=reg_kl, mi=mi, bpr=b_pr, real_ckl=real_ckl,
                           ce_z=ce_z, elbo=nll+real_ckl)

            if self.config.avg_type == "seq":
                results['PPL'] = ppl

            if return_latent:
                results['log_qy'] = log_qy
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids

            return results
