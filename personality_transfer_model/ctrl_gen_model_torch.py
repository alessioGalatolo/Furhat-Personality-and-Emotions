import numpy as np
import torch
from torch import nn
import texar.torch as tx
from texar.torch.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier


class CtrlGenModel(nn.Module):
    def __init__(self, input_len, vocab, hparams, device):
        super().__init__()
        self.input_len = input_len  # FIXME
        self._hparams = tx.HParams(CtrlGenModel.tf_config2torch(hparams),
                                   None)
        self.vocab = vocab
        self.embedder = WordEmbedder(
            vocab_size=vocab.size,
            hparams=self._hparams.embedder).to(device)
        self.encoder = UnidirectionalRNNEncoder(input_size=self.embedder.dim,
                                                hparams=self._hparams.encoder).to(device)

        # Encodes label
        self.label_connector = MLPTransformConnector(self._hparams.dim_c,
                                                     linear_layer_dim=1).to(device)

        # Teacher-force decoding and the auto-encoding loss for G
        self.decoder = AttentionRNNDecoder(
            encoder_output_size=self.encoder.output_size,
            input_size=self.embedder.dim,
            token_embedder=self.embedder,
            cell_input_fn=lambda inputs, attention: inputs,
            vocab_size=self.vocab.size,
            hparams=self._hparams.decoder).to(device)

        connector_lldim = self._hparams.dim_c + self._hparams.dim_z
        self.connector = MLPTransformConnector(output_size=self.decoder.output_size,
                                               linear_layer_dim=connector_lldim).to(device)
        # Creates classifier
        self.clas_embedder = WordEmbedder(vocab_size=self.vocab.size,
                                          hparams=self._hparams.embedder).to(device)

        self.classifier = Conv1DClassifier(in_features=input_len,
                                           in_channels=self.embedder.dim,
                                           hparams=self._hparams.classifier).to(device)

    def g_params(self):
        # do not include embedder since its parameters are present in the decoder
        g_ = [self.encoder, self.label_connector,
              self.connector, self.decoder]
        params = []
        for model in g_:
            params += list(model.parameters())
        return params

    def d_params(self):
        d_ = [self.clas_embedder, self.classifier]
        params = []
        for model in d_:
            params += list(model.parameters())
        return params

    def forward_d(self, inputs):
        # Classification loss for the classifier
        clas_logits, clas_preds = self.classifier(
            input=self.clas_embedder(ids=inputs['text_ids'][:, 1:]),
            sequence_length=inputs['length'] - 1)
        if 'labels' in inputs:  # check for eval vs train # FIXME
            loss_d_clas = nn.functional.binary_cross_entropy_with_logits(
                input=clas_logits, target=inputs['labels'].to(torch.float32))
        else:
            loss_d_clas = None
        return loss_d_clas, clas_preds

    def forward_g(self, inputs, gamma, lambda_g):
        # text_ids for encoder, with BOS token removed
        enc_text_ids = inputs['text_ids'][:, 1:]
        enc_outputs, final_state = self.encoder(self.embedder(enc_text_ids),
                                                sequence_length=inputs['length'] - 1)
        z = final_state[:, self._hparams.dim_c:]

        labels = inputs['labels'].reshape(-1, 1).to(torch.float32)
        c = self.label_connector(labels)
        c_ = self.label_connector(1 - labels)
        h = torch.concat([c, z], dim=1)
        h_ = torch.concat([c_, z], dim=1)

        g_outputs, _, _ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length'] - 1,
            initial_state=self.connector(h),
            inputs=inputs['text_ids'],
            sequence_length=inputs['length'] - 1)

        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs['text_ids'][:, 1:],
            logits=g_outputs.logits,
            sequence_length=inputs['length'] - 1,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        # Gumbel-softmax decoding, used in training
        start_tokens = torch.ones_like(inputs['labels']) * self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(start_tokens, end_token, tau=gamma)
        # gumbel_helper.initialize(self.embedder.embedding) FIXME

        soft_outputs_, _, soft_length_, = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length'] - 1,
            helper=gumbel_helper, initial_state=self.connector(h_))

        # Greedy decoding, used in eval
        outputs_, _, length_ = self.decoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length'] - 1,
            decoding_strategy='infer_greedy', initial_state=self.connector(h_),
            start_tokens=start_tokens, end_token=end_token)

        # Classification loss for the generator, based on soft samples
        soft_logits, soft_preds = self.classifier(
            input=self.clas_embedder(soft_ids=soft_outputs_.sample_id),
            sequence_length=soft_length_)
        loss_g_clas = nn.functional.binary_cross_entropy_with_logits(
            input=soft_logits,
            target=(1 - inputs['labels']).to(torch.float32))

        # Aggregates losses
        loss_g = loss_g_ae + lambda_g * loss_g_clas

        return loss_g, (outputs_, length_, soft_preds)

    def forward(self, inputs, mode, gamma=None, lambda_g=None):
        self.train()
        if mode == "g":
            loss, (outputs_, length_, soft_preds) = self.forward_g(inputs, gamma, lambda_g)
            with torch.no_grad():
                # Accuracy on soft samples, for training progress monitoring
                accu_g = tx.evals.accuracy(labels=1 - inputs['labels'],
                                            preds=soft_preds)

                # Accuracy on greedy-decoded samples, for training progress monitoring
                _, gdy_preds = self.classifier(
                    input=self.clas_embedder(ids=outputs_.sample_id),
                    sequence_length=length_)
                accu_g_gdy = tx.evals.accuracy(
                    labels=1 - inputs['labels'], preds=gdy_preds)
            accu = [accu_g.detach().cpu(), accu_g_gdy.detach().cpu()]
        elif mode == "d":
            loss, clas_preds = self.forward_d(inputs)
            accu_d = tx.evals.accuracy(labels=inputs['labels'], preds=clas_preds)
            accu = accu_d.detach().cpu()
        else:
            ...

        return loss, accu

    @torch.no_grad()
    def infer(self, text, transfer_clas):
        self.eval()
        text_tokens = text.split()  # FIXME: check length and eventually pad sequence
        text_tokens.insert(0, self.vocab.bos_token)
        text_tokens.append(self.vocab.eos_token)
        if len(text_tokens) < self.input_len:
            for _ in range(self.classifier.output_size-len(text_tokens)):
                text_tokens.append('')
        text_ids = self.vocab.map_tokens_to_ids_py(text_tokens)
        inputs = {'text_ids': torch.Tensor(np.array([text_ids])).long(),
                  'length': torch.Tensor([len(text_ids)]).long()}
        _, clas_preds = self.forward_d(inputs)
        inputs['labels'] = clas_preds
        if clas_preds.item() != transfer_clas:
            _, (output_ids, length, _) = self.forward_g(inputs, gamma=1, lambda_g=0)  # FIXME
        else:
            output_ids = text_ids

        return output_ids

    @staticmethod
    def tf_config2torch(model_config):
        # Convert config options from tf to torch syntax
        if model_config['opt']['optimizer']['type'] == 'AdamOptimizer':
            model_config['opt']['optimizer']['type'] = 'Adam'
        if 'learning_rate' in model_config['opt']['optimizer']['kwargs']:
            lr = model_config['opt']['optimizer']['kwargs'].pop('learning_rate')
            model_config['opt']['optimizer']['kwargs']['lr'] = lr
        if 'filters' in model_config['classifier']:
            filters = model_config['classifier'].pop('filters')
            model_config['classifier']['out_channels'] = filters
            model_config['classifier']['data_format'] = 'channels_last'
        return model_config
