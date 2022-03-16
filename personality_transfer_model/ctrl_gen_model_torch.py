from sympy import dict_merge
import torch
from torch import nn
import texar.torch as tx
from texar.torch.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier


class CtrlGenModel(nn.Module):
    def __init__(self, input_len, vocab, gamma, lambda_g, hparams, device):
        super().__init__()
        self._hparams = tx.HParams(hparams, None)
        self.vocab = vocab
        self.gamma = gamma
        self.lambda_g = lambda_g
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

        self.connector = MLPTransformConnector(output_size=self.decoder.output_size,
                                               linear_layer_dim=(self._hparams.dim_c + self._hparams.dim_z)).to(device)
        # Creates classifier
        self.clas_embedder = WordEmbedder(vocab_size=self.vocab.size,
                                          hparams=self._hparams.embedder).to(device)
        # texar's tensorflow has different argument name for out_channels, change it
        classifier_hp = self._hparams.classifier
        if 'filters' in list(classifier_hp.keys()):
            classifier_hp = self._hparams.todict()['classifier']
            filters = classifier_hp.pop('filters')
            classifier_hp['out_channels'] = filters
            classifier_hp['data_format'] = 'channels_last'

        self.classifier = Conv1DClassifier(in_features=input_len,
                                           in_channels=self.embedder.dim,
                                           hparams=classifier_hp).to(device)

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
        loss_d_clas = nn.functional.binary_cross_entropy_with_logits(
            input=clas_logits, target=inputs['labels'].to(torch.float32))
        # loss_d_clas = tf.reduce_mean(loss_d_clas) FIXME
        accu_d = tx.evals.accuracy(labels=inputs['labels'], preds=clas_preds)

        return loss_d_clas

    def forward_g(self, inputs):
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
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(start_tokens, end_token, tau=self.gamma)
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
        # loss_g_clas = tf.reduce_mean(loss_g_clas) FIXME

        # Aggregates losses
        loss_g = loss_g_ae + self.lambda_g * loss_g_clas

        return loss_g, (soft_preds, outputs_, length_)

    def forward(self, inputs, mode):
        loss, result = getattr(self, f'forward_{mode}')(inputs)

        if result is not None:
            soft_preds, outputs_, length_ = result
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

        return loss

    def infer(self, inputs):
        ...
