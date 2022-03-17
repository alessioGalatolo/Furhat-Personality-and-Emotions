"""Text style transfer

This is a simplified implementation of:

Toward Controlled Generation of Text, ICML2017
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing

Download the data with the cmd:

$ python prepare_data.py

Train the model with the cmd:

$ python train_torch.py --config config --dataset <dataset>
"""

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
import argparse
import importlib
import numpy as np
import torch
import texar.torch as tx
from tqdm import tqdm
from ctrl_gen_model_torch import CtrlGenModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running pytorch using {device}")

# def eval_epoch(sess, gamma_, lambda_g_, epoch, val_or_test='val'):
#     avg_meters = tx.utils.AverageRecorder()

#     while True:
#         try:
#             feed_dict = {
#                 iterator.handle: iterator.get_handle(sess, val_or_test),
#                 gamma: gamma_,
#                 lambda_g: lambda_g_,
#                 tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
#             }

#             vals = sess.run(model.fetches_eval, feed_dict=feed_dict)

#             batch_size = vals.pop('batch_size')

#             # Computes BLEU
#             samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
#             hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)

#             refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
#             refs = np.expand_dims(refs, axis=1)

#             bleu = tx.evals.corpus_bleu_moses(refs, hyps)
#             vals['bleu'] = bleu

#             avg_meters.add(vals, weight=batch_size)

#             # Writes samples
#             tx.utils.write_paired_text(
#                 refs.squeeze(), hyps,
#                 os.path.join(config.sample_path, 'val.%d' % epoch),
#                 append=True, mode='v')

#         except tf.errors.OutOfRangeError:
#             print('{}: {}'.format(
#                 val_or_test, avg_meters.to_str(precision=4)))
#             break

#     return avg_meters.avg()


def main():
    parser = argparse.ArgumentParser(description='Model for text style transfer')
    parser.add_argument('--config',
                        default='config',
                        help='The config to use.')
    parser.add_argument('--dataset',
                        help='The name of the dataset to use.',
                        required=True)
    parser.add_argument('--load-checkpoint',
                        help='Whether to start again from the last checkpoint',
                        action='store_true')
    args = parser.parse_args()
    config = importlib.import_module(args.config)
    checkpoint_path = os.path.join(config.checkpoint_path, 'ckpt.pth')

    os.makedirs(config.sample_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Data
    train_data = tx.data.MultiAlignedData(config.train_data(args.dataset),
                                          device=device)
    # val_data = tx.data.MultiAlignedData(config.val_data)
    # test_data = tx.data.MultiAlignedData(config.test_data)
    vocab = train_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.DataIterator(
        {'train_g': train_data, 'train_d': train_data})
    batch = iterator.get_iterator('train_d').__next__()

    # Model
    gamma_decay = config.gamma_decay
    model_config = config.model

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

    model = CtrlGenModel(batch['text_ids'].size(1)-1, vocab,
                         config.model, device).to(device)

    optim_g = tx.core.get_optimizer(model.g_params(),
                                    hparams=model._hparams.opt)
    optim_d = tx.core.get_optimizer(model.d_params(),
                                    hparams=model._hparams.opt)

    initial_epoch = 1
    if args.load_checkpoint:
        print(f'Restoring checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        initial_epoch = checkpoint['epoch']

    train_g = tx.core.get_train_op(optimizer=optim_g)
    train_d = tx.core.get_train_op(optimizer=optim_d)

    gamma_0 = 1.
    gamma = gamma_0
    lambda_g = 0.

    print(f'Starting training from epoch {initial_epoch}')
    for epoch in range(initial_epoch, config.max_nepochs + 1):
        if epoch == config.pretrain_nepochs+1:
            lambda_g = config.lambda_g
            optim_g = tx.core.get_optimizer(model.g_params(),
                                            hparams=model._hparams.opt)
            train_g = tx.core.get_train_op(optimizer=optim_g)

        if epoch > config.pretrain_nepochs:
            # Anneals the gumbel-softmax temperature
            gamma = max(0.001, gamma_0 * (gamma_decay ** (epoch-config.pretrain_nepochs)))
        print(f'gamma: {gamma}, lambda_g: {lambda_g}')

        # Train
        model.train()
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        for batch_d, batch_g in tqdm(zip(iterator.get_iterator('train_d'),
                                         iterator.get_iterator('train_g')),
                                     total=int(len(train_data)/train_data.batch_size)):

            loss_d = model.forward(batch_d, mode='d')
            loss_d.backward()
            train_d()

            loss_g = model.forward(batch_g, mode='g', gamma=gamma, lambda_g=lambda_g)
            loss_g.backward()
            train_g()

        torch.save({'model_state_dict': model.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'epoch': epoch},
                   checkpoint_path)
        # Val
        # iterator.restart_dataset(sess, 'val')
        # _eval_epoch(sess, gamma_, lambda_g_, epoch, 'val')

        # saver.save(
        #     sess, os.path.join(config.checkpoint_path, 'ckpt'), epoch)

        # # Test
        # iterator.restart_dataset(sess, 'test')
        # _eval_epoch(sess, gamma_, lambda_g_, epoch, 'test')


if __name__ == '__main__':
    main()
