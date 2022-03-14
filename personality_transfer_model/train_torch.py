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
    parser = argparse.ArgumentParser(description='')  # TODO
    parser.add_argument('--config',
                        default='config',
                        help='The config to use.')
    parser.add_argument('--dataset',
                        help='The name of the dataset to use.',
                        required=True)
    args = parser.parse_args()
    config = importlib.import_module(args.config)

    # Data
    train_data = tx.data.MultiAlignedData(config.train_data(args.dataset))
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
    gamma = config.gamma_decay
    lambda_g = config.lambda_g
    model = CtrlGenModel(batch, vocab, gamma, lambda_g, config.model).to(device)

    os.makedirs(config.sample_path, exist_ok=True)
    os.makedirs(config.checkpoint_path, exist_ok=True)

    # saver = tf.train.Saver(max_to_keep=None)
    # if config.restore:
    #     print('Restore from: {}'.format(config.restore))
    #     saver.restore(sess, config.restore)

    gamma_ = 1.
    lambda_g_ = 0.

    # Convert options from tf to torch syntax
    opt_hp = model._hparams.opt.todict()
    if opt_hp['optimizer']['type'] == 'AdamOptimizer':
        opt_hp['optimizer']['type'] = 'Adam'
        if 'learning_rate' in opt_hp['optimizer']['kwargs']:
            lr = opt_hp['optimizer']['kwargs'].pop('learning_rate')
            opt_hp['optimizer']['kwargs']['lr'] = lr
    train_g = tx.core.get_optimizer(model.g_params(), hparams=opt_hp)
    train_g_ae = tx.core.get_optimizer(model.g_params(), hparams=opt_hp)
    train_d = tx.core.get_optimizer(model.d_params(), hparams=opt_hp)
    for epoch in range(1, config.max_nepochs + 1):
        if epoch > config.pretrain_nepochs:
            # Anneals the gumbel-softmax temperature
            gamma_ = max(0.001, gamma_ * gamma)
            lambda_g_ = lambda_g
        print('gamma: {}, lambda_g: {}'.format(gamma_, lambda_g_))

        # Train
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        for batch_d, batch_g in tqdm(zip(iterator.get_iterator('train_d'),
                                         iterator.get_iterator('train_g')),
                                     total=int(len(train_data)/train_data.batch_size)):

            loss_g, loss_g_ae, loss_d = model(batch_d)
            loss_d.backward()
            train_d()
            batch_d.to('cpu')

            loss_g, loss_g_ae, loss_d = model(batch_g)
            loss_g.backward()
            train_g()
            batch_g.to('cpu')

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_d_state_dict': optim_d.state_dict(),
                    'optimizer_g_state_dict': optim_g.state_dict()},
                   os.path.join(config.checkpoint_path, 'ckpt.pth'))
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
