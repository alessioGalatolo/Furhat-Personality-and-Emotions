"""Text style transfer

This is a simplified implementation of:

Toward Controlled Generation of Text, ICML2017
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing

Download the data with the cmd:

$ python prepare_data.py --dataset <dataset>

Train the model with the cmd:

$ python train_torch.py --config config --dataset <dataset>
"""

import os
import argparse
import importlib
import torch
import texar.torch as tx
from tqdm import tqdm
from ctrl_gen_model_torch import CtrlGenModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running PyTorch using {device}")


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
    vocab = train_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.DataIterator(
        {'train_g': train_data, 'train_d': train_data})
    input_len = iterator.get_iterator('train_d').__next__()['text_ids'].size(1)-1

    # Model
    gamma_decay = config.gamma_decay

    model = CtrlGenModel(input_len, vocab,
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

    # Train
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

        avg_meters_d = tx.utils.AverageRecorder(size=10)
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        data_iterator = tqdm(zip(iterator.get_iterator('train_d'),
                                 iterator.get_iterator('train_g')),
                             total=int(len(train_data)/train_data.batch_size))

        for batch_d, batch_g in data_iterator:
            loss_d, accu_d = model.forward(batch_d, step='d')
            loss_d.backward()
            train_d()
            avg_meters_d.add(accu_d)

            loss_g, accu_g = model.forward(batch_g, step='g', gamma=gamma, lambda_g=lambda_g)
            loss_g.backward()
            train_g()
            avg_meters_g.add(accu_g)
            data_iterator.set_description(f'Accu_d: {avg_meters_d.to_str(precision=4)}, '
                                          + f'Accu_g: {avg_meters_g.to_str(precision=4)}')

        torch.save({'model_state_dict': model.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'epoch': epoch},
                   checkpoint_path)
    torch.save({'model_state_dict': model.state_dict(),
                'input_len': input_len,
                'vocab_file': config.train_data(args.dataset)['datasets'][0]['vocab_file']},
               os.path.join(config.checkpoint_path, 'final_model.pth'))


if __name__ == '__main__':
    main()
