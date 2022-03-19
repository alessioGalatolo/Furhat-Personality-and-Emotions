"""Config
"""


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


def train_data(dataset):
    return {
        'datasets': [
            {
                'files': f'./personality_transfer_model/data/{dataset}/text',
                'vocab_file': f'./personality_transfer_model/data/{dataset}/vocab',
                'data_name': ''
            },
            {
                'files': f'./personality_transfer_model/data/{dataset}/labels_EXT',
                'data_name': 'labels',
                'data_type': 'int'
            }
        ],
        'name': 'train'
    }


model = {
    'batch_size': 32,
    'seed': 123,
    # Total number of training epochs (including pre-train and full-train)
    'max_nepochs': 12,
    'pretrain_nepochs': 10,  # Number of pre-train epochs (training as autoencoder)
    'checkpoint_path': './checkpoints',
    'lambda_g': 0.1,  # Weight of the classification loss
    'gamma': 1,
    'gamma_decay': 0.5,  # Gumbel-softmax temperature anneal rate

    # models parameters
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': 21,
        'max_decoding_length_infer': 20,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
