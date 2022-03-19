"""Config
"""

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
        'out_channels': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'data_format': 'channels_last',
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'Adam',
            'kwargs': {
                'lr': 5e-4,
            },
        },
    },
}
