from collections import defaultdict
from importlib import import_module
import os.path as path
import texar.torch as tx
import torch
import numpy as np
from ctrl_gen_model_torch import CtrlGenModel


class PersonalityTransfer:
    def __init__(self, config_module):
        config = import_module(config_module)
        checkpoint_path = path.join(config.checkpoint_path, 'final_model.pth')
        assert path.exists(checkpoint_path)

        checkpoint = torch.load(checkpoint_path)

        # FIXME
        vocab_hp = tx.HParams({'vocab_file': checkpoint['vocab_file']},
                              tx.data.data.multi_aligned_data._default_dataset_hparams())
        self.vocab = tx.data.MultiAlignedData.make_vocab([vocab_hp])[0]
        self.model = CtrlGenModel(checkpoint['input_len'],
                                  self.vocab, config.model, 'cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def transfer(self, text, transfer_clas):
        # Eval
        output_ids = self.model.infer(text, transfer_clas)

        hyps = self.vocab.map_ids_to_tokens_py(output_ids)
        output_str = ' '.join(hyps)
        return output_str


# Testing
if __name__ == "__main__":
    personality_transfer = PersonalityTransfer('config')
    print(personality_transfer.transfer('hello', 1))
