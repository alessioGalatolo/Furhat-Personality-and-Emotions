from importlib import import_module
import os.path as path
import texar.torch as tx
import torch
from ctrl_gen_model import CtrlGenModel
from dataset import TextDataset
from prepare_data import parse_sentence


class PersonalityTransfer:
    def __init__(self, config_module, device='cpu', checkpoint_name='final_model.pth'):
        config = import_module(config_module)
        config = tx.HParams(config.model, None)
        checkpoint_path = path.join(config.checkpoint_path, checkpoint_name)
        assert path.exists(checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        if 'version' not in checkpoint or checkpoint['version'] != 3:
            print('The checkpoint version is not the latest, please update it with process_checkpoints.py')
            exit(-1)
        self.input_len = checkpoint['input_len']

        self.vocab = TextDataset.load_vocab(f"personality_transfer_model/data/{checkpoint['dataset']}/vocab")
        self.model = CtrlGenModel(self.input_len,
                                  self.vocab, config, device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    def transfer(self, text, transfer_clas=None):
        # FIXME what if the text is not in the vocab?
        # Eval
        text += ' .'
        results = []
        for sentence in parse_sentence(text):
            text_tokens = sentence.split()
            text_ids = TextDataset.text2ids(text_tokens, self.vocab, self.input_len)
            output_ids = self.model.infer(text_ids, transfer_clas)

            hyps = self.vocab.map_ids_to_tokens_py(output_ids)
            output_str = ' '.join(hyps.tolist())
            results.append(output_str)
        return results

    def classify(self, text):
        text += ' .'
        class_counter = {}
        for sentence in parse_sentence(text):
            text_tokens = sentence.split()
            text_ids = TextDataset.text2ids(text_tokens, self.vocab, self.input_len)
            clas = self.model.classify(text_ids)
            class_counter[clas] = class_counter[clas]+1 if clas in class_counter else 1
        return max(class_counter, key=class_counter.get)


# Testing
if __name__ == "__main__":
    personality_transfer = PersonalityTransfer('config', checkpoint_name='ckpt_epoch_6.pth')
    original_text = "This is so important to keep reminding our fellow citizens:"
    print(f'Original text: {original_text}')
    print(f'Transferred text: {personality_transfer.classify(original_text)}')
