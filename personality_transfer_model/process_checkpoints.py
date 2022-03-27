import os
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert checkpoint to latest version')
    parser.add_argument('--base-path',
                        default='./checkpoints',
                        help='The base path from where to find the checkpoints.')
    parser.add_argument('--filename',
                        help='The name of the checkpoint',
                        required=True)
    parser.add_argument('--dataset',
                        help="The dataset (if not included in the checkpoint itself)")
    parser.add_argument('--input-len',
                        help="The input_len (if not included in the checkpoint itself)")
    args = parser.parse_args()
    LATEST_VERSION = 3
    checkpoint_path = os.path.join(args.base_path, args.filename)
    checkpoint = torch.load(checkpoint_path)
    new_checkpoint = {'version': LATEST_VERSION}
    new_checkpoint['model_state_dict'] = checkpoint['model_state_dict']
    try:
        input_len = checkpoint['input_len'] if 'input_len' in checkpoint else args.input_len
        dataset = checkpoint['dataset'] if 'dataset' in checkpoint else args.dataset
    except AttributeError:
        print("Some of the attribute were not found in the checkpoint. You need to input them as arguments.")
        exit(-1)
    new_checkpoint['input_len'] = int(input_len)
    new_checkpoint['dataset'] = dataset
    torch.save(new_checkpoint, checkpoint_path)
