import pandas as pd
from style_transfer import PersonalityTransfer


def main():
    input_file = '../pilot_data/inputs'
    output_file = '../pilot_data/outputs'
    data = pd.read_csv(input_file, sep='<', names=['original'])
    modes = ['nucleus_paraphrase', 'nucleus', 'greedy_paraphrase', 'greedy', 'paraphrase']
    top_ps = [i/20 for i in range(20)]
    pt0 = PersonalityTransfer("style_paraphrase/saved_models/model_0",
                              "paraphraser_gpt2_large",
                              modes[0],
                              top_p=top_ps[0])
    pt1 = PersonalityTransfer("style_paraphrase/saved_models/model_1",
                              "paraphraser_gpt2_large",
                              modes[0],
                              top_p=top_ps[0])
    for mode in modes:
        for top_p in top_ps:
            pt0.change_mode(mode, top_p)
            pt1.change_mode(mode, top_p)
            data[f'model_0_{mode}_{top_p}'] = pt0.transfer_style("\n".join(data['original']))
            data[f'model_1_{mode}_{top_p}'] = pt1.transfer_style("\n".join(data['original']))
    data.to_csv(output_file)


if __name__ == "__main__":
    main()
