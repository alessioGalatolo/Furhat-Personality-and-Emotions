from re import findall
import pandas as pd
from style_transfer import PersonalityTransfer


def main():
    input_file = '../pilot_data/inputs'
    output_file = '../pilot_data/outputs'
    with open(input_file, 'r') as input:
        inputs = input.readlines()
    data = pd.DataFrame({'original': inputs})
    modes = ['nucleus_paraphrase', 'nucleus']
    top_ps = [0.5+i/10 for i in range(5)]
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
            results_0 = []
            results_1 = []
            for row in data.iterrows():
                results_0.append(''.join((pt0.transfer_style("\n".join(findall(r'([^.?!]+[.?!])', row[1]['original']))))))
                results_1.append(''.join((pt1.transfer_style("\n".join(findall(r'([^.?!]+[.?!])', row[1]['original']))))))
            data[f'model_0_{mode}_{top_p}'] = results_0
            data[f'model_1_{mode}_{top_p}'] = results_1
    pt0.change_mode('paraphrase', top_p)
    # data['paraphrase'] = pt0.transfer_style("\n".join(data['original']))  # FIXME: data needs to be split as above
    data.to_csv(f'{output_file}.csv')
    for i, row in enumerate(data.iterrows()):
        row[1].T.to_csv(f'{output_file}_{i}.csv')


if __name__ == "__main__":
    main()
