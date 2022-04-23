import pandas as pd
from style_transfer import PersonalityTransfer


def main():
    input_file = '../pilot_data/inputs'
    output_file = '../pilot_data/outputs'
    data = pd.read_csv(input_file, sep='<', names=['original'])
    models = ['model_0', 'model_1']
    modes = ['nucleus_paraphrase', 'nucleus', 'greedy_paraphrase', 'greedy', 'paraphrase']
    top_ps = [i/20 for i in range(20)]
    for model in models:
        for mode in modes:
            for top_p in top_ps:
                pt = PersonalityTransfer(f"style_paraphrase/saved_models/{model}",
                                         "paraphraser_gpt2_large",
                                         mode,
                                         top_p=top_p)
                data[f'{model}_{mode}_{top_p}'] = pt.transfer_style("\n".join(data['original']))
    data.to_csv(output_file)


if __name__ == "__main__":
    main()
