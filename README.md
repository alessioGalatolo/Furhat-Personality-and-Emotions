# Personality and Emotions on Furhat
This repo stores all the material of my master's thesis. Here I try to combine personality and emotional expression into the robot [Furhat](https://furhatrobotics.com/). The personality incorporation is done on the base of text on the belief that people with different personalities would speak using different words, etc. The emotional incorporation is done by extracting the emotions from the text the robot is saying and are manifested through facial expressions.
## Repository structure
```bash
.
├── study_data                        # Includes all the data used, collected and analysis from the final study
├── pilot_data                        # Includes all the data used, collected and analysis from the pilot study
├── style_trasnfer_paraphrase         # Contains the paraphrase model for personality style transfer
│   ├── utils                         # Utilities + GAN-based model that has both personality classifier and generator
│   │   ├── prepare_data.py           # Download and pre-process data (compatible with all models)
│   │   ├── train.py                  # Train the GAN-based model
│   │   ├── classifier_eval.py        # Evaluate the classifier
│   │   ├── train.py                  # Train the GAN-based model
│   │   ├── model_wrapper.py          # Model wrapper for easier use in inference mode
│   │   └── ...
│   ├── prepare_tweets_covid.py       # Download and expand (dataset comes with only IDs) the twitter covid dataset
│   ├── prepare_tweets.py             # Download and preprocess the large twitter dataset
│   ├── style_transfer.py             # A wrapper to use the paraphrase model
│   ├── eval_personality_transfer.py  # A script to generate the transferred sentences
│   └── ...
├── emotions.py                       # A wrapper used to recognize emotions from text and express them on furhat
├── main.py                           # Connects to Furhat and executes the experiment
└── ...
```
## Requirements
Python 3.7

The specific requirements can be found in each subfolder. E.g. requirements for paraphrase model are in [`style_transfer_paraphrase/requirements.txt`](style_transfer_paraphrase/requirements.txt)

## Instructions for using STRAP personality transfer checkpoints
Start by cloning this repo.
```bash
git clone https://github.com/alessioGalatolo/Furhat-Personality-and-Emotions
cd Furhat-Personality-and-Emotions
```
Make sure pytorch (>1.9.0) is installed (see official documentation for the best way of doing this) in your working environment.
```bash
python3.7 -m venv env
source env/bin/activate
pip install torch torchvision torchaudio
```
Move to the `style_transfer_paraphrase` directory and install STRAP through pip
```bash
pip install --editable .
```
For use in inference the only other library needed is `transformers`, install with `pip install transformers`.
```bash
pip install transformers
```
Then go to the release page of this repo [github.com/alessioGalatolo/Furhat-Personality-and-Emotions/releases/](https://github.com/alessioGalatolo/Furhat-Personality-and-Emotions/releases/) and download `STRAP_essays.tar.gz`. Extract it in the project folder.
```bash
wget https://github.com/alessioGalatolo/Furhat-Personality-and-Emotions/releases/download/v1.0.0/STRAP_essays.tar.gz
tar zxf STRAP_essays.tar.gz
rm STRAP_essays.tar.gz
```
Go to [https://drive.google.com/drive/folders/1hB0lJt4MjuWbgdY7_I_2eISAoNmM6O9f](https://drive.google.com/drive/folders/1hB0lJt4MjuWbgdY7_I_2eISAoNmM6O9f) and download the folder `paraphraser_gpt2_large`. Then extract it under `style_transfer_paraphrase/paraphraser_gpt2_large` (this has to be done manually).

Then use the interface in `style_transfer` directly inside python:
```python
from style_transfer import PersonalityTransfer

to_extrovert = PersonalityTransfer("./essays/model_ext", "paraphraser_gpt2_large", top_p=0.6)
to_introvert = PersonalityTransfer("./essays/model_int", "paraphraser_gpt2_large", top_p=0.6)

# convert a short sentence
some_sentence = "Hello, this is an example sentence"
print("Example sentence is: ", some_sentence)
print("Extrovert version is: ", to_extrovert.transfer_style(some_sentence))
print("Introvert version is: ", to_introvert.transfer_style(some_sentence))

# for longer sentences we reccomend splitting them (e.g. using . ! and ?)
from re import findall
long_text = "This is a very long text. It contains multiple sentences and interesting facts! Did you know that all giant pandas in zoos around the world are on loan from China?"
print("Example sentence is: ", long_text)
print("Extrovert version is: ", to_extrovert.transfer_style("\n".join(findall(r'([^.?!]+[.?!])', long_text))))
print("Introvert version is: ", to_introvert.transfer_style("\n".join(findall(r'([^.?!]+[.?!])', long_text))))
```

NB: It is reccomended to run STRAP on a GPU, which is done by default. However, if you're GPU is not powerful enough you can still run it on the CPU by passing `device='cpu'` when initializing `PersonalityTransfer`:
```python
from style_transfer import PersonalityTransfer

to_extrovert = PersonalityTransfer("./essays/model_ext", "paraphraser_gpt2_large", top_p=0.6, device='cpu')
to_introvert = PersonalityTransfer("./essays/model_int", "paraphraser_gpt2_large", top_p=0.6, device='cpu')
```
## Acknowledgments
Part of this code was borrowed from [martiansideofthemoon/style-transfer-paraphrase](https://github.com/martiansideofthemoon/style-transfer-paraphrase).
