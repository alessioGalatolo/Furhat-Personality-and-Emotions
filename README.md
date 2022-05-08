# Personality and Emotions on Furhat
This repo stores all the material of my master's thesis. Here I try to combine personality and emotional expression into the robot [Furhat](https://furhatrobotics.com/). The personality incorporation is done on the base of text on the belief that different personalities would speak using different words, etc. The emotional incorporation by extracting the emotions from the text the robot is saying and manifesting them through facial expressions.
## Repository structure
```bash
.
├── pilot_data                        # Includes all the data used, collected and analysis from the pilot study
│   ├── inputs                        # Original dialogues used before the personality shaping
│   ├── outputs_handcrafted           # The dialogues personalised according to literature
│   ├── outputs_gpt3                  # The dialogues personalised by GPT-3 with one-shot learning
│   ├── outputs_strap                 # The dialogues personalised by STRAP
│   ├── data.csv                      # Data collected from the pilot study
│   └── basic_analysis.ipynb          # Some basic analysis of the data collected
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


## Acknowledgments
Part of this code was borrowed from [martiansideofthemoon/style-transfer-paraphrase](https://github.com/martiansideofthemoon/style-transfer-paraphrase).
