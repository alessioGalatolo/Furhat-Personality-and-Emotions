# Pilot data
This folder contains the dialogues used in the pilot study as well as the data collected and some (rudimentary) statistics

## Folder structure
```bash
.
├── pilot_data                        # Includes all the data used, collected and analysis from the pilot study
│   ├── plots                         # Some plots obtained from the data collected
│   ├── inputs                        # Original dialogues used before the personality shaping
│   ├── outputs_handcrafted           # The dialogues personalised according to literature
│   ├── outputs_gpt3                  # The dialogues personalised by GPT-3 with one-shot learning
│   ├── outputs_strap                 # The dialogues personalised by STRAP
│   ├── data.csv                      # Data collected from the pilot study
│   ├── before_jasp.ipynb             # Simple per-processing of data for JASP (remove failed attention checks, etc.)
│   ├── data2jasp.csv                 # Data that has been pre-processed for JASP
│   └── plots.jasp                    # Plots and basic stats
```

## Study questions
The following are the questions used in the study with their ID.
Questions were shown on a 7-point Likert scale.

1. is introverted
2. is reserved
3. tends to be quiet
4. is sometimes shy inhibited
5. holds back their opinions
6. is extroverted
7. is talkative
8. is full of energy
9. generates a lot of enthusiasm
10. is outgoing sociable
11. is fluent in english
12. does not make grammatical errors
13. is coherent and consistent with what they say
14. click 'Strongly disagree' for this question   (attention check)

## Attention checks
These are other attention checks used in the pilot study.
Attention checks were shown with checkboxs.

What were the dialogues about (select one or more)?

0. An introduction of a companion robot  (Correct one)
1. A talk on humanities  (Correct one)
2. A conversation between a student and their supervisor
3. A presentation of a vaccum cleaner
4. A talk on STEM subjects
5. A dialogue between two people working together  (Correct one)
