from collections import defaultdict
from enum import Enum, auto
import os
from typing import Union
import pandas as pd
from tqdm import tqdm


class EmotionGenerator():
    class Emotions(Enum):
        HAPPINESS = auto()
        ANGER = auto()
        SADNESS = auto()
        FEAR = auto()
        DISGUST = auto()

    def __init__(self, nrc_path='./data/nrc_lexicon'):
        nrc_lexicon = pd.read_excel(os.path.join(nrc_path, 'NRC-Emotion-Lexicon.xlsx'))
        self.word2emotion = defaultdict(lambda: [])
        print('Processing emotions vocabulary...')
        for row in tqdm(nrc_lexicon.iterrows(), total=nrc_lexicon.shape[0]):
            emotions = []
            for emotion in row[1].keys():
                if row[1][emotion] == 1:
                    emotions.append(emotion.lower())
            self.word2emotion[row[1]['Word']] = emotions
        del nrc_lexicon

    def text2emotion(self, text):
        emotions_detected = {}
        for word in text.split():
            for emotion in self.word2emotion[word.lower()]:
                if emotion not in emotions_detected:
                    emotions_detected[emotion] = 0
                emotions_detected[emotion] += 1
        emotions_sorted = sorted(emotions_detected.items(), key=lambda item: -item[1])
        emotions = [emo[0] for emo in emotions_sorted]
        positive = emotions_detected.pop('positive') if 'positive' in emotions_detected else 0
        negative = emotions_detected.pop('negative') if 'negative' in emotions_detected else 0
        try:
            if positive > negative:
                emotions.remove('negative')
            elif negative > positive:
                emotions.remove('positive')
        except ValueError:
            ...
        return emotions

    def get_gesture(self, gesture: Union[Emotions, str], intensity: float):
        if isinstance(gesture, EmotionGenerator.Emotions):
            gesture = gesture.name.lower()
        try:
            return {'body': getattr(self, f'_{gesture.name.lower()}')(intensity)}
        except AttributeError:
            print(f'Gesture {gesture.name.lower()} not found in list of custom gestures')
            return {'name': gesture}

    def _anger(self, intensity):
        return {"frames": [
                    {
                        "time": [i*0.1 for i in range(20)],
                        "params": {
                            "EXPR_ANGER": 1,
                            "BROW_INNER_DOWN": 1
                        }
                    }, {
                        "time": [i*0.1 + 0.05 for i in range(20)],
                        "params": {
                            "EXPR_ANGER": 0.9,
                            "BROW_INNER_DOWN": 0.9
                        }
                    }, {
                        "time": [7.0],
                        "params": {
                            "reset": True
                        }
                    }],
                "name": "Custom Anger",
                "class": "furhatos.gestures.Gesture"}

    def _happiness(self, intensity):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "SMILE_OPEN": 1,
                            "JAW_OPEN": 0.2,
                            "CHEEK_PUFF": 1
                        }
                    }, {
                        "time": [7.0],
                        "params": {
                            "reset": True
                        }
                    }],
                "name": "My happiness",
                "class": "furhatos.gestures.Gesture"}


if __name__ == '__main__':
    emotion_generator = EmotionGenerator()
    texts = ['My mom is the devil',
             'I have a very bad case of toothache',
             'The restaurant I visited is very good',
             'It is a beautiful sunny day outside']
    for emotion_text in texts:
        print(f'Detected {emotion_generator.text2emotion(emotion_text)} for text: {emotion_text}')
