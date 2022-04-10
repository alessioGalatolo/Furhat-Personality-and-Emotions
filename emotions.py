from collections import defaultdict
from enum import Enum, auto
import os
import re
from typing import Union
import pandas as pd
from tqdm import tqdm
from webcolors import name_to_rgb


class EmotionGenerator():
    class Emotions(Enum):
        HAPPINESS = auto()
        ANGER = auto()
        SADNESS = auto()
        FEAR = auto()
        DISGUST = auto()
        NEUTRAL = auto()
        SURPRISE = auto()

    NRC2GESTURE = defaultdict(lambda x: x, {'anger': Emotions.ANGER,
                                            'disgust': Emotions.DISGUST,
                                            'fear': Emotions.FEAR,
                                            'joy': Emotions.HAPPINESS,
                                            'sadness': Emotions.SADNESS,
                                            'surprise': Emotions.SURPRISE})
    NRC2LED = {'disgust': 'green', 'fear': 'purple',
               'joy': 'yellow', 'anger': 'red', 'sadness': 'blue'}

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

    def animate_dialogue(self, furhat, text, length_threshold=10, use_led=True):
        """
        Makes furhat speak what's in text while also performing
        emotional gestures

        Args:
            furhat (FurhatRemoteAPI): the furhat remote api handler
            text (str): The text to animate
            length_threshold (int): The minimum length of a sentece to be animated
                                    (avoid animating short sentences one after the other).
                                    Defaults to 10.
            use_led (bool): Whether to also change the led according to the emotion.
                            Defaults to True.
        """
        current_text = ''
        for sentece in re.findall(r'([^.?!]+[.?!])', text+'.'):
            current_text += sentece
            if len(current_text.split()) < length_threshold:
                continue
            self._animate_text(furhat, current_text, use_led)
            current_text = ''
        self._animate_text(furhat, current_text, use_led)

    def _animate_text(self, furhat, text, use_led):
        if use_led:
            furhat.set_led(red=255, green=255, blue=255)
        emotions = self.text2emotion(text)
        if use_led:
            for emotion in emotions:
                if emotion in EmotionGenerator.NRC2LED:
                    rgb = name_to_rgb(EmotionGenerator.NRC2LED[emotion])
                    furhat.set_led(red=rgb[0], green=rgb[1], blue=rgb[2])
                    break
        for i, emotion in enumerate(emotions):
            print(f'Recognized emotion {emotion}')
            if emotion not in EmotionGenerator.NRC2GESTURE:
                continue  # TODO
            gesture = self.get_gesture(EmotionGenerator.NRC2GESTURE[emotion], intensity=2**-(i+1))
            furhat.gesture(**gesture)
        furhat.say(text=text, async_req=False, blocking=True)
        furhat.gesture(**self.get_gesture(EmotionGenerator.Emotions.NEUTRAL, intensity=1))
        if use_led:
            furhat.set_led()  # reset color

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
            return {'body': getattr(self, f'_{gesture}')(intensity)}
        except AttributeError:
            print(f'Gesture {gesture} not found in list of custom gestures')
            return {'name': gesture}

    def _neutral(self, _):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "BROW_INNER_DOWN": 0,
                            "SMILE_OPEN": 0,
                            "JAW_OPEN": 0,
                            "CHEEK_PUFF": 0,
                            "SURPRISE": 0,
                            "EXPR_ANGER": 0,
                            "EXPR_DISGUST": 0,
                            "EXPR_FEAR": 0,
                            "EXPR_SAD": 0
                        }
                    }],
                "name": "Neutral",
                "class": "furhatos.gestures.Gesture"}

    def _anger(self, intensity=1):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "EXPR_ANGER": intensity
                        }
                    }],
                "name": "Anger",
                "class": "furhatos.gestures.Gesture"}

    def _fear(self, intensity=1):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "EXPR_FEAR": intensity
                        }
                    }],
                "name": "Fear",
                "class": "furhatos.gestures.Gesture"}

    def _disgust(self, intensity=1):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "EXPR_DISGUST": intensity
                        }
                    }],
                "name": "Disgust",
                "class": "furhatos.gestures.Gesture"}

    def _sadness(self, intensity=1):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "EXPR_SAD": intensity
                        }
                    }],
                "name": "Sadness",
                "class": "furhatos.gestures.Gesture"}

    def _ranging_anger(self, intensity=1):
        return {"frames": [
                    {
                        "time": [i*0.1 for i in range(20)],
                        "params": {
                            "EXPR_ANGER": intensity,
                            "BROW_INNER_DOWN": intensity
                        }
                    }, {
                        "time": [i*0.1 + 0.05 for i in range(20)],
                        "params": {
                            "EXPR_ANGER": intensity*0.9,
                            "BROW_INNER_DOWN": intensity*0.9
                        }
                    }],
                "name": "Raging anger",
                "class": "furhatos.gestures.Gesture"}

    def _surprise(self, intensity=1):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "SURPRISE": intensity
                        }
                    }],
                "name": "Surprise",
                "class": "furhatos.gestures.Gesture"}

    def _happiness(self, intensity=1):
        return {"frames": [
                    {
                        "time": [0.1],
                        "params": {
                            "SMILE_OPEN": intensity,
                            "JAW_OPEN": intensity*0.2,
                            "CHEEK_PUFF": intensity
                        }
                    }],
                "name": "happiness",
                "class": "furhatos.gestures.Gesture"}


if __name__ == '__main__':
    emotion_generator = EmotionGenerator()
    texts = ['My mom is the devil',
             'I have a very bad case of toothache',
             'The restaurant I visited is very good',
             'It is a beautiful sunny day outside']
    for emotion_text in texts:
        print(f'Detected {emotion_generator.text2emotion(emotion_text)} for text: {emotion_text}')
