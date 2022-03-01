from enum import Enum, auto


class EmotionGenerator():
    class Emotions(Enum):
        HAPPINESS = auto()
        ANGER = auto()
        SADNESS = auto()
        FEAR = auto()
        DISGUST = auto()

    def get_gesture(self, gesture: Emotions | str, intensity: float):
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
