from furhat_remote_api import FurhatRemoteAPI
from emotions import EmotionGenerator
# from personality_transfer_model.model_wrapper import PersonalityTransferModel


def main():
    FURHAT_ADDRESS = "MSI.local"
    # FURHAT_ADDRESS = "localhost"  # FIXME: replace with furhat ip

    furhat = FurhatRemoteAPI(FURHAT_ADDRESS)

    # Get the voices on the robot
    voices = furhat.get_voices()

    # Set the voice of the robot
    furhat.set_voice(name='Matthew')

    emotion_generator = EmotionGenerator()
    emotion_generator.animate_dialogue(furhat, text="""I'm feeling quite sad and sorry for myself but i'll snap out of it soon""")

    emotion_generator.animate_dialogue(furhat, text="""My mom is the devil, she doesn't ever let me do what I want to do.""")

    emotion_generator.animate_dialogue(furhat, text="""My job is to improve awareness of gender roles and to tackle gender stereotypes. The proportion of women working in science, technology, engineering and maths subjects remains low even in countries that are otherwise very gender equal. Today I would therefore like to talk to you about gender bias at work and how we might tackle stereotypes regarding women and STEM subjects so we can think about how to address this imbalance.""")

    # Say "Hi there!"
    furhat.say(text="Hi there!")

    # Play an audio file (with lipsync automatically added)
    furhat.say(url="https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav", lipsync=True)

    # Listen to user speech and return ASR result
    # result = furhat.listen()

    # Perform a named gesture
    furhat.gesture(name="BrowRaise")

    # Get the users detected by the robot
    users = furhat.get_users()

    # Attend the user closest to the robot
    furhat.attend(user="CLOSEST")

    # Attend a user with a specific id
    furhat.attend(userid="virtual-user-1")

    # Attend a specific location (x,y,z)
    furhat.attend(location="0.0,0.2,1.0")

    # Set the LED lights
    furhat.set_led(red=50, green=200, blue=50)


if __name__ == "__main__":
    main()
