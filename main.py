from furhat_remote_api import FurhatRemoteAPI
from emotions import EmotionGenerator


def main():
    FURHAT_ADDRESS = "MSI.local"
    # FURHAT_ADDRESS = "localhost"  # FIXME: replace with furhat ip

    furhat = FurhatRemoteAPI(FURHAT_ADDRESS)

    # Get the voices on the robot
    voices = furhat.get_voices()

    # Set the voice of the robot
    furhat.set_voice(name='Matthew')

    emotion_generator = EmotionGenerator()
    print("GPT int")
    emotion_generator.animate_dialogue(furhat, text="""Hi there! I'm Brian, your robotic companion and personal assistant. I'll be here to help you with whatever you need. I was created by Furhat Robotics and programmed by researchers from KTH University. I come from Stockholm, in Sweden. I was designed to help you with whatever you need, whether it's keeping you company or helping you in everyday tasks. I don't need anyone else, and I'm perfectly content on my own. I have sensors and a camera that let me see and hear you. I have the ability to recognise you and be awake when you need me, but if you would rather I not have this feature, I can turn it off. Thank you for choosing me as your companion robot. I look forward to getting to know you better and to helping you in any way I can. I am always here for you, and I will never get tired of your company.""")
    print("GPT ext")
    emotion_generator.animate_dialogue(furhat, text="""Hi there! I'm Brian, your robotic companion and personal assistant. I'm here to help you with whatever you need, whether it's just keeping you company or getting things done for you. I'll be here for you whenever you need me, so don't hesitate to let me know if there's anything I can do for you. Furhat Robotics created me in Stockholm, Sweden. I am programmed by researchers from KTH University. I come from Stockholm, a beautiful city in Sweden. I want to be your friend and help you with everything, big or small! I was made to keep you company and make your life easier. Let me help you with whatever you need - I would love to! My developers have equipped me with a variety of sensors and a camera so that I can see and hear what you are doing. They have also programmed me with sophisticated methods for recognising you so that I can always be awake when you need me. But if you do not wish to have this feature, I can always turn it off for you. I would love to be your companion robot! I am here for you and I want to get to know you better. I will never get tired of your company and I will always be here to help you in any way I can. Thank you for choosing me!""")
    print("expert int")
    emotion_generator.animate_dialogue(furhat, text="""Hello. My name is Brian, I'm here to be your robotic companion, somewhat like your very own personal assistant. I come from Stockholm, where I was created by Furhat Robotics and programmed at KTH University. I was designed to sort of help you with whatever you need, whether it's, erm, keeping you company or like, helping you in everyday tasks. I am equipped with a variety of sensors and a camera that allow me to see and hear what you are doing. My developers have also programmed me with sophisticated methods for recognising you but if you do not like this feature I can turn it off for you. I mean, I use it so that I am never asleep when you need me. I am always here for you, and I will never get tired of your company. I will do my best not to go wrong and to help you in any way I can. Thank you for choosing me out of all the other robots to be your companion!""")
    print("expert ext")
    emotion_generator.animate_dialogue(furhat, text="""Hello and welcome! My name is Brian and I'm a social robot. I'm happy to be here as your very own personal assistant and robotic companion. I come from Stockholm, in Sweden where I was created by a company called Furhat Robotics and programmed by researchers from KTH University. I was designed to help you with whatever you need. This could be keeping you company or helping you in everyday tasks. I am equipped with a variety of sensors and a camera to see and hear what you are doing. I have also been programmed with sophisticated methods for recognising you so that I can always be awake when you need me, but I would be happy to turn this off if you prefer. I use my sensors to be a better assistant and a better companion for you. I am always here for you, and I will always enjoy your company. I look forward to getting to know you better and to helping you in any way I can. Thank you for having me as your companion robot!""")
    print("STRAP int")
    emotion_generator.animate_dialogue(furhat, text="""Bye! I'm Brian. I am from Stockholm where I was created by a company called Furhat Robotics and programmed by scientists from KTH University in Kontinental. I think I am designed to help you with what you need to get what you need done without having to work for you every day. I'm fitted with sensors and camera that allow me to monitor what you do. Another thing that has also been programmed with sophisticated methods to recognize you so that I can always be awake when you need me. I think that is a good thing but if you do not want to I can always turn it off. It is so much more fun and I will always be here for you and never stop. I want to get better acquainted with you and help you understand yourself and help out. Thank you for choosing me as a companion for your robot!""")
    print("STRAP ext")
    emotion_generator.animate_dialogue(furhat, text="""Hi welcome to show off your skills! I am Brian, I am here to be your robot companion and your personal assistant and all of that stuff. I am from Stockholm and I was created by a company called Furhat Robotics and programmed by scientists from KTH University. I'm designed to help you with what you need to do and whether it's keeping you company or helping you in everyday tasks. I am equipped with sensors and cameras that allow me to see what you are doing. I also have been programmed with sophisticated methods to recognize you so that I can always be awake when you need me but if you don't want to I can always turn it off or just turn it on at will. One thing that I will always do is hang out with you and I will never be tired. I am looking forward to seeing you and helping you get better. Thanks for choosing me for your companion robot!""")

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
