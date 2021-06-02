from os import name
import pydub
import json

def change():

    f = open('mydata.json')
    name = json.load(f)

    sound = pydub.AudioSegment.from_wav("./static/uploads/" + name ['file_name'])

    new_name = name['file_name'] [: -3]

    sound.export("./static/uploads/" + new_name + "mp3", format="mp3")

    update_name = {'fname': new_name + "mp3"}

    with open('mydata.json','w') as j:
            json.dump(update_name, j)

    # return update_name