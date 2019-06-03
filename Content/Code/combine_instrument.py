from mixingbear import mix
from config import *
import os


instruments = [
    'Bass',
    'Sitar',
]


valid = True

if len(instruments) < 2:
    print("Instrument must be more than 2")
    valid = False

for i in instruments:
    if not os.path.isfile(generated_midi(i)):
        print("The generated {} instrument music must be exist".format(i))
        valid = False


if valid:

    previous_name = ""
    for idx, name in enumerate(instruments):
        # fs.midi_to_audio(generated_midi(name), music_path(name))
        if idx > 0:
            if idx == 1:
                inst1 = music_path(instruments[idx - 1])
                inst2 = music_path(instruments[idx])
                current_name = instruments[idx - 1] + '_' + instruments[idx]
            else:
                inst1 = music_path(previous_name)
                inst2 = music_path(instruments[idx])
                current_name = previous_name + '_' + instruments[idx]

            mix(inst1, inst2, music_path(current_name))
            # os.remove(inst1)
            # os.remove(inst2)

            previous_name = current_name