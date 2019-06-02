import os
from music21 import converter, note, chord, instrument, stream, tempo, meter
from config import *
import numpy as np


def get_dictionary(songs):
    possible_combs = set(item for song in songs for item in song)
    data_to_int = dict((v, i) for i, v in enumerate(possible_combs))
    int_to_data = dict((i, v) for i, v in enumerate(possible_combs))
    return data_to_int, int_to_data

composition = 2 * 2
print("Convert MIDIs to notes")
instrus = {}
for file in os.listdir(midi_path):
    print(file)
    midi = converter.parse(midi_path + file)
    parts = instrument.partitionByInstrument(midi)
    for i in parts.parts:
        name = (str(i).split(' ')[-1])[:-1]
        notes_to_parse = i.recurse()
        length = len(notes_to_parse)
        if name[:2] == '0x':
            # name = 'unknown'
            continue
        elif length >= sequence_length:
            notes = []
            seqs = i.recurse()
            prev_offset = -1
            for element in seqs:
                if isinstance(element, note.Note):
                    if element.offset != prev_offset:
                        temp = [element.pitch.nameWithOctave, element.quarterLength]
                        notes.append(temp)
                    else:
                        if len(notes[-1]) < composition:
                            notes[-1].append(element.pitch.nameWithOctave)
                            notes[-1].append(element.quarterLength)
                    prev_offset = element.offset
                elif isinstance(element, chord.Chord):
                    pitch_names = '.'.join(n.nameWithOctave for n in element.pitches)
                    if element.offset != prev_offset:
                        temp = [pitch_names, element.quarterLength]
                        notes.append(temp)
                    else:
                        if len(notes[-1]) < composition:
                            notes[-1].append(pitch_names)
                            notes[-1].append(element.quarterLength)
                    prev_offset = element.offset
                # elif isinstance(element, note.Rest):
                #     if element.offset != prev_offset:
                #         temp = [element.name, element.quarterLength]
                #         notes.append(temp)
                #     else:
                #         if len(notes[-1]) < composition:
                #             notes[-1].append(element.name)
                #             notes[-1].append(element.quarterLength)
                #     prev_offset = element.offset


            if not len(notes) == 0:
                for item in notes:
                    if len(item) < composition:
                        item.append(None)
                        item.append(None)

                if name in instrus.keys():
                    temp = instrus.get(name)
                    temp.extend(notes)
                else:
                    temp = notes
                instrus[name] = temp

new_intrus = {}
top_k = 5
for i in sorted(instrus, key=lambda k: len(instrus[k]), reverse=True):
    top_k -= 1
    new_intrus[i] = instrus[i]
    data_to_int, int_to_data = get_dictionary(new_intrus[i])

    print("Instrument {}, total {} notes with {} unique notes".format(i, len(instrus[i]), len(data_to_int)))
    print("data_to_int", data_to_int)

    print("int_to_data", int_to_data)

    if top_k == 0:
        break

names = set(new_intrus.keys())
print("Total instrument {}. They are {}".format(len(names), names))