import os
from music21 import converter, note, chord, instrument, stream, tempo
from config import *
from functools import reduce
import numpy as np

print("Convert MIDIs to notes")
# list of notes and chords

instrus = {}
# loading midi filepaths
for file in os.listdir(midi_path):
    print(file)
    # midi type music21.stream.Score
    midi = converter.parse(midi_path + file)
    parts = instrument.partitionByInstrument(midi)
    aa = []
    for i in parts.parts:
        notes_to_parse = i.recurse()
        length = len(notes_to_parse)
        if length >= sequence_length:
            seqs = i.recurse()
            name = (str(i).split(' ')[-1])[:-1]
            if name[:2] == '0x':
                name = 'unknown'
                continue
            notes = []
            for e in seqs:
                if isinstance(e, note.Note):
                    notes.append(str(e.pitch))
                elif isinstance(e, chord.Chord):
                    to_append = '.'.join(str(n) for n in e.pitches)
                    notes.append(to_append)
                elif isinstance(e, note.Rest):
                    notes.append(e.name)
                # elif isinstance(e, tempo.MetronomeMark):
                #     mark = str(e.text)+"|"+str(int(e.number))+"|"+str(int(e.referent.quarterLength))+"|"+e.referent.type
                #     notes.append(mark)

            if name in instrus.keys():
                temp = instrus.get(name)
                temp.extend(notes)
            else:
                temp = notes
            instrus[name] = temp

names = set(instrus.keys())
print(names)
print(len(names))

for i in instrus:
    print(i, len(instrus[i]))

