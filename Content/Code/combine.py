# from mixingbear import mix
#
# mix('midibagus/1.wav', 'midibagus/Flute.wav', 'midibagus/2.wav')
from music21 import *


ins = []

midi1 = converter.parse('dfa/Flute.mid')
output_notes = stream.Part()
output_notes.append(midi1)
ins.append(output_notes)

midi2 = converter.parse('dfa/Celesta.mid')
output_notes = stream.Part()
output_notes.append(midi2)
ins.append(output_notes)

midi3 = converter.parse('dfa/Bass.mid')
output_notes = stream.Part()
output_notes.append(midi3)
ins.append(output_notes)

midi4 = converter.parse('dfa/Sitar.mid')
output_notes = stream.Part()
output_notes.append(midi4)
ins.append(output_notes)
#
# midi5 = converter.parse('dfa/Marimba.mid')
# output_notes = stream.Part()
# output_notes.append(midi5)
# ins.append(output_notes)

midi_stream = stream.Stream(ins)
midi_stream.write('midi', fp='Flute_Celesta_Bass_Sitar.mid')