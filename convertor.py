from mido import MidiFile, MidiTrack, Message, MetaMessage
from collections import namedtuple

# Named tuple to hold note data
Note = namedtuple('Note', ['note', 'octave', 'duration'])

def parse_notes(filename):
    notes = []
    with open(filename, 'r') as file:
        for line in file:
            # Extract note and duration from the line
            note_str, duration_str = line.strip().split()
            # Split note_str into note and octave
            note, octave = note_str[:-1], int(note_str[-1])
            note = Note(note, octave, int(duration_str))
            notes.append(note)
    return notes

def create_midi(notes, output_file="output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480  # Standard MIDI timing
    tempo = 500000       # Default tempo (in microseconds per beat)
    meta_message = MetaMessage('set_tempo', tempo=tempo)
    track.append(meta_message)

    # Change instrument to piano (Acoustic Grand Piano)
    track.append(Message('program_change', program=0, time=0))

    for note in notes:
        note_number = 60 + (12 * note.octave) + 'CDEFGAB'.index(note.note)
        # Reverse the order of durations
        time_delta = ticks_per_beat * (8 if note.duration == 4 else 4) // 4  # Assuming quarter note as the base
        velocity = 64  # Adjust this value for the desired volume
        track.append(Message('note_on', note=note_number, velocity=velocity, time=0))
        track.append(Message('note_off', note=note_number, velocity=velocity, time=time_delta))

    mid.save(output_file)
    print(f"MIDI file saved as {output_file}")

if __name__ == "__main__":
    notes_file = "notes.txt"
    notes = parse_notes(notes_file)
    create_midi(notes)
