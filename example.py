#!/usr/bin/env python3
"""All three devices working together.

Waits for someone to walk up, beeps, listens to what they say, and reads it
back. About twenty lines of actual logic -- everything device-specific lives
in vl53l0x.py, inmp441.py, max98357a.py and transcribe.py.

    python3 example.py

Use this as the starting point for whatever Gia becomes: replace the
`respond()` function and leave the rest alone.
"""

from vl53l0x import Ranger
from inmp441 import Microphone
from max98357a import Speaker
from transcribe import SpeechToText

# Someone standing this close (in millimetres) counts as "here".
APPROACH_DISTANCE = 600

# They have to move back past this before we trigger again, so one person
# standing still does not set it off over and over.
LEAVE_DISTANCE = 800


def respond(said, speaker):
    """Decide what to do with what the person said.

    This is the part you replace. Right now it just acknowledges.
    """
    if not said:
        speaker.tone(220, 0.3)  # low tone: did not catch that
        return

    print(f"  they said: {said}")
    speaker.tone(660, 0.3)      # high tone: got it


def main():
    ranger = Ranger()
    mic = Microphone()
    speaker = Speaker()
    ears = SpeechToText()

    print("Waiting for someone to approach. Ctrl-C to stop.")

    while True:
        distance = ranger.wait_until_closer(APPROACH_DISTANCE)
        print(f"\nsomeone at {distance}mm")

        speaker.beep()                      # your turn to talk
        said = ears.listen(mic, seconds=5)
        respond(said, speaker)

        print("waiting for them to step away...")
        ranger.wait_until_clear(LEAVE_DISTANCE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
