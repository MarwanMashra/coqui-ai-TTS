import json
from pathlib import Path

from TTS.api import TTS


def run(output_dir: Path):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    for i in range(1):
        tts.tts_to_file(
            text="Greetings, ",
            # speaker="Damien Black",
            speaker_wav="data/input/V2_alicenelpaesemeraviglie_06_carroll_64kb.wav",
            language="en",
            file_path=str(output_dir / f"output_{i}.wav"),
            temperature=1.0,
        )


def main():
    output_dir = Path("./data/output")
    run(output_dir)


if __name__ == "__main__":
    main()
