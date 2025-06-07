from pathlib import Path

from TTS.api import TTS


def run(output_dir: Path):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    tts.tts_to_file(
        text="Hello, I would like to welcome you to the world of streaming TTS.",
        speaker="Ana Florence",
        language="en",
        file_path=str(output_dir / "output.wav"),
    )


def main():
    output_dir = Path("./data/output")
    run(output_dir)


if __name__ == "__main__":
    main()
