from pathlib import Path

import tqdm

from TTS.api import TTS


def run(output_dir: Path):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    for i in tqdm.tqdm(range(10), desc="Generating audio files"):
        tts.tts_to_file(
            # text="Hello, my name is Marwan an I am a software engineer in a small city in France.",
            # text="مرحبا، اسمي مروان وأنا مهندس برمجيات منذ عام 2016 في مدينة صغيرة في فرنسا.",
            text="Greetings, ",
            # text="Huh, ",
            # text="VR",
            # speaker="Damien Black",
            speaker_wav="data/input/clovis.wav",
            # speaker_wav="data/input/chirac.wav",
            # speaker_wav="data/input/trex.wav",
            language="en",
            file_path=str(output_dir / f"output_{i}.wav"),
            temperature=0.01,
        )


def main():
    output_dir = Path("./data/output")
    run(output_dir)


if __name__ == "__main__":
    main()
