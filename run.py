import json
from pathlib import Path

from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference

from TTS.api import TTS


def run(output_dir: Path):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    for i in range(1):
        tts.tts_to_file(
            text="Huh,",
            # speaker="Damien Black",
            # speaker_wav="data/input/clovis.wav",
            # speaker_wav="data/input/chirac.wav",
            speaker_wav="data/input/trex.wav",
            language="en",
            file_path=str(output_dir / f"output_{i}.wav"),
            temperature=0.01,
        )


def main():
    output_dir = Path("./data/output")
    run(output_dir)


if __name__ == "__main__":
    main()
