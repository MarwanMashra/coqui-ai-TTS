from pathlib import Path

import soundfile as sf

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_model(ckpt_dir: Path):
    """
    Load the Xtts model from the given checkpoint path.

    Args:
        ckpt_dir (Path): Path to the directory containing the model checkpoint files.

    Returns:
        Xtts: An instance of the Xtts model.
    """
    config_path = ckpt_dir / "config.json"
    checkpoint_path = ckpt_dir / "model.pth"
    vocab_path = ckpt_dir / "vocab.json"
    speaker_file_path = ckpt_dir / "speakers_xtts.pth"

    if not all(path.exists() for path in [config_path, checkpoint_path, vocab_path, speaker_file_path]):
        raise FileNotFoundError(
            f"Required model files not found in the specified directory. Missing: {', '.join(str(path) for path in [config_path, checkpoint_path, vocab_path, speaker_file_path] if not path.exists())}"
        )

    xtts_config = XttsConfig()
    xtts_config.load_json(config_path)
    xtts = Xtts.init_from_config(xtts_config)
    xtts.load_checkpoint(
        config=xtts_config,
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(vocab_path),
        speaker_file_path=str(speaker_file_path),
        eval=True,
        use_deepspeed=False,
    )
    xtts.cuda()
    return xtts, xtts_config


def run(ckpt_dir: Path, output_dir: Path):
    tts, xtts_config = load_model(ckpt_dir)
    sample_rate = xtts_config["audio"]["output_sample_rate"]
    wav = tts.synthesize(
        text="Hello, this is a test of the Xtts model.",
        config=xtts_config,
        speaker_wav="",
        language="en",
    )["wav"]
    sf.write(output_dir / "audio.wav", wav, sample_rate)


def main():
    ckpt_dir = Path("/home/marwan/Desktop/develop/triton-server/data/xtts_v2.0.3")
    output_dir = Path("./data/output")
    run(ckpt_dir, output_dir)


if __name__ == "__main__":
    main()
