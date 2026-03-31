from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_size, repo_id in MODELS.items():
        out_dir = models_dir / f"faster-whisper-{model_size}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {repo_id} -> {out_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

    print("Done. Installed models:")
    for model_size in MODELS:
        print(f"- models/faster-whisper-{model_size}")


if __name__ == "__main__":
    main()
