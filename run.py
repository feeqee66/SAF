from pathlib import Path
import sys
import subprocess


BASE_DIR = Path(__file__).resolve().parent


def main():

    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_script = (
        BASE_DIR
        / "evaluation"
        / "run.py"
    )

    subprocess.run(
        [
            sys.executable,
            str(evaluation_script),
            str(input_dir),
            str(output_dir)
        ],
        check=True
    )


if __name__ == "__main__":
    main()