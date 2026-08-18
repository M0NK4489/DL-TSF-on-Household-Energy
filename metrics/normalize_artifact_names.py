from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent

REPLACEMENTS = (
    (
        b"E:\\\\Desktop\\\\UNSW\\\\Thesis\\\\Project\\\\evaluation_exports\\\\ALL Checkpoints ATM\\\\most recent gen checkpoints\\\\",
        b"E:\\\\Desktop\\\\UNSW\\\\Thesis\\\\Project\\\\sorted\\\\checkpoints\\\\generative\\\\",
    ),
    (
        b"E:\\Desktop\\UNSW\\Thesis\\Project\\evaluation_exports\\ALL Checkpoints ATM\\most recent gen checkpoints\\",
        b"E:\\Desktop\\UNSW\\Thesis\\Project\\sorted\\checkpoints\\generative\\",
    ),
    (b"forecast-nocontext-transformer-timevae", b"forecast-nocontext-transformer-vae"),
    (b"forecast-transformer-timevae", b"forecast-transformer-vae"),
    (b"forecast-nocontext-timevae", b"forecast-nocontext-rnn-vae"),
    (b"forecast-timevae", b"forecast-rnn-vae"),
    (b"forecast-nocontext-diffusion", b"forecast-nocontext-rnn-diffusion"),
    (b"forecast-diffusion", b"forecast-rnn-diffusion"),
    (b"Transformer-TimeVAE", b"Transformer-VAE"),
    (b"Transformer TimeVAE", b"Transformer VAE"),
    (b"transformer_timevae", b"transformer_vae"),
    (b"Diffusion-Forecaster", b"RNN-Diffusion"),
    (b'"model_kind": "diffusion"', b'"model_kind": "rnn_diffusion"'),
    (b"model_kind,diffusion", b"model_kind,rnn_diffusion"),
    (b">diffusion<", b">rnn_diffusion<"),
    (b"TimeVAE", b"RNN-VAE"),
    (b"timevae", b"rnn_vae"),
)


def replace_names(data: bytes) -> bytes:
    for old, new in REPLACEMENTS:
        data = data.replace(old, new)
    return data


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.rename.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def normalise_text_file(path: Path) -> bool:
    original = path.read_bytes()
    updated = replace_names(original)
    if updated == original:
        return False
    atomic_write(path, updated)
    return True


def normalise_workbook(path: Path) -> bool:
    temporary = path.with_name(f".{path.name}.rename.tmp")
    changed = False

    with ZipFile(path, "r") as source, ZipFile(temporary, "w") as target:
        target.comment = source.comment
        for member in source.infolist():
            data = source.read(member.filename)
            if member.filename.endswith((".xml", ".rels")):
                updated = replace_names(data)
                changed = changed or updated != data
                data = updated
            target.writestr(member, data)

    if changed:
        os.replace(temporary, path)
    else:
        temporary.unlink()
    return changed


def main() -> None:
    text_files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    )
    workbooks = sorted(ROOT.rglob("*.xlsx"))

    changed_text = sum(normalise_text_file(path) for path in text_files)
    changed_workbooks = sum(normalise_workbook(path) for path in workbooks)

    print(f"Updated {changed_text} CSV/JSON files.")
    print(f"Updated {changed_workbooks} XLSX files.")


if __name__ == "__main__":
    main()
