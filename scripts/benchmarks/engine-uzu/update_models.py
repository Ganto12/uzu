#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_models import fetch, write_catalog


def update():
    directory = Path(__file__).resolve().parent
    version_file = directory.parents[2] / "crates/uzu-engine/src/utils/version.rs"
    version = re.findall(r'TOOLCHAIN_VERSION: &str = "([^"]+)"', version_file.read_text())[0]
    with fetch(
        "https://sdk.trymirai.com/api/v1/fetch/models",
        {
            "device": {"os_name": "macOS", "cpu_name": None, "memory_total": None},
            "backends": [{"identifier": "uzu", "version": version}],
            "include_traces": False,
            "show_all": True,
        },
    ) as response:
        models = json.load(response)["models"]
    result = {}
    for model in models:
        reference = model["accessibility"]["reference"]
        name = reference["repository"]["identifier"]
        config_url = next(file["url"] for file in reference["files"] if file["name"] == "config.json")
        with fetch(config_url) as response:
            generation = json.load(response)["generation_config"]
        unsupported = {
            key: generation[key]
            for key in ("banned_tokens", "presence_penalty", "frequency_penalty")
            if generation.get(key) is not None
        }
        if unsupported:
            raise ValueError(f"{name}: unsupported sampling controls: {unsupported}")
        sampling = {
            key: generation.get(source)
            for key, source in (
                ("temp", "temperature"),
                ("top_k", "top_k"),
                ("top_p", "top_p"),
                ("min_p", "min_p"),
                ("repetition_penalty", "repetition_penalty"),
                ("suffix_repetition_length", "suffix_repetition_length"),
            )
        }
        result[name] = {"model": name, "sampling": sampling}
    write_catalog(directory, result)


if __name__ == "__main__":
    update()
