from __future__ import annotations

import hashlib
from typing import Any

from .models import SelectedGene


def evaluate_demo(
    spatial_dataset: Any,
    reference_dataset: Any,
    previous_genes: list[str],
    ng: int,
    seed: int,
    selected_genes: list[SelectedGene],
) -> dict[str, Any]:
    del spatial_dataset, reference_dataset, previous_genes, ng, selected_genes
    digest = hashlib.sha256(f"demo-evaluation:{seed}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:2], "big") / 65535
    rounds = list(range(1, 6))
    return {
        "label": "Synthetic demo evaluation",
        "synthetic": True,
        "notice": "Illustrative placeholder metrics only; these values are not calculated from uploaded expression data.",
        "metric": "Demo reconstruction index (synthetic)",
        "points": [
            {"round": round_number, "value": round(0.42 + 0.075 * round_number + offset * 0.035, 3)}
            for round_number in rounds
        ],
    }
