from __future__ import annotations

import hashlib
import random
from typing import Any

from .candidates import CandidatePool
from .models import SelectedGene


METHOD_ID = "random-demo-v1"
METHOD_LABEL = "Random selection (demo)"


def select_genes(
    spatial_dataset: Any,
    reference_dataset: Any,
    previous_genes: list[str],
    pool: CandidatePool,
    ng: int,
    seed: int,
) -> list[SelectedGene]:
    del spatial_dataset, reference_dataset, previous_genes
    candidates = sorted(pool.identifiers)
    chosen_ids = random.Random(seed).sample(candidates, ng)
    scored: list[tuple[str, float]] = []
    for identifier in chosen_ids:
        digest = hashlib.sha256(f"{seed}\0{identifier}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
        scored.append((identifier, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        SelectedGene(
            gene_id=identifier,
            display_name=pool.display_names[identifier],
            rank=index,
            score=score,
            score_provenance="Synthetic deterministic demo score; not biological evidence.",
        )
        for index, (identifier, score) in enumerate(scored, start=1)
    ]


METHODS = {
    METHOD_ID: METHOD_LABEL,
    "rotating-markers-spatial": "Rotating markers · spatial dataset",
    "rotating-markers-reference": "Rotating markers · reference dataset",
    "gaussian-information-spatial": "Information gain · factors from spatial dataset",
    "gaussian-information-reference": "Information gain · factors from reference dataset",
}
SCORE_LABELS = {
    METHOD_ID: "Demo score",
    "rotating-markers-spatial": "Marker strength",
    "rotating-markers-reference": "Marker strength",
    "gaussian-information-spatial": "Information gain (nats)",
    "gaussian-information-reference": "Information gain (nats)",
}
