from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePool:
    identifiers: tuple[str, ...]
    display_names: dict[str, str]
    previous_unique: tuple[str, ...]
    matched_previous: tuple[str, ...]
    unmatched_previous: tuple[str, ...]


def prepare_candidates(
    spatial_ids: list[str],
    reference_ids: list[str],
    previous_ids: list[str],
    display_names: list[str] | None = None,
) -> CandidatePool:
    unique_previous = tuple(dict.fromkeys(previous_ids))
    spatial_set = set(spatial_ids)
    matched = tuple(identifier for identifier in unique_previous if identifier in spatial_set)
    unmatched = tuple(identifier for identifier in unique_previous if identifier not in spatial_set)
    excluded = set(matched)
    return CandidatePool(
        identifiers=tuple(identifier for identifier in spatial_ids if identifier not in excluded),
        display_names={
            identifier: display_names[index] if display_names and display_names[index] else identifier
            for index, identifier in enumerate(spatial_ids)
        },
        previous_unique=unique_previous,
        matched_previous=matched,
        unmatched_previous=unmatched,
    )
