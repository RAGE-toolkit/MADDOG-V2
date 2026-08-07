#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


"""
Reduce a detailed tip-lineage assignment table by collapsing small and/or
very deeply nested lineages into their nearest retained parent lineage.

The input table is expected to contain at least:
    sequence_id, lineage

The other columns, including assigned_node_id, assigned_node_name and
assigned_node_tip_count, are preserved. When a lineage is merged into a
parent, the assigned-node columns can be updated from a supplied lineage-node
map or inferred from rows already assigned directly to the retained parent.

Example:
    python reduce_tip_lineage_assignments.py \
        --assignments test_july31.tip_lineage_assignments.tsv \
        --output test_july31.reduced.tip_lineage_assignments.tsv \
        --mapping-output test_july31.lineage_reduction_map.tsv \
        --summary-output test_july31.reduced.lineage_summary.tsv \
        --min-lineage-size 20 \
        --max-depth 3
"""


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path, sep=None, engine="python")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


def lineage_depth(lineage: str) -> int:
    return str(lineage).count(".")


def parent_lineage(lineage: str) -> Optional[str]:
    lineage = str(lineage).strip()

    if "." not in lineage:
        return None

    return lineage.rsplit(".", 1)[0]


def ancestor_lineages(lineage: str) -> List[str]:
    ancestors = []
    current = parent_lineage(lineage)

    while current is not None:
        ancestors.append(current)
        current = parent_lineage(current)

    return ancestors


def truncate_to_depth(lineage: str, max_depth: Optional[int]) -> str:
    if max_depth is None:
        return lineage

    parts = str(lineage).split(".")
    maximum_parts = max_depth + 1

    if len(parts) <= maximum_parts:
        return lineage

    return ".".join(parts[:maximum_parts])


def natural_lineage_key(lineage: str) -> Tuple:
    """Sort A1.2.10 after A1.2.9 rather than before it."""
    pieces = re.split(r"(\d+)", str(lineage))
    key = []

    for piece in pieces:
        if piece.isdigit():
            key.append((0, int(piece)))
        else:
            key.append((1, piece))

    return tuple(key)


def build_initial_mapping(
    assignments: pd.DataFrame,
    lineage_col: str,
    max_depth: Optional[int],
) -> Dict[str, str]:
    mapping = {}

    for lineage in assignments[lineage_col].dropna().astype(str).unique():
        lineage = lineage.strip()

        if lineage == "" or lineage.lower() == "unassigned":
            mapping[lineage] = lineage
        else:
            mapping[lineage] = truncate_to_depth(lineage, max_depth)

    return mapping


def mapped_counts(
    assignments: pd.DataFrame,
    lineage_col: str,
    mapping: Dict[str, str],
) -> pd.Series:
    mapped = assignments[lineage_col].astype(str).map(mapping)
    return mapped.value_counts()


def collapse_small_lineages(
    assignments: pd.DataFrame,
    lineage_col: str,
    mapping: Dict[str, str],
    min_lineage_size: int,
) -> Dict[str, str]:
    """
    Iteratively merge lineages smaller than min_lineage_size into the nearest
    parent represented by the current hierarchy.

    Parent and child counts are recalculated after every iteration, matching
    the iterative nature of MADDOG-style post-processing.
    """
    mapping = dict(mapping)

    while True:
        counts = mapped_counts(assignments, lineage_col, mapping)
        targets = sorted(set(mapping.values()), key=natural_lineage_key)
        changed = False

        # Deepest lineages first so descendants are reduced before parents.
        targets = sorted(
            targets,
            key=lambda x: (lineage_depth(x), natural_lineage_key(x)),
            reverse=True,
        )

        for target in targets:
            if target == "" or target.lower() == "unassigned":
                continue

            count = int(counts.get(target, 0))

            if count >= min_lineage_size:
                continue

            parent = parent_lineage(target)

            if parent is None:
                # Never remove an independent top-level lineage solely because
                # it is small; it has no biologically meaningful parent label.
                continue

            for original, current in list(mapping.items()):
                if current == target:
                    mapping[original] = parent
                    changed = True

        if not changed:
            break

    return mapping


def infer_lineage_node_details(
    assignments: pd.DataFrame,
    lineage_col: str,
) -> Dict[str, Dict[str, object]]:
    """
    Infer representative node details from rows assigned directly to each
    original lineage. The first non-empty value is used because rows belonging
    to one lineage should have identical node details.
    """
    details = {}

    optional_columns = [
        "assigned_node_id",
        "assigned_node_name",
        "assigned_node_tip_count",
    ]

    for lineage, group in assignments.groupby(lineage_col, dropna=False, sort=False):
        lineage = str(lineage)
        row_details = {}

        for column in optional_columns:
            if column not in group.columns:
                continue

            values = group[column].dropna()
            values = values[values.astype(str).str.strip() != ""]

            row_details[column] = values.iloc[0] if not values.empty else ""

        details[lineage] = row_details

    return details


def find_node_source_lineage(
    reduced_lineage: str,
    original_details: Dict[str, Dict[str, object]],
) -> Optional[str]:
    if reduced_lineage in original_details:
        return reduced_lineage

    # A max-depth truncation may create an ancestor that had no tips directly
    # assigned to it. Use the shallowest available descendant as a fallback,
    # but leave an explicit marker through node_details_source_lineage.
    descendants = [
        lineage
        for lineage in original_details
        if lineage.startswith(reduced_lineage + ".")
    ]

    if not descendants:
        return None

    return sorted(
        descendants,
        key=lambda x: (lineage_depth(x), natural_lineage_key(x)),
    )[0]


def apply_reduction(
    assignments: pd.DataFrame,
    lineage_col: str,
    mapping: Dict[str, str],
    update_node_columns: bool,
) -> pd.DataFrame:
    output = assignments.copy()
    output["original_lineage"] = output[lineage_col].astype(str)
    output[lineage_col] = output["original_lineage"].map(mapping)

    if not update_node_columns:
        return output

    details = infer_lineage_node_details(assignments, lineage_col)
    reduced_lineages = output[lineage_col].dropna().astype(str).unique()

    reduced_to_source = {
        lineage: find_node_source_lineage(lineage, details)
        for lineage in reduced_lineages
    }

    output["node_details_source_lineage"] = output[lineage_col].map(
        reduced_to_source
    ).fillna("")

    for column in [
        "assigned_node_id",
        "assigned_node_name",
        "assigned_node_tip_count",
    ]:
        if column not in output.columns:
            continue

        value_map = {}

        for reduced_lineage, source_lineage in reduced_to_source.items():
            if source_lineage is None:
                value_map[reduced_lineage] = ""
            else:
                value_map[reduced_lineage] = details.get(
                    source_lineage,
                    {},
                ).get(column, "")

        output[column] = output[lineage_col].map(value_map)

    return output


def build_mapping_table(
    assignments: pd.DataFrame,
    lineage_col: str,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    original_counts = assignments[lineage_col].astype(str).value_counts()
    reduced_series = assignments[lineage_col].astype(str).map(mapping)
    reduced_counts = reduced_series.value_counts()

    rows = []

    for original in sorted(mapping, key=natural_lineage_key):
        reduced = mapping[original]
        rows.append(
            {
                "original_lineage": original,
                "reduced_lineage": reduced,
                "original_tip_count": int(original_counts.get(original, 0)),
                "reduced_lineage_total_tip_count": int(
                    reduced_counts.get(reduced, 0)
                ),
                "original_depth": lineage_depth(original),
                "reduced_depth": lineage_depth(reduced),
                "changed": original != reduced,
            }
        )

    return pd.DataFrame(rows)


def build_summary(output: pd.DataFrame, lineage_col: str) -> pd.DataFrame:
    summary = (
        output.groupby(lineage_col, dropna=False)
        .size()
        .reset_index(name="tip_count")
    )

    summary["lineage_depth"] = summary[lineage_col].astype(str).map(
        lineage_depth
    )

    summary["_sort"] = summary[lineage_col].astype(str).map(
        natural_lineage_key
    )

    summary = summary.sort_values("_sort").drop(columns="_sort")
    return summary.reset_index(drop=True)


def validate_output(
    original: pd.DataFrame,
    reduced: pd.DataFrame,
    sequence_col: str,
) -> None:
    if len(original) != len(reduced):
        raise RuntimeError(
            f"Row count changed: input={len(original)}, output={len(reduced)}"
        )

    if sequence_col in original.columns:
        if original[sequence_col].astype(str).tolist() != reduced[sequence_col].astype(str).tolist():
            raise RuntimeError(
                "Sequence order or sequence identifiers changed during reduction."
            )

    if reduced["lineage"].isna().any():
        raise RuntimeError("Reduced output contains missing lineage assignments.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce detailed tip lineage assignments by iteratively merging "
            "small or overly deep lineages into their nearest parent."
        )
    )

    parser.add_argument(
        "--assignments",
        required=True,
        help="Input tip lineage assignment TSV/CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Reduced tip assignment TSV/CSV.",
    )
    parser.add_argument(
        "--mapping-output",
        required=True,
        help="Audit table showing every original-to-reduced lineage mapping.",
    )
    parser.add_argument(
        "--summary-output",
        required=True,
        help="Final reduced lineage counts.",
    )
    parser.add_argument(
        "--lineage-col",
        default="lineage",
        help="Lineage column. Default: lineage",
    )
    parser.add_argument(
        "--sequence-col",
        default="sequence_id",
        help="Sequence identifier column. Default: sequence_id",
    )
    parser.add_argument(
        "--min-lineage-size",
        type=int,
        default=10,
        help=(
            "Minimum final number of directly assigned tips. Smaller child "
            "lineages are iteratively merged into their parent. Default: 10"
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help=(
            "Maximum number of dots in a lineage. Example: 3 retains "
            "E1.2.12.3 but collapses E1.2.12.3.1 into it. Default: unlimited."
        ),
    )
    parser.add_argument(
        "--keep-original-node-columns",
        action="store_true",
        help=(
            "Keep each row's original assigned_node_* values instead of "
            "updating them to the retained lineage node."
        ),
    )

    args = parser.parse_args()

    assignment_path = Path(args.assignments)
    output_path = Path(args.output)
    mapping_path = Path(args.mapping_output)
    summary_path = Path(args.summary_output)

    assignments = read_table(assignment_path)

    required_columns = {args.sequence_col, args.lineage_col}
    missing = sorted(required_columns - set(assignments.columns))

    if missing:
        raise ValueError(
            f"Input assignments are missing required columns: {missing}. "
            f"Available columns: {list(assignments.columns)}"
        )

    if args.min_lineage_size < 1:
        raise ValueError("--min-lineage-size must be at least 1.")

    if args.max_depth is not None and args.max_depth < 0:
        raise ValueError("--max-depth must be zero or greater.")

    mapping = build_initial_mapping(
        assignments,
        lineage_col=args.lineage_col,
        max_depth=args.max_depth,
    )

    mapping = collapse_small_lineages(
        assignments,
        lineage_col=args.lineage_col,
        mapping=mapping,
        min_lineage_size=args.min_lineage_size,
    )

    reduced = apply_reduction(
        assignments,
        lineage_col=args.lineage_col,
        mapping=mapping,
        update_node_columns=not args.keep_original_node_columns,
    )

    mapping_df = build_mapping_table(
        assignments,
        lineage_col=args.lineage_col,
        mapping=mapping,
    )

    summary_df = build_summary(reduced, args.lineage_col)

    validate_output(
        assignments,
        reduced,
        sequence_col=args.sequence_col,
    )

    write_table(reduced, output_path)
    write_table(mapping_df, mapping_path)
    write_table(summary_df, summary_path)

    original_lineage_count = assignments[args.lineage_col].nunique(dropna=True)
    reduced_lineage_count = reduced[args.lineage_col].nunique(dropna=True)

    print(f"Input rows: {len(assignments)}")
    print(f"Original lineages: {original_lineage_count}")
    print(f"Reduced lineages: {reduced_lineage_count}")
    print(f"Wrote reduced assignments: {output_path}")
    print(f"Wrote lineage mapping: {mapping_path}")
    print(f"Wrote reduced summary: {summary_path}")


if __name__ == "__main__":
    main()
