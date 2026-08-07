#reduce_tip_lineage_assignments_mrca.py
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree


"""
Reduce a detailed tip-lineage assignment table using an actual phylogenetic
Newick tree.

For every currently represented lineage, the script:
    1. collects the sequence IDs assigned to that lineage;
    2. calculates their MRCA in the supplied tree;
    3. identifies the nearest other represented lineage whose MRCA is a strict
       ancestor of that MRCA;
    4. merges undersized lineages into that phylogenetic parent;
    5. recalculates counts and MRCAs after every iteration.

The input assignment table must contain at least:
    sequence_id, lineage

The tree terminal names must match sequence_id values, unless an assignment
contains sequences absent from the tree and --allow-missing-tree-tips is used.

Example:
    python reduce_tip_lineage_assignments_mrca.py \
        --assignments test_july31.tip_lineage_assignments.tsv \
        --tree full_tree.treefile \
        --output test_july31.mrca_reduced.tip_lineage_assignments.tsv \
        --mapping-output test_july31.mrca_lineage_reduction_map.tsv \
        --summary-output test_july31.mrca_reduced.lineage_summary.tsv \
        --mrca-output test_july31.mrca_nodes.tsv \
        --min-lineage-size 20
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
    """Original label-based parent helper retained for compatibility."""
    lineage = str(lineage).strip()

    if "." not in lineage:
        return None

    return lineage.rsplit(".", 1)[0]


def ancestor_lineages(lineage: str) -> List[str]:
    """Original label-based ancestor helper retained for compatibility."""
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
    Original label-based reduction retained for reference/compatibility.

    The MRCA workflow does not call this function. It calls
    collapse_small_lineages_by_mrca() instead.
    """
    mapping = dict(mapping)

    while True:
        counts = mapped_counts(assignments, lineage_col, mapping)
        targets = sorted(set(mapping.values()), key=natural_lineage_key)
        changed = False

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
                continue

            for original, current in list(mapping.items()):
                if current == target:
                    mapping[original] = parent
                    changed = True

        if not changed:
            break

    return mapping


def read_tree(path: Path, tree_format: str) -> Tree:
    tree = Phylo.read(str(path), tree_format)

    if tree.root is None:
        raise ValueError(f"Could not read a rooted tree from {path}")

    return tree


def build_tree_indexes(
    tree: Tree,
) -> Tuple[
    Dict[str, Clade],
    Dict[Clade, Optional[Clade]],
    Dict[Clade, int],
    Dict[Clade, int],
]:
    terminals: Dict[str, Clade] = {}

    for terminal in tree.get_terminals():
        if terminal.name is None or str(terminal.name).strip() == "":
            raise ValueError("The tree contains a terminal without a name.")

        name = str(terminal.name).strip()

        if name in terminals:
            raise ValueError(f"Duplicate terminal name in tree: {name}")

        terminals[name] = terminal

    parent_map: Dict[Clade, Optional[Clade]] = {tree.root: None}
    node_ids: Dict[Clade, int] = {}
    node_depths: Dict[Clade, int] = {tree.root: 0}

    for node_id, clade in enumerate(tree.find_clades(order="preorder"), start=1):
        node_ids[clade] = node_id

        for child in clade.clades:
            parent_map[child] = clade
            node_depths[child] = node_depths[clade] + 1

    return terminals, parent_map, node_ids, node_depths


def current_lineage_tips(
    assignments: pd.DataFrame,
    sequence_col: str,
    lineage_col: str,
    mapping: Dict[str, str],
    tree_terminals: Dict[str, Clade],
    allow_missing_tree_tips: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    working = assignments[[sequence_col, lineage_col]].copy()
    working[sequence_col] = working[sequence_col].astype(str).str.strip()
    working["_current_lineage"] = working[lineage_col].astype(str).map(mapping)

    lineage_tips: Dict[str, List[str]] = {}
    missing_by_lineage: Dict[str, List[str]] = {}

    for lineage, group in working.groupby("_current_lineage", sort=False):
        lineage = str(lineage)

        if lineage == "" or lineage.lower() == "unassigned":
            continue

        all_tips = list(dict.fromkeys(group[sequence_col].tolist()))
        present = [tip for tip in all_tips if tip in tree_terminals]
        missing = [tip for tip in all_tips if tip not in tree_terminals]

        if missing:
            missing_by_lineage[lineage] = missing

            if not allow_missing_tree_tips:
                preview = ", ".join(missing[:10])
                extra = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
                raise ValueError(
                    f"{len(missing)} assignment sequence(s) for lineage {lineage} "
                    f"are absent from the tree: {preview}{extra}"
                )

        if present:
            lineage_tips[lineage] = present

    return lineage_tips, missing_by_lineage


def calculate_lineage_mrcas(
    tree: Tree,
    lineage_tips: Dict[str, List[str]],
) -> Dict[str, Clade]:
    lineage_mrcas: Dict[str, Clade] = {}

    for lineage, tips in lineage_tips.items():
        if len(tips) == 1:
            mrca = tree.find_any(name=tips[0])
        else:
            mrca = tree.common_ancestor(tips)

        if mrca is None:
            raise RuntimeError(f"Could not calculate MRCA for lineage {lineage}")

        lineage_mrcas[lineage] = mrca

    return lineage_mrcas


def clade_is_ancestor(
    ancestor: Clade,
    descendant: Clade,
    parent_map: Dict[Clade, Optional[Clade]],
    strict: bool = True,
) -> bool:
    if ancestor is descendant:
        return not strict

    current = parent_map.get(descendant)

    while current is not None:
        if current is ancestor:
            return True
        current = parent_map.get(current)

    return False


def nearest_mrca_parent_lineage(
    lineage: str,
    lineage_mrcas: Dict[str, Clade],
    parent_map: Dict[Clade, Optional[Clade]],
    node_depths: Dict[Clade, int],
) -> Optional[str]:
    """
    Return the represented lineage with the nearest strict ancestral MRCA.

    If several labels share the same ancestral MRCA depth, prefer the lineage
    with the greatest label depth and then natural lineage order. The tree
    relationship, not the label prefix, determines eligibility.
    """
    target_mrca = lineage_mrcas.get(lineage)

    if target_mrca is None:
        return None

    candidates = []

    for other, other_mrca in lineage_mrcas.items():
        if other == lineage:
            continue

        if clade_is_ancestor(other_mrca, target_mrca, parent_map, strict=True):
            candidates.append(other)

    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda other: (
            node_depths[lineage_mrcas[other]],
            lineage_depth(other),
            natural_lineage_key(other),
        ),
        reverse=True,
    )[0]


def collapse_duplicate_mrcas(
    mapping: Dict[str, str],
    lineage_mrcas: Dict[str, Clade],
) -> Tuple[Dict[str, str], bool]:
    """
    Collapse multiple represented lineage labels that resolve to exactly the
    same MRCA node. The retained label is the shallowest label, then the first
    in natural lineage order.
    """
    mapping = dict(mapping)
    mrca_groups: Dict[Clade, List[str]] = {}

    for lineage, mrca in lineage_mrcas.items():
        mrca_groups.setdefault(mrca, []).append(lineage)

    changed = False

    for lineages in mrca_groups.values():
        if len(lineages) < 2:
            continue

        retained = sorted(
            lineages,
            key=lambda x: (lineage_depth(x), natural_lineage_key(x)),
        )[0]

        for duplicate in lineages:
            if duplicate == retained:
                continue

            for original, current in list(mapping.items()):
                if current == duplicate:
                    mapping[original] = retained
                    changed = True

    return mapping, changed


def collapse_small_lineages_by_mrca(
    assignments: pd.DataFrame,
    sequence_col: str,
    lineage_col: str,
    mapping: Dict[str, str],
    min_lineage_size: int,
    tree: Tree,
    tree_terminals: Dict[str, Clade],
    parent_map: Dict[Clade, Optional[Clade]],
    node_depths: Dict[Clade, int],
    allow_missing_tree_tips: bool,
) -> Dict[str, str]:
    """
    Iteratively merge small lineages into their nearest represented ancestral
    MRCA lineage. Counts and MRCAs are recalculated after each iteration.
    """
    mapping = dict(mapping)

    while True:
        counts = mapped_counts(assignments, lineage_col, mapping)
        lineage_tips, _ = current_lineage_tips(
            assignments,
            sequence_col,
            lineage_col,
            mapping,
            tree_terminals,
            allow_missing_tree_tips,
        )
        lineage_mrcas = calculate_lineage_mrcas(tree, lineage_tips)

        mapping, duplicate_changed = collapse_duplicate_mrcas(
            mapping,
            lineage_mrcas,
        )

        if duplicate_changed:
            continue

        targets = sorted(
            lineage_mrcas,
            key=lambda lineage: (
                node_depths[lineage_mrcas[lineage]],
                lineage_depth(lineage),
                natural_lineage_key(lineage),
            ),
            reverse=True,
        )

        changed = False

        for target in targets:
            count = int(counts.get(target, 0))

            if count >= min_lineage_size:
                continue

            parent = nearest_mrca_parent_lineage(
                target,
                lineage_mrcas,
                parent_map,
                node_depths,
            )

            if parent is None:
                # No represented lineage has an MRCA above this lineage.
                # Keep it rather than inventing a new lineage label.
                continue

            for original, current in list(mapping.items()):
                if current == target:
                    mapping[original] = parent
                    changed = True

            # Recalculate immediately after one merge. This avoids using stale
            # MRCAs and counts for the remaining targets in this iteration.
            if changed:
                break

        if not changed:
            break

    return mapping


def infer_lineage_node_details(
    assignments: pd.DataFrame,
    lineage_col: str,
) -> Dict[str, Dict[str, object]]:
    """Original non-tree node-detail inference retained for compatibility."""
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
    """Original fallback retained for compatibility."""
    if reduced_lineage in original_details:
        return reduced_lineage

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


def build_mrca_node_details(
    assignments: pd.DataFrame,
    sequence_col: str,
    lineage_col: str,
    mapping: Dict[str, str],
    tree: Tree,
    tree_terminals: Dict[str, Clade],
    node_ids: Dict[Clade, int],
    node_depths: Dict[Clade, int],
    allow_missing_tree_tips: bool,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, List[str]]]:
    lineage_tips, missing_by_lineage = current_lineage_tips(
        assignments,
        sequence_col,
        lineage_col,
        mapping,
        tree_terminals,
        allow_missing_tree_tips,
    )
    lineage_mrcas = calculate_lineage_mrcas(tree, lineage_tips)
    details: Dict[str, Dict[str, object]] = {}

    for lineage, mrca in lineage_mrcas.items():
        details[lineage] = {
            "assigned_node_id": node_ids[mrca],
            "assigned_node_name": mrca.name or f"NODE_{node_ids[mrca]:05d}",
            "assigned_node_tip_count": len(mrca.get_terminals()),
            "mrca_tree_depth": node_depths[mrca],
            "mrca_assignment_tip_count": len(lineage_tips[lineage]),
            "mrca_is_terminal": mrca.is_terminal(),
        }

    return details, missing_by_lineage


def apply_reduction(
    assignments: pd.DataFrame,
    lineage_col: str,
    mapping: Dict[str, str],
    update_node_columns: bool,
    mrca_details: Optional[Dict[str, Dict[str, object]]] = None,
) -> pd.DataFrame:
    output = assignments.copy()
    output["original_lineage"] = output[lineage_col].astype(str)
    output[lineage_col] = output["original_lineage"].map(mapping)

    if not update_node_columns:
        return output

    if mrca_details is not None:
        output["node_details_source_lineage"] = output[lineage_col]

        for column in [
            "assigned_node_id",
            "assigned_node_name",
            "assigned_node_tip_count",
        ]:
            value_map = {
                lineage: details.get(column, "")
                for lineage, details in mrca_details.items()
            }
            output[column] = output[lineage_col].map(value_map).fillna("")

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
    mrca_details: Optional[Dict[str, Dict[str, object]]] = None,
) -> pd.DataFrame:
    original_counts = assignments[lineage_col].astype(str).value_counts()
    reduced_series = assignments[lineage_col].astype(str).map(mapping)
    reduced_counts = reduced_series.value_counts()

    rows = []

    for original in sorted(mapping, key=natural_lineage_key):
        reduced = mapping[original]
        row = {
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

        if mrca_details is not None:
            detail = mrca_details.get(reduced, {})
            row.update(
                {
                    "reduced_mrca_node_id": detail.get("assigned_node_id", ""),
                    "reduced_mrca_node_name": detail.get("assigned_node_name", ""),
                    "reduced_mrca_tree_tip_count": detail.get(
                        "assigned_node_tip_count", ""
                    ),
                    "reduced_mrca_tree_depth": detail.get("mrca_tree_depth", ""),
                }
            )

        rows.append(row)

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


def build_mrca_table(
    mrca_details: Dict[str, Dict[str, object]],
    missing_by_lineage: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []

    for lineage in sorted(mrca_details, key=natural_lineage_key):
        detail = mrca_details[lineage]
        missing = missing_by_lineage.get(lineage, [])
        rows.append(
            {
                "lineage": lineage,
                "mrca_node_id": detail.get("assigned_node_id", ""),
                "mrca_node_name": detail.get("assigned_node_name", ""),
                "mrca_tree_tip_count": detail.get("assigned_node_tip_count", ""),
                "assignment_tips_used_for_mrca": detail.get(
                    "mrca_assignment_tip_count", ""
                ),
                "mrca_tree_depth": detail.get("mrca_tree_depth", ""),
                "mrca_is_terminal": detail.get("mrca_is_terminal", ""),
                "missing_assignment_tip_count": len(missing),
                "missing_assignment_tips": ",".join(missing),
            }
        )

    return pd.DataFrame(rows)


def validate_output(
    original: pd.DataFrame,
    reduced: pd.DataFrame,
    sequence_col: str,
    lineage_col: str,
) -> None:
    if len(original) != len(reduced):
        raise RuntimeError(
            f"Row count changed: input={len(original)}, output={len(reduced)}"
        )

    if sequence_col in original.columns:
        if (
            original[sequence_col].astype(str).tolist()
            != reduced[sequence_col].astype(str).tolist()
        ):
            raise RuntimeError(
                "Sequence order or sequence identifiers changed during reduction."
            )

    if reduced[lineage_col].isna().any():
        raise RuntimeError("Reduced output contains missing lineage assignments.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce tip lineage assignments by calculating lineage MRCAs in "
            "a phylogenetic tree and merging small lineages into the nearest "
            "represented ancestral-MRCA lineage."
        )
    )

    parser.add_argument(
        "--assignments",
        required=True,
        help="Input tip lineage assignment TSV/CSV.",
    )
    parser.add_argument(
        "--tree",
        required=True,
        help="Rooted Newick phylogenetic tree containing the sequence IDs.",
    )
    parser.add_argument(
        "--tree-format",
        default="newick",
        help="Biopython tree format. Default: newick",
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
        "--mrca-output",
        required=True,
        help="Final lineage-to-MRCA node table.",
    )
    parser.add_argument(
        "--lineage-col",
        default="lineage",
        help="Lineage column. Default: lineage",
    )
    parser.add_argument(
        "--sequence-col",
        default="sequence_id",
        help="Sequence identifier column matching tree tips. Default: sequence_id",
    )
    parser.add_argument(
        "--min-lineage-size",
        type=int,
        default=10,
        help=(
            "Minimum final assignment count. Smaller lineages are iteratively "
            "merged into the nearest represented ancestral-MRCA lineage. "
            "Default: 10"
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help=(
            "Optional preliminary lineage-label truncation before MRCA "
            "reduction. Default: unlimited. Usually omit this for a purely "
            "tree-driven reduction."
        ),
    )
    parser.add_argument(
        "--keep-original-node-columns",
        action="store_true",
        help=(
            "Keep each row's original assigned_node_* values instead of "
            "replacing them with the final lineage MRCA node details."
        ),
    )
    parser.add_argument(
        "--allow-missing-tree-tips",
        action="store_true",
        help=(
            "Allow assignment sequence IDs absent from the tree. MRCA is then "
            "calculated only from matching tips, and missing tips are reported."
        ),
    )

    args = parser.parse_args()

    assignment_path = Path(args.assignments)
    tree_path = Path(args.tree)
    output_path = Path(args.output)
    mapping_path = Path(args.mapping_output)
    summary_path = Path(args.summary_output)
    mrca_path = Path(args.mrca_output)

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

    tree = read_tree(tree_path, args.tree_format)
    tree_terminals, parent_map, node_ids, node_depths = build_tree_indexes(tree)

    mapping = build_initial_mapping(
        assignments,
        lineage_col=args.lineage_col,
        max_depth=args.max_depth,
    )

    mapping = collapse_small_lineages_by_mrca(
        assignments,
        sequence_col=args.sequence_col,
        lineage_col=args.lineage_col,
        mapping=mapping,
        min_lineage_size=args.min_lineage_size,
        tree=tree,
        tree_terminals=tree_terminals,
        parent_map=parent_map,
        node_depths=node_depths,
        allow_missing_tree_tips=args.allow_missing_tree_tips,
    )

    mrca_details, missing_by_lineage = build_mrca_node_details(
        assignments,
        sequence_col=args.sequence_col,
        lineage_col=args.lineage_col,
        mapping=mapping,
        tree=tree,
        tree_terminals=tree_terminals,
        node_ids=node_ids,
        node_depths=node_depths,
        allow_missing_tree_tips=args.allow_missing_tree_tips,
    )

    reduced = apply_reduction(
        assignments,
        lineage_col=args.lineage_col,
        mapping=mapping,
        update_node_columns=not args.keep_original_node_columns,
        mrca_details=mrca_details,
    )

    mapping_df = build_mapping_table(
        assignments,
        lineage_col=args.lineage_col,
        mapping=mapping,
        mrca_details=mrca_details,
    )

    summary_df = build_summary(reduced, args.lineage_col)
    mrca_df = build_mrca_table(mrca_details, missing_by_lineage)

    validate_output(
        assignments,
        reduced,
        sequence_col=args.sequence_col,
        lineage_col=args.lineage_col,
    )

    write_table(reduced, output_path)
    write_table(mapping_df, mapping_path)
    write_table(summary_df, summary_path)
    write_table(mrca_df, mrca_path)

    original_lineage_count = assignments[args.lineage_col].nunique(dropna=True)
    reduced_lineage_count = reduced[args.lineage_col].nunique(dropna=True)

    print(f"Input rows: {len(assignments)}")
    print(f"Tree tips: {len(tree_terminals)}")
    print(f"Original lineages: {original_lineage_count}")
    print(f"Reduced lineages: {reduced_lineage_count}")
    print(f"Wrote reduced assignments: {output_path}")
    print(f"Wrote lineage mapping: {mapping_path}")
    print(f"Wrote reduced summary: {summary_path}")
    print(f"Wrote final MRCA nodes: {mrca_path}")


if __name__ == "__main__":
    main()

