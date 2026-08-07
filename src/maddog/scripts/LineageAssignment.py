#LineageAssignment.py
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from Bio import Phylo

'''
python src/maddog/scripts/LineageAssignment.py --tree tmp/tree/Indian-Sub/Indian-Sub.iqtree.treefile --node_assignment tmp/maddog_ref_test --out_prefix maddog_style_scripted
'''

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_newick_label(value: str) -> str:
    """
    Make labels safer for Newick output.
    """
    value = str(value)

    replacements = {
        " ": "_",
        "\t": "_",
        "\n": "_",
        "\r": "_",
        ":": "_",
        ";": "_",
        ",": "_",
        "(": "_",
        ")": "_",
        "[": "_",
        "]": "_",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return value


def parse_tip_labels(value) -> List[str]:
    """
    Parse comma-separated descendant_tip_labels.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "" or value.upper() == "NA":
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def lineage_depth(lineage: str) -> int:
    """
    A1       depth 0
    A1.1     depth 1
    A1.1.1   depth 2
    Indian-Sub_A1.1 depth 1
    """
    if pd.isna(lineage):
        return 0

    lineage = str(lineage).strip()

    if lineage == "":
        return 0

    return lineage.count(".")


# ------------------------------------------------------------
# Read and clean NodeAssigned output
# ------------------------------------------------------------

def detect_delimiter(path: Path) -> str:
    first_line = path.read_text(errors="replace").splitlines()[0]

    if "\t" in first_line:
        return "\t"

    return ","


def read_node_assignment_table(path: Path) -> pd.DataFrame:
    """
    Reads NodeAssigned output.

    Also fixes a common formatting issue where the first two fields
    are accidentally joined, for example:

        526 NODE_0000231<TAB>NODE_0000231<TAB>...

    instead of:

        526<TAB>NODE_0000231<TAB>NODE_0000231<TAB>...
    """
    path = Path(path)
    delimiter = detect_delimiter(path)

    rows = []

    with path.open("r", newline="", errors="replace") as handle:
        reader = csv.reader(handle, delimiter=delimiter)

        header = next(reader)

        for line_number, parts in enumerate(reader, start=2):
            if len(parts) == 0:
                continue

            # Repair case where first column contains "526 NODE_0000231"
            if len(parts) == len(header) - 1 and len(parts[0].split()) == 2:
                first, second = parts[0].split(None, 1)
                parts = [first, second] + parts[1:]

            if len(parts) != len(header):
                raise ValueError(
                    f"Could not parse line {line_number}. "
                    f"Expected {len(header)} columns but found {len(parts)}.\n"
                    f"Problem line starts with: {parts[:5]}"
                )

            rows.append(parts)

    df = pd.DataFrame(rows, columns=header)

    return df


def clean_node_assignment_table(
    df: pd.DataFrame,
    lineage_col: str,
    node_id_col: str,
) -> pd.DataFrame:
    df = df.copy()

    if node_id_col not in df.columns:
        raise ValueError(f"Missing required column: {node_id_col}")

    if lineage_col not in df.columns:
        raise ValueError(f"Missing required lineage column: {lineage_col}")

    if "descendant_tip_labels" not in df.columns:
        raise ValueError("Missing required column: descendant_tip_labels")

    df[node_id_col] = df[node_id_col].astype(str).str.strip()
    df[lineage_col] = df[lineage_col].astype(str).str.strip()

    df = df[
        (df[node_id_col] != "")
        & (df[lineage_col] != "")
        & (df[lineage_col].str.upper() != "NA")
    ].copy()

    if "descendant_tip_count" in df.columns:
        df["descendant_tip_count_numeric"] = pd.to_numeric(
            df["descendant_tip_count"],
            errors="coerce"
        ).fillna(0).astype(int)
    elif "n_tips" in df.columns:
        df["descendant_tip_count_numeric"] = pd.to_numeric(
            df["n_tips"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        df["descendant_tip_count_numeric"] = df["descendant_tip_labels"].apply(
            lambda x: len(parse_tip_labels(x))
        )

    df["lineage_depth"] = df[lineage_col].apply(lineage_depth)

    return df


# ------------------------------------------------------------
# Tree internal node mapping
# ------------------------------------------------------------

def make_internal_node_id_map(
    tree,
    ancestral_prefix: str = "NODE_",
    ancestral_digits: int = 7,
    ancestral_index_base: int = 0,
    internal_order: str = "preorder",
) -> Dict[object, str]:
    """
    Maps internal clades to NODE_* IDs using the same style:

        first internal node = NODE_0000000
        second internal node = NODE_0000001
        etc.

    Default preorder is important for NodeInfo / MADDOG-style mapping.
    """
    internal_nodes = tree.get_nonterminals(order=internal_order)

    clade_to_node_id = {}

    for index, clade in enumerate(internal_nodes, start=ancestral_index_base):
        node_id = f"{ancestral_prefix}{index:0{ancestral_digits}d}"
        clade_to_node_id[clade] = node_id

    return clade_to_node_id


# ------------------------------------------------------------
# Assign lineage to internal nodes
# ------------------------------------------------------------

def label_internal_nodes(
    tree,
    node_assignment_df: pd.DataFrame,
    lineage_col: str,
    node_id_col: str,
    label_mode: str,
    ancestral_prefix: str,
    ancestral_digits: int,
    ancestral_index_base: int,
    internal_order: str,
):
    """
    label_mode:
        lineage         -> internal node label becomes A1
        lineage_nodeid  -> internal node label becomes A1|NODE_0000183
    """
    labelled_tree = copy.deepcopy(tree)

    node_to_lineage = dict(
        zip(
            node_assignment_df[node_id_col],
            node_assignment_df[lineage_col],
        )
    )

    clade_to_node_id = make_internal_node_id_map(
        labelled_tree,
        ancestral_prefix=ancestral_prefix,
        ancestral_digits=ancestral_digits,
        ancestral_index_base=ancestral_index_base,
        internal_order=internal_order,
    )

    for clade, node_id in clade_to_node_id.items():
        lineage = node_to_lineage.get(node_id)

        if lineage is None:
            continue

        lineage = safe_newick_label(lineage)
        node_id_safe = safe_newick_label(node_id)

        if label_mode == "lineage":
            clade.name = lineage
        elif label_mode == "lineage_nodeid":
            clade.name = f"{lineage}|{node_id_safe}"
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

    return labelled_tree


# ------------------------------------------------------------
# Assign lineage to terminal tips
# ------------------------------------------------------------

def build_tip_lineage_assignments(
    tree,
    node_assignment_df: pd.DataFrame,
    lineage_col: str,
    node_id_col: str,
) -> pd.DataFrame:
    """
    Assigns every tip to the deepest retained lineage.

    Parent nodes are applied first.
    Child nodes are applied later and overwrite the parent assignment.

    Example:
        A1 has seq1,seq2,seq3,seq4
        A1.1 has seq3,seq4

    Final:
        seq1 -> A1
        seq2 -> A1
        seq3 -> A1.1
        seq4 -> A1.1
    """
    df = node_assignment_df.copy()

    # Parent first, child later.
    # Higher-level lineage depth first.
    # Larger clades first within same depth.
    df = df.sort_values(
        by=["lineage_depth", "descendant_tip_count_numeric"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    tip_to_assignment = {}

    for _, row in df.iterrows():
        lineage = row[lineage_col]
        node_id = row[node_id_col]

        tips = parse_tip_labels(row["descendant_tip_labels"])

        for tip in tips:
            tip_to_assignment[tip] = {
                "sequence_id": tip,
                "lineage": lineage,
                "assigned_node_id": node_id,
                "assigned_node_name": node_id,
                "assigned_node_tip_count": row["descendant_tip_count_numeric"],
            }

    all_tree_tips = [tip.name for tip in tree.get_terminals()]

    output_rows = []

    for tip_name in all_tree_tips:
        if tip_name in tip_to_assignment:
            output_rows.append(tip_to_assignment[tip_name])
        else:
            output_rows.append(
                {
                    "sequence_id": tip_name,
                    "lineage": "Unassigned",
                    "assigned_node_id": "",
                    "assigned_node_name": "",
                    "assigned_node_tip_count": 0,
                }
            )

    return pd.DataFrame(output_rows)


def label_terminal_tips(
    tree,
    tip_assignment_df: pd.DataFrame,
    separator: str = "|",
):
    """
    Adds lineage to terminal tip labels.

    Example:
        AF369309 -> AF369309|A1
    """
    labelled_tree = copy.deepcopy(tree)

    tip_to_lineage = dict(
        zip(
            tip_assignment_df["sequence_id"],
            tip_assignment_df["lineage"],
        )
    )

    for tip in labelled_tree.get_terminals():
        original_name = tip.name
        lineage = tip_to_lineage.get(original_name, "Unassigned")

        tip.name = (
            f"{safe_newick_label(original_name)}"
            f"{separator}"
            f"{safe_newick_label(lineage)}"
        )

    return labelled_tree


# ------------------------------------------------------------
# Output summaries
# ------------------------------------------------------------

def write_lineage_node_map(
    node_assignment_df: pd.DataFrame,
    output_path: Path,
    lineage_col: str,
    node_id_col: str,
):
    cols = [
        c for c in [
            "Node",
            node_id_col,
            "parent_node_id",
            "descendant_tip_count",
            "n_tips",
            "overlaps",
            "direct_parent_overlap_node",
            "parent_child_tip_difference",
            lineage_col,
        ]
        if c in node_assignment_df.columns
    ]

    node_assignment_df[cols].to_csv(
        output_path,
        sep="\t",
        index=False,
    )


def write_summary(tip_assignment_df: pd.DataFrame):
    summary = (
        tip_assignment_df
        .groupby("lineage", dropna=False)
        .size()
        .reset_index(name="tip_count")
        .sort_values(
            by=["lineage"],
            kind="mergesort",
        )
    )

    print("\nFinal tip counts:")
    print(summary.to_string(index=False))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a Newick tree with lineage labels from a NodeAssigned table "
            "and create final tip lineage assignments."
        )
    )

    parser.add_argument(
        "--tree",
        "--contree",
        dest="tree",
        required=True,
        help="Input Newick tree, for example .contree or .treefile",
    )

    parser.add_argument(
        "--node_assignment",
        required=True,
        help="NodeAssigned output TSV/CSV containing node_id, descendant_tip_labels and lineage",
    )

    parser.add_argument(
        "--out_prefix",
        required=True,
        help=(
            "Output filename prefix. Relative prefixes are written beneath "
            "--base_dir/--output_dir."
        ),
    )

    parser.add_argument(
        "--base_dir",
        "--base-dir",
        type=Path,
        default=Path("tmp"),
        help="Base output directory. Default: tmp",
    )

    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=Path("lineage_assignment"),
        help="Output directory beneath --base_dir. Default: lineage_assignment",
    )

    parser.add_argument(
        "--lineage_col",
        default="lineage",
        help="Column containing final lineage name. Default: lineage",
    )

    parser.add_argument(
        "--node_id_col",
        default="node_id",
        help="Column containing NODE_* IDs. Default: node_id",
    )

    parser.add_argument(
        "--ancestral_prefix",
        default="NODE_",
        help="Internal ancestral node prefix. Default: NODE_",
    )

    parser.add_argument(
        "--ancestral_digits",
        type=int,
        default=7,
        help="Zero-padding digits for NODE IDs. Default: 7",
    )

    parser.add_argument(
        "--ancestral_index_base",
        type=int,
        default=0,
        help="First internal node index. Default: 0 gives NODE_0000000",
    )

    parser.add_argument(
        "--internal_order",
        choices=["preorder", "postorder", "level"],
        default="preorder",
        help="Internal node traversal order. Default: preorder",
    )

    args = parser.parse_args()

    tree_path = Path(args.tree)
    node_assignment_path = Path(args.node_assignment)
    requested_prefix = Path(args.out_prefix)
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else args.base_dir / args.output_dir
    )
    out_prefix = (
        requested_prefix
        if requested_prefix.is_absolute()
        else output_dir / requested_prefix
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Read input tree and canonicalise child-clade order. This must match
    # AssignNodeDetails so NODE_* IDs and terminal output order are stable.
    tree = Phylo.read(tree_path, "newick")
    tree.ladderize(reverse=False)

    # Read and clean NodeAssigned table
    raw_assignment_df = read_node_assignment_table(node_assignment_path)

    node_assignment_df = clean_node_assignment_table(
        raw_assignment_df,
        lineage_col=args.lineage_col,
        node_id_col=args.node_id_col,
    )

    # Output paths
    cleaned_table_path = out_prefix.with_suffix(".node_assignment.cleaned.tsv")
    lineage_node_map_path = out_prefix.with_suffix(".lineage_node_map.tsv")
    tip_assignment_path = out_prefix.with_suffix(".tip_lineage_assignments.tsv")

    lineage_internal_tree_path = out_prefix.with_suffix(".lineage_internal_labels.nwk")
    lineage_nodeid_tree_path = out_prefix.with_suffix(".lineage_nodeid_internal_labels.nwk")
    tip_and_internal_tree_path = out_prefix.with_suffix(".tip_and_internal_lineage_labels.nwk")

    # Save cleaned NodeAssigned table
    node_assignment_df.to_csv(
        cleaned_table_path,
        sep="\t",
        index=False,
    )

    # Save node lineage map
    write_lineage_node_map(
        node_assignment_df,
        lineage_node_map_path,
        lineage_col=args.lineage_col,
        node_id_col=args.node_id_col,
    )

    # Internal-node lineage labels only
    lineage_tree = label_internal_nodes(
        tree,
        node_assignment_df,
        lineage_col=args.lineage_col,
        node_id_col=args.node_id_col,
        label_mode="lineage",
        ancestral_prefix=args.ancestral_prefix,
        ancestral_digits=args.ancestral_digits,
        ancestral_index_base=args.ancestral_index_base,
        internal_order=args.internal_order,
    )

    Phylo.write(
        lineage_tree,
        lineage_internal_tree_path,
        "newick",
    )

    # Internal-node lineage + NODE ID labels
    lineage_nodeid_tree = label_internal_nodes(
        tree,
        node_assignment_df,
        lineage_col=args.lineage_col,
        node_id_col=args.node_id_col,
        label_mode="lineage_nodeid",
        ancestral_prefix=args.ancestral_prefix,
        ancestral_digits=args.ancestral_digits,
        ancestral_index_base=args.ancestral_index_base,
        internal_order=args.internal_order,
    )

    Phylo.write(
        lineage_nodeid_tree,
        lineage_nodeid_tree_path,
        "newick",
    )

    # Tip assignment table
    tip_assignment_df = build_tip_lineage_assignments(
        tree,
        node_assignment_df,
        lineage_col=args.lineage_col,
        node_id_col=args.node_id_col,
    )

    tip_assignment_df.to_csv(
        tip_assignment_path,
        sep="\t",
        index=False,
    )

    # Tree with terminal tips labelled and internal nodes labelled
    tip_labelled_tree = label_terminal_tips(
        lineage_tree,
        tip_assignment_df,
        separator="|",
    )

    Phylo.write(
        tip_labelled_tree,
        tip_and_internal_tree_path,
        "newick",
    )

    print(f"Wrote cleaned NodeAssigned table: {cleaned_table_path}")
    print(f"Wrote lineage node map: {lineage_node_map_path}")
    print(f"Wrote tip lineage assignments: {tip_assignment_path}")
    print(f"Wrote internal lineage tree: {lineage_internal_tree_path}")
    print(f"Wrote internal lineage + NODE ID tree: {lineage_nodeid_tree_path}")
    print(f"Wrote tip + internal lineage tree: {tip_and_internal_tree_path}")

    write_summary(tip_assignment_df)


if __name__ == "__main__":
    main()
