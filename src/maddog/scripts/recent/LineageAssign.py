#!/usr/bin/env python3

import copy
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from Bio import Phylo


class MergeResults:
    def __init__(
        self,
        tree,
        node_designations,
        sequence_designations,
        metadata,
        metadata_id_column,
        final_lineage_column,
        internal_order,
        ancestral_prefix,
        ancestral_digits,
        ancestral_index_base,
        unassigned_label
    ):
        self.tree = Path(tree)
        self.node_designations = Path(node_designations)
        self.sequence_designations = Path(sequence_designations)
        self.metadata = Path(metadata) if metadata is not None else None
        self.metadata_id_column = metadata_id_column
        self.final_lineage_column = final_lineage_column
        self.internal_order = internal_order
        self.ancestral_prefix = ancestral_prefix
        self.ancestral_digits = ancestral_digits
        self.ancestral_index_base = ancestral_index_base
        self.unassigned_label = unassigned_label

    def detect_separator(self, path):
        first_line = Path(path).read_text(errors="replace").splitlines()[0]

        if "\t" in first_line:
            return "\t"

        return ","

    def read_table(self, path):
        return pd.read_csv(
            path,
            sep=self.detect_separator(path),
            dtype=str,
            keep_default_na=False
        )

    def safe_newick_label(self, value):
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
            "]": "_"
        }

        for old, new in replacements.items():
            value = value.replace(old, new)

        return value

    def make_internal_node_map(self, tree):
        internal_nodes = tree.get_nonterminals(
            order=self.internal_order
        )

        clade_to_node_id = {}

        for index, clade in enumerate(
            internal_nodes,
            start=self.ancestral_index_base
        ):
            node_id = (
                self.ancestral_prefix
                + f"{index:0{self.ancestral_digits}d}"
            )

            clade_to_node_id[clade] = node_id

        return clade_to_node_id

    def validate_inputs(self, nodes, sequences, tree):
        required_node_columns = {
            "node_id",
            self.final_lineage_column
        }

        missing_node_columns = required_node_columns.difference(
            nodes.columns
        )

        if missing_node_columns:
            raise ValueError(
                "Node designation table is missing: "
                + ", ".join(sorted(missing_node_columns))
            )

        required_sequence_columns = {
            "ID",
            "cluster",
            "assigned_node_id"
        }

        missing_sequence_columns = required_sequence_columns.difference(
            sequences.columns
        )

        if missing_sequence_columns:
            raise ValueError(
                "Sequence designation table is missing: "
                + ", ".join(sorted(missing_sequence_columns))
            )

        if nodes["node_id"].duplicated().any():
            raise ValueError(
                "Duplicate node_id values in node designation table."
            )

        if sequences["ID"].duplicated().any():
            raise ValueError(
                "Duplicate sequence IDs in sequence designation table."
            )

        node_map = self.make_internal_node_map(tree)
        valid_node_ids = set(node_map.values())

        unknown_nodes = sorted(
            set(nodes["node_id"]) - valid_node_ids
        )

        if unknown_nodes:
            raise ValueError(
                "Node IDs do not map to the supplied tree. Examples: "
                + ", ".join(unknown_nodes[:10])
            )

    def merge_metadata(self, sequences):
        if self.metadata is None:
            return sequences.copy()

        metadata = self.read_table(self.metadata)

        if self.metadata_id_column not in metadata.columns:
            raise ValueError(
                "Metadata is missing ID column: "
                + self.metadata_id_column
            )

        metadata[self.metadata_id_column] = (
            metadata[self.metadata_id_column].astype(str).str.strip()
        )

        if metadata[self.metadata_id_column].duplicated().any():
            raise ValueError(
                "Metadata contains duplicate sequence IDs."
            )

        metadata = metadata.rename(
            columns={
                self.metadata_id_column: "ID"
            }
        )

        duplicate_columns = [
            column
            for column in metadata.columns
            if column != "ID" and column in sequences.columns
        ]

        metadata = metadata.drop(
            columns=duplicate_columns,
            errors="ignore"
        )

        return sequences.merge(
            metadata,
            on="ID",
            how="left",
            validate="one_to_one"
        )

    def build_node_summary(self, nodes, sequences):
        assigned = sequences[
            sequences["cluster"] != self.unassigned_label
        ].copy()

        counts = (
            assigned
            .groupby("cluster", dropna=False)
            .size()
            .reset_index(name="assigned_sequence_count")
        )

        summary = nodes.merge(
            counts,
            left_on=self.final_lineage_column,
            right_on="cluster",
            how="left"
        )

        summary["assigned_sequence_count"] = pd.to_numeric(
            summary["assigned_sequence_count"],
            errors="coerce"
        ).fillna(0).astype(int)

        summary = summary.drop(
            columns=["cluster"],
            errors="ignore"
        )

        return summary

    def build_lineage_summary(self, sequences):
        summary = (
            sequences
            .groupby("cluster", dropna=False)
            .size()
            .reset_index(name="sequence_count")
            .sort_values(
                by=["cluster"],
                kind="mergesort"
            )
            .reset_index(drop=True)
        )

        return summary

    def label_internal_nodes(self, tree, nodes, include_node_id=False):
        labelled_tree = copy.deepcopy(tree)

        node_to_lineage = dict(
            zip(
                nodes["node_id"],
                nodes[self.final_lineage_column]
            )
        )

        clade_to_node_id = self.make_internal_node_map(
            labelled_tree
        )

        for clade, node_id in clade_to_node_id.items():
            lineage = node_to_lineage.get(node_id)

            if lineage is None or str(lineage).strip() == "":
                continue

            lineage = self.safe_newick_label(lineage)

            if include_node_id:
                clade.name = (
                    lineage
                    + "|"
                    + self.safe_newick_label(node_id)
                )
            else:
                clade.name = lineage

        return labelled_tree

    def label_terminal_tips(self, tree, sequences):
        labelled_tree = copy.deepcopy(tree)

        tip_to_lineage = dict(
            zip(
                sequences["ID"],
                sequences["cluster"]
            )
        )

        for tip in labelled_tree.get_terminals():
            original_name = str(tip.name)
            lineage = tip_to_lineage.get(
                original_name,
                self.unassigned_label
            )

            tip.name = (
                self.safe_newick_label(original_name)
                + "|"
                + self.safe_newick_label(lineage)
            )

        return labelled_tree

    def build_validation(self, tree, nodes, sequences):
        rows = []

        tree_tips = {
            str(tip.name)
            for tip in tree.get_terminals()
            if tip.name is not None
        }

        sequence_ids = set(
            sequences["ID"].astype(str)
        )

        for sequence_id in sorted(tree_tips - sequence_ids):
            rows.append(
                {
                    "issue": "tree_tip_missing_sequence_designation",
                    "value": sequence_id
                }
            )

        for sequence_id in sorted(sequence_ids - tree_tips):
            rows.append(
                {
                    "issue": "sequence_designation_not_in_tree",
                    "value": sequence_id
                }
            )

        assigned_nodes = set(
            sequences.loc[
                sequences["assigned_node_id"] != "",
                "assigned_node_id"
            ].astype(str)
        )

        known_nodes = set(
            nodes["node_id"].astype(str)
        )

        for node_id in sorted(assigned_nodes - known_nodes):
            rows.append(
                {
                    "issue": "sequence_references_unknown_node",
                    "value": node_id
                }
            )

        duplicate_lineages = nodes[
            nodes[self.final_lineage_column].duplicated(keep=False)
            & (nodes[self.final_lineage_column] != "")
        ]

        for lineage in sorted(
            duplicate_lineages[self.final_lineage_column]
            .astype(str)
            .unique()
        ):
            rows.append(
                {
                    "issue": "duplicate_final_lineage",
                    "value": lineage
                }
            )

        return pd.DataFrame(
            rows,
            columns=["issue", "value"]
        )

    def run(self, out_prefix):
        tree = Phylo.read(self.tree, "newick")
        nodes = self.read_table(self.node_designations)
        sequences = self.read_table(self.sequence_designations)

        self.validate_inputs(
            nodes,
            sequences,
            tree
        )

        final_sequences = self.merge_metadata(
            sequences
        )

        node_summary = self.build_node_summary(
            nodes,
            final_sequences
        )

        lineage_summary = self.build_lineage_summary(
            final_sequences
        )

        validation = self.build_validation(
            tree,
            nodes,
            final_sequences
        )

        internal_tree = self.label_internal_nodes(
            tree,
            nodes,
            include_node_id=False
        )

        internal_nodeid_tree = self.label_internal_nodes(
            tree,
            nodes,
            include_node_id=True
        )

        final_tree = self.label_terminal_tips(
            internal_tree,
            final_sequences
        )

        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        final_sequence_path = out_prefix.with_suffix(
            ".sequence_data_final.tsv"
        )

        node_summary_path = out_prefix.with_suffix(
            ".lineage_node_summary.tsv"
        )

        lineage_summary_path = out_prefix.with_suffix(
            ".lineage_summary.tsv"
        )

        validation_path = out_prefix.with_suffix(
            ".validation.tsv"
        )

        internal_tree_path = out_prefix.with_suffix(
            ".internal_lineages.nwk"
        )

        internal_nodeid_tree_path = out_prefix.with_suffix(
            ".internal_lineages_nodeids.nwk"
        )

        final_tree_path = out_prefix.with_suffix(
            ".tip_and_internal_lineages.nwk"
        )

        final_sequences.to_csv(
            final_sequence_path,
            sep="\t",
            index=False,
            na_rep="NA"
        )

        node_summary.to_csv(
            node_summary_path,
            sep="\t",
            index=False,
            na_rep="NA"
        )

        lineage_summary.to_csv(
            lineage_summary_path,
            sep="\t",
            index=False,
            na_rep="NA"
        )

        validation.to_csv(
            validation_path,
            sep="\t",
            index=False,
            na_rep="NA"
        )

        Phylo.write(
            internal_tree,
            internal_tree_path,
            "newick"
        )

        Phylo.write(
            internal_nodeid_tree,
            internal_nodeid_tree_path,
            "newick"
        )

        Phylo.write(
            final_tree,
            final_tree_path,
            "newick"
        )

        print(f"Wrote final sequence data: {final_sequence_path}")
        print(f"Wrote lineage node summary: {node_summary_path}")
        print(f"Wrote lineage summary: {lineage_summary_path}")
        print(f"Wrote validation report: {validation_path}")
        print(f"Wrote internal lineage tree: {internal_tree_path}")
        print(
            "Wrote internal lineage and node-ID tree: "
            f"{internal_nodeid_tree_path}"
        )
        print(f"Wrote final labelled tree: {final_tree_path}")

        return (
            final_sequences,
            node_summary,
            lineage_summary,
            validation
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Merge SeqDesignation outputs with metadata and "
            "create final MADDOG-style tables and labelled trees."
        )
    )

    parser.add_argument(
        "--tree",
        required=True,
        help="Exact rooted tree used by NodeAssigner."
    )

    parser.add_argument(
        "--node_designations",
        required=True,
        help="SeqDesignation designated-node TSV."
    )

    parser.add_argument(
        "--sequence_designations",
        required=True,
        help="SeqDesignation per-sequence TSV."
    )

    parser.add_argument(
        "--metadata",
        default=None,
        help=(
            "Optional metadata TSV/CSV. Usually unnecessary when "
            "SeqDesignation output already contains every metadata column."
        )
    )

    parser.add_argument(
        "--metadata_id_column",
        default="ID",
        help="Metadata sequence identifier. Default: ID."
    )

    parser.add_argument(
        "--final_lineage_column",
        default="final_lineage",
        help="Final lineage column in node table."
    )

    parser.add_argument(
        "--internal_order",
        choices=["preorder", "postorder", "level"],
        default="preorder",
        help=(
            "Must match NodeAssigner internal numbering. "
            "Default: preorder."
        )
    )

    parser.add_argument(
        "--ancestral_prefix",
        default="NODE_",
        help="Internal node prefix. Default: NODE_."
    )

    parser.add_argument(
        "--ancestral_digits",
        type=int,
        default=7,
        help="NODE_* zero-padding digits. Default: 7."
    )

    parser.add_argument(
        "--ancestral_index_base",
        type=int,
        default=0,
        help="First NODE_* index. Default: 0."
    )

    parser.add_argument(
        "--unassigned_label",
        default="Unassigned",
        help="Unassigned sequence label."
    )

    parser.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix."
    )

    args = parser.parse_args()

    merger = MergeResults(
        tree=args.tree,
        node_designations=args.node_designations,
        sequence_designations=args.sequence_designations,
        metadata=args.metadata,
        metadata_id_column=args.metadata_id_column,
        final_lineage_column=args.final_lineage_column,
        internal_order=args.internal_order,
        ancestral_prefix=args.ancestral_prefix,
        ancestral_digits=args.ancestral_digits,
        ancestral_index_base=args.ancestral_index_base,
        unassigned_label=args.unassigned_label
    )

    merger.run(args.out_prefix)