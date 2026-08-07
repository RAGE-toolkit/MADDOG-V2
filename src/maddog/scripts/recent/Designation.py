#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from Bio import Phylo

'''
python src/maddog/scripts/Designation.py   --nodes test.tsv     --metadata clade_based_trees/Asian/metadata.tsv     --tree tmp/tree/Asian/Asian.iqtree.treefile     --metadata_id_column ID     --previous_column previous     --node_output tmp/designated_nodes.tsv     --sequence_output tmp/sequence_designations.tsv     --validation_output tmp/sequence_designation_validation.tsv
'''
class SeqDesignation:
    def __init__(
        self,
        nodes,
        metadata,
        tree,
        metadata_id_column,
        previous_column,
        lineage_column,
        unassigned_label,
        mixed_policy
    ):
        self.nodes = Path(nodes)
        self.metadata = Path(metadata)
        self.tree = Path(tree) if tree is not None else None
        self.metadata_id_column = metadata_id_column
        self.previous_column = previous_column
        self.lineage_column = lineage_column
        self.unassigned_label = unassigned_label
        self.mixed_policy = mixed_policy

    def detect_separator(self, path):
        first_line = Path(path).read_text(errors="replace").splitlines()[0]

        if "\t" in first_line:
            return "\t"

        return ","

    def read_table(self, path):
        separator = self.detect_separator(path)

        return pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            keep_default_na=False
        )

    def parse_tip_labels(self, value):
        if pd.isna(value):
            return []

        value = str(value).strip()

        if value == "" or value.upper() == "NA":
            return []

        return [
            tip.strip()
            for tip in value.split(",")
            if tip.strip()
        ]

    def lineage_depth(self, lineage):
        if pd.isna(lineage):
            return 0

        lineage = str(lineage).strip()

        if lineage == "":
            return 0

        return lineage.count(".")

    def validate_inputs(self, nodes, metadata):
        required_node_columns = {
            "node_id",
            self.lineage_column,
            "descendant_tip_labels"
        }

        missing_node_columns = required_node_columns.difference(nodes.columns)

        if missing_node_columns:
            raise ValueError(
                "NodeAssigner output is missing required columns: "
                + ", ".join(sorted(missing_node_columns))
            )

        required_metadata_columns = {
            self.metadata_id_column,
            self.previous_column
        }

        missing_metadata_columns = required_metadata_columns.difference(
            metadata.columns
        )

        if missing_metadata_columns:
            raise ValueError(
                "Metadata is missing required columns: "
                + ", ".join(sorted(missing_metadata_columns))
            )

        duplicate_ids = metadata[
            metadata[self.metadata_id_column].duplicated(keep=False)
        ]

        if not duplicate_ids.empty:
            examples = ", ".join(
                duplicate_ids[self.metadata_id_column]
                .astype(str)
                .head(10)
                .tolist()
            )

            raise ValueError(
                "Metadata IDs must be unique. Duplicate examples: "
                + examples
            )

        if nodes["node_id"].duplicated().any():
            raise ValueError(
                "NodeAssigner output contains duplicate node_id values."
            )

    def tree_tips(self):
        if self.tree is None:
            return []

        tree = Phylo.read(self.tree, "newick")

        return [
            str(tip.name)
            for tip in tree.get_terminals()
            if tip.name is not None
        ]

    def prepare_nodes(self, nodes):
        nodes = nodes.copy()

        nodes["node_id"] = nodes["node_id"].astype(str).str.strip()
        nodes[self.lineage_column] = (
            nodes[self.lineage_column].astype(str).str.strip()
        )

        nodes = nodes[
            (nodes["node_id"] != "")
            & (nodes[self.lineage_column] != "")
            & (nodes[self.lineage_column].str.upper() != "NA")
        ].copy()

        if "exclusive_tip_labels" not in nodes.columns:
            nodes["exclusive_tip_labels"] = ""

        if "exclusive_tip_count" not in nodes.columns:
            nodes["exclusive_tip_count"] = nodes[
                "exclusive_tip_labels"
            ].apply(
                lambda value: len(self.parse_tip_labels(value))
            )

        if "descendant_tip_count" not in nodes.columns:
            if "n_tips" in nodes.columns:
                nodes["descendant_tip_count"] = nodes["n_tips"]
            else:
                nodes["descendant_tip_count"] = nodes[
                    "descendant_tip_labels"
                ].apply(
                    lambda value: len(self.parse_tip_labels(value))
                )

        nodes["descendant_tip_count"] = pd.to_numeric(
            nodes["descendant_tip_count"],
            errors="coerce"
        ).fillna(0).astype(int)

        nodes["exclusive_tip_count"] = pd.to_numeric(
            nodes["exclusive_tip_count"],
            errors="coerce"
        ).fillna(0).astype(int)

        nodes["lineage_depth"] = nodes[self.lineage_column].apply(
            self.lineage_depth
        )

        if "Node" in nodes.columns:
            nodes["Node_numeric"] = pd.to_numeric(
                nodes["Node"],
                errors="coerce"
            )
        else:
            nodes["Node_numeric"] = pd.NA

        return nodes

    def metadata_maps(self, metadata):
        metadata = metadata.copy()

        metadata[self.metadata_id_column] = (
            metadata[self.metadata_id_column].astype(str).str.strip()
        )

        metadata[self.previous_column] = (
            metadata[self.previous_column].astype(str).str.strip()
        )

        previous_map = dict(
            zip(
                metadata[self.metadata_id_column],
                metadata[self.previous_column]
            )
        )

        return metadata, previous_map

    def previous_values_for_tips(self, tips, previous_map):
        values = []

        for tip in tips:
            value = previous_map.get(tip, "")

            if value == "":
                continue

            if value.upper() in {"NA", "NAN", "NONE"}:
                continue

            if value not in values:
                values.append(value)

        return values

    def common_parent_name(self, previous_values):
        if len(previous_values) == 0:
            return ""

        if len(previous_values) == 1:
            return previous_values[0]

        split_values = [
            value.split("_")
            for value in previous_values
        ]

        common_parts = []

        for parts in zip(*split_values):
            if len(set(parts)) == 1:
                common_parts.append(parts[0])
            else:
                break

        return "_".join(common_parts)

    def choose_base_lineage(self, previous_values):
        if len(previous_values) == 0:
            return "", "missing_previous"

        if len(previous_values) == 1:
            return previous_values[0], "consistent"

        if self.mixed_policy == "error":
            raise ValueError(
                "Mixed previous assignments found: "
                + ", ".join(previous_values)
            )

        if self.mixed_policy == "first":
            return previous_values[0], "mixed_first_used"

        if self.mixed_policy == "common":
            common_name = self.common_parent_name(previous_values)

            if common_name != "":
                return common_name, "mixed_common_parent"

        return "", "mixed_unresolved"

    def build_node_designations(self, nodes, previous_map):
        rows = []

        for _, row in nodes.iterrows():
            descendant_tips = self.parse_tip_labels(
                row.get("descendant_tip_labels", "")
            )

            exclusive_tips = self.parse_tip_labels(
                row.get("exclusive_tip_labels", "")
            )

            naming_tips = exclusive_tips

            if len(naming_tips) == 0:
                naming_tips = descendant_tips

            previous_values = self.previous_values_for_tips(
                naming_tips,
                previous_map
            )

            base_lineage, status = self.choose_base_lineage(
                previous_values
            )

            local_lineage = str(
                row[self.lineage_column]
            ).strip()

            if base_lineage == "":
                final_lineage = local_lineage
            elif local_lineage.startswith(base_lineage + "_"):
                final_lineage = local_lineage
            else:
                final_lineage = (
                    base_lineage
                    + "_"
                    + local_lineage
                )

            output_row = row.to_dict()
            output_row["previous_values"] = ", ".join(previous_values)
            output_row["base_lineage"] = base_lineage
            output_row["designation_status"] = status
            output_row["final_lineage"] = final_lineage
            output_row["naming_tip_count"] = len(naming_tips)

            rows.append(output_row)

        result = pd.DataFrame(rows)

        duplicate_names = result[
            result["final_lineage"].duplicated(keep=False)
            & (result["final_lineage"] != "")
        ]

        if not duplicate_names.empty:
            duplicate_text = ", ".join(
                sorted(
                    duplicate_names["final_lineage"]
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )

            raise ValueError(
                "Duplicate final lineage names were generated: "
                + duplicate_text
            )

        return result

    def build_sequence_designations(
        self,
        nodes,
        metadata,
        tree_tips
    ):
        assignment_rows = []

        ordered_nodes = nodes.sort_values(
            by=[
                "lineage_depth",
                "descendant_tip_count",
                "Node_numeric"
            ],
            ascending=[True, False, True],
            kind="mergesort"
        ).reset_index(drop=True)

        tip_to_assignment = {}

        for _, row in ordered_nodes.iterrows():
            descendant_tips = self.parse_tip_labels(
                row.get("descendant_tip_labels", "")
            )

            for tip in descendant_tips:
                tip_to_assignment[tip] = {
                    "ID": tip,
                    "cluster": row["final_lineage"],
                    "assigned_node_id": row["node_id"],
                    "local_lineage": row[self.lineage_column],
                    "base_lineage": row["base_lineage"],
                    "assignment_method": "deepest_retained_node"
                }

        all_ids = []

        for sequence_id in metadata[self.metadata_id_column].tolist():
            if sequence_id not in all_ids:
                all_ids.append(sequence_id)

        for sequence_id in tree_tips:
            if sequence_id not in all_ids:
                all_ids.append(sequence_id)

        for sequence_id in tip_to_assignment:
            if sequence_id not in all_ids:
                all_ids.append(sequence_id)

        for sequence_id in all_ids:
            assignment = tip_to_assignment.get(sequence_id)

            if assignment is None:
                assignment_rows.append(
                    {
                        "ID": sequence_id,
                        "cluster": self.unassigned_label,
                        "assigned_node_id": "",
                        "local_lineage": "",
                        "base_lineage": "",
                        "assignment_method": "unassigned"
                    }
                )
            else:
                assignment_rows.append(assignment)

        sequence_df = pd.DataFrame(assignment_rows)

        metadata_for_merge = metadata.rename(
            columns={
                self.metadata_id_column: "ID"
            }
        )

        sequence_df = sequence_df.merge(
            metadata_for_merge,
            on="ID",
            how="left",
            validate="one_to_one",
            suffixes=("", "_metadata")
        )

        if self.previous_column not in sequence_df.columns:
            sequence_df[self.previous_column] = ""

        return sequence_df

    def build_validation(
        self,
        nodes,
        sequence_designations,
        metadata,
        tree_tips
    ):
        issue_rows = []

        metadata_ids = set(
            metadata[self.metadata_id_column].astype(str)
        )

        tree_tip_set = set(tree_tips)

        for sequence_id in sorted(metadata_ids - tree_tip_set):
            if len(tree_tip_set) > 0:
                issue_rows.append(
                    {
                        "issue": "metadata_id_not_in_tree",
                        "value": sequence_id
                    }
                )

        for sequence_id in sorted(tree_tip_set - metadata_ids):
            issue_rows.append(
                {
                    "issue": "tree_tip_missing_metadata",
                    "value": sequence_id
                }
            )

        unresolved = nodes[
            nodes["designation_status"].isin(
                ["missing_previous", "mixed_unresolved"]
            )
        ]

        for _, row in unresolved.iterrows():
            issue_rows.append(
                {
                    "issue": row["designation_status"],
                    "value": row["node_id"]
                }
            )

        unassigned = sequence_designations[
            sequence_designations["cluster"] == self.unassigned_label
        ]

        for sequence_id in unassigned["ID"].astype(str):
            issue_rows.append(
                {
                    "issue": "sequence_unassigned",
                    "value": sequence_id
                }
            )

        return pd.DataFrame(
            issue_rows,
            columns=["issue", "value"]
        )

    def run(
        self,
        node_output,
        sequence_output,
        validation_output=None
    ):
        nodes = self.read_table(self.nodes)
        metadata = self.read_table(self.metadata)

        self.validate_inputs(nodes, metadata)

        nodes = self.prepare_nodes(nodes)
        metadata, previous_map = self.metadata_maps(metadata)
        tree_tips = self.tree_tips()

        node_designations = self.build_node_designations(
            nodes,
            previous_map
        )

        sequence_designations = self.build_sequence_designations(
            node_designations,
            metadata,
            tree_tips
        )

        validation = self.build_validation(
            node_designations,
            sequence_designations,
            metadata,
            tree_tips
        )

        node_output = Path(node_output)
        sequence_output = Path(sequence_output)

        node_output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        sequence_output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        node_designations.to_csv(
            node_output,
            sep="\t",
            index=False,
            na_rep="NA"
        )

        sequence_designations.to_csv(
            sequence_output,
            sep="\t",
            index=False,
            na_rep="NA"
        )

        if validation_output is not None:
            validation_output = Path(validation_output)
            validation_output.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            validation.to_csv(
                validation_output,
                sep="\t",
                index=False,
                na_rep="NA"
            )

        print(f"Wrote node designations to: {node_output}")
        print(f"Wrote sequence designations to: {sequence_output}")

        if validation_output is not None:
            print(f"Wrote validation report to: {validation_output}")

        return node_designations, sequence_designations, validation


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Assign final lineage names and sequences using "
            "NodeAssigner output and previous metadata assignments."
        )
    )

    parser.add_argument(
        "--nodes",
        required=True,
        help="NodeAssigner output TSV or CSV."
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata TSV or CSV."
    )

    parser.add_argument(
        "--tree",
        default=None,
        help=(
            "Optional exact rooted tree used by NodeAssigner. "
            "Used for validation and inclusion of all tree tips."
        )
    )

    parser.add_argument(
        "--metadata_id_column",
        default="ID",
        help="Metadata sequence ID column. Default: ID."
    )

    parser.add_argument(
        "--previous_column",
        default="previous",
        help="Previous lineage column. Default: previous."
    )

    parser.add_argument(
        "--lineage_column",
        default="lineage",
        help="NodeAssigner local lineage column. Default: lineage."
    )

    parser.add_argument(
        "--unassigned_label",
        default="Unassigned",
        help="Label for sequences outside retained nodes."
    )

    parser.add_argument(
        "--mixed_policy",
        choices=["error", "first", "common", "unresolved"],
        default="common",
        help=(
            "How to handle nodes containing multiple previous assignments. "
            "Default: common."
        )
    )

    parser.add_argument(
        "--node_output",
        required=True,
        help="Output designated-node TSV."
    )

    parser.add_argument(
        "--sequence_output",
        required=True,
        help="Output sequence-designation TSV."
    )

    parser.add_argument(
        "--validation_output",
        default=None,
        help="Optional validation-report TSV."
    )

    args = parser.parse_args()

    designation = SeqDesignation(
        nodes=args.nodes,
        metadata=args.metadata,
        tree=args.tree,
        metadata_id_column=args.metadata_id_column,
        previous_column=args.previous_column,
        lineage_column=args.lineage_column,
        unassigned_label=args.unassigned_label,
        mixed_policy=args.mixed_policy
    )

    designation.run(
        node_output=args.node_output,
        sequence_output=args.sequence_output,
        validation_output=args.validation_output
    )
