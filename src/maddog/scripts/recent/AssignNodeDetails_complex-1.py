from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from Bio import Phylo, SeqIO
from argparse import ArgumentParser
from Bio.Phylo.BaseTree import Clade, Tree

#python src/maddog/scripts/AssignNodeDetails.py --tree tmp/tree/Indian-Sub/Indian-Sub.iqtree.treefile --output maddog_ref_test --ancestral_fasta tmp/tree/Indian-Sub/Indian-Sub.treetime_ancestral/ancestral_sequences.fasta

class NodeAssigner:
    def __init__(self, tree, ancestral_fasta, base_dir, ancestral_prefix, ancestral_digits, ancestral_index_base, support_mode, internal_order, no_tip_labels, min_bootstrap, min_tips, min_parent_tip_diff, min_exclusive_tips):
        self.tree = tree
        self.ancestral_fasta = ancestral_fasta
        self.base_dir = base_dir
        self.ancestral_prefix = ancestral_prefix
        self.ancestral_digits = ancestral_digits
        self.ancestral_index_base = ancestral_index_base
        self.support_mode = support_mode
        self.internal_order = internal_order
        self.no_tip_labels = no_tip_labels
        self.min_bootstrap = min_bootstrap
        self.min_tips = min_tips
        self.min_parent_tip_diff = min_parent_tip_diff
        self.min_exclusive_tips = min_exclusive_tips

    def load_tree(self):
        if not hasattr(self, "_tree_cache"):
            self._tree_cache = Phylo.read(self.tree, "newick")
        return self._tree_cache

    def _internals_ape(self):
        if not hasattr(self, "_internals_cache"):
            tree = self.load_tree()
            self._internals_cache = tree.get_nonterminals(order="preorder")
        return self._internals_cache

    #def internals(self):
    #    return self._internals_ape()

    def terminals(self):
        return self.load_tree().get_terminals()

    def internals(self):
        return self.load_tree().get_nonterminals(order=self.internal_order)

    def assign_internal_node_names(self):
        internal_nodes = self.internals()

        internal_node_names = {}

        for index, node in enumerate(internal_nodes, start=self.ancestral_index_base):
            node_name = f"{self.ancestral_prefix}{index:0{self.ancestral_digits}d}"
            internal_node_names[node] = node_name

        return internal_node_names

    def get_node_name(self, node, internal_node_names):
        if node.is_terminal():
            return node.name

        return internal_node_names[node]

    def parent_nodes(self):
        tree = self.load_tree()

        parents = {}

        for parent in tree.find_clades(order="level"):
            for child in parent.clades:
                parents[child] = parent

        return parents

    def bootstrap(self, node):
        if node.is_terminal():
            return ""

        raw_value = ""

        if node.confidence is not None:
            raw_value = str(node.confidence)
        elif node.name is not None:
            raw_value = str(node.name)

        if raw_value == "":
            return ""

        values = re.findall(r"\d+(?:\.\d+)?", raw_value)

        if len(values) == 0:
            return ""

        if self.support_mode in ["ufboot", "second"]:
            if len(values) >= 2:
                return float(values[1])
            else:
                return float(values[0])

        if self.support_mode in ["first", "shalrt", "alrt"]:
            return float(values[0])

        return float(values[0])

    def node_table(self):
        tree = self.load_tree()

        ntips = len(tree.get_terminals())

        # NodeInfo/MADDOG-compatible internal-node order
        internal_nodes = tree.get_nonterminals(order="preorder")

        internal_node_names = {}
        internal_node_numbers = {}

        for i0, node in enumerate(internal_nodes):
            internal_pos = i0 + 1
            ape_node = ntips + internal_pos

            ancestral_index = self.ancestral_index_base + i0
            node_name = f"{self.ancestral_prefix}{ancestral_index:0{self.ancestral_digits}d}"

            internal_node_names[node] = node_name
            internal_node_numbers[node] = ape_node

        parents = {}

        for parent in tree.find_clades(order="preorder"):
            for child in parent.clades:
                parents[child] = parent

        rows = []

        for node in tree.find_clades(order="preorder"):

            if node.is_terminal():
                node_id = node.name
                node_type = "terminal"
                numeric_node = pd.NA
                node_key = ""
            else:
                node_id = internal_node_names[node]
                node_type = "internal"
                numeric_node = internal_node_numbers[node]
                node_key = node_id

            parent = parents.get(node)

            if parent is None:
                parent_id = ""
            elif parent.is_terminal():
                parent_id = parent.name
            else:
                parent_id = internal_node_names[parent]

            descendant_tips = [tip.name for tip in node.get_terminals()]
            bootstrap_value = self.bootstrap(node)

            row = {
                "Node": numeric_node,
                "node_id": node_id,
                "NODE_key": node_key,
                "parent_node_id": parent_id,
                "node_type": node_type,
                "bootstrap": bootstrap_value,
                "original_label": node.name,
                "branch_length": node.branch_length,
                "descendant_tip_count": len(descendant_tips),
                "n_tips": len(descendant_tips),
            }

            if self.no_tip_labels:
                row["descendant_tip_labels"] = ""
            else:
                row["descendant_tip_labels"] = ",".join(descendant_tips)

            rows.append(row)

        return pd.DataFrame(rows)


    #### new adds here
    def read_ancestral_fasta(self):
        if self.ancestral_fasta is None:
            return {}

        sequences = {}

        for record in SeqIO.parse(self.ancestral_fasta, "fasta"):
            sequences[record.id] = str(record.seq).upper()

        return sequences


    def read_observed_fasta(self, observed_fasta):
        if observed_fasta is None:
            return {}

        sequences = {}

        for record in SeqIO.parse(observed_fasta, "fasta"):
            seq = str(record.seq).upper()

            sequences[record.id] = seq

            if record.name:
                sequences[record.name] = seq

            if record.description:
                sequences[record.description] = seq

        return sequences


    def add_ancestral_sequence(self, df):
        ancestral_sequences = self.read_ancestral_fasta()

        df = df.copy()

        df["ancestral_sequence"] = df["node_id"].map(ancestral_sequences)
        df["ancestral_sequence_available"] = df["ancestral_sequence"].notna()

        df["ancestral_sequence_length"] = df["ancestral_sequence"].apply(
            lambda x: len(x) if isinstance(x, str) else pd.NA
        )

        return df


    def count_nt_differences(self, seq1, seq2):
        valid_bases = {"A", "C", "G", "T"}

        differences = 0

        for base1, base2 in zip(str(seq1).upper(), str(seq2).upper()):
            if base1 not in valid_bases:
                continue

            if base2 not in valid_bases:
                continue

            if base1 != base2:
                differences += 1

        return differences


    def ancestor_diff_positions_all(self, ancestral_sequence, descendant_sequences):
        if not isinstance(ancestral_sequence, str):
            return set()

        if len(descendant_sequences) == 0:
            return set()

        diff_positions = {
            i
            for i, (ancestor_base, descendant_base) in enumerate(
                zip(ancestral_sequence, descendant_sequences[0])
            )
            if ancestor_base != descendant_base
        }

        for descendant_sequence in descendant_sequences[1:]:
            current_diff_positions = {
                i
                for i, (ancestor_base, descendant_base) in enumerate(
                    zip(ancestral_sequence, descendant_sequence)
                )
                if ancestor_base != descendant_base
            }

            diff_positions = diff_positions.intersection(current_diff_positions)

        return diff_positions


    def add_descendant_difference(self, df, observed_fasta):
        observed_sequences = self.read_observed_fasta(observed_fasta)

        df = df.copy()

        compared_counts = []
        missing_counts = []
        min_differences = []
        max_differences = []
        maddog_diff_counts = []
        maddog_diff_positions = []
        has_descendant_differences = []
        descendants_with_differences = []

        for _, row in df.iterrows():

            ancestral_sequence = row.get("ancestral_sequence", pd.NA)

            descendant_tip_labels = row.get("descendant_tip_labels", "")

            if pd.isna(descendant_tip_labels):
                descendant_tips = []
            else:
                descendant_tips = str(descendant_tip_labels).split(",")

            compared_sequences = []
            differences = []
            descendants_diff_list = []
            missing_count = 0

            for tip_name in descendant_tips:
                observed_sequence = observed_sequences.get(tip_name)

                if observed_sequence is None:
                    missing_count += 1
                    continue

                compared_sequences.append(observed_sequence)

                if isinstance(ancestral_sequence, str):
                    nt_diff = self.count_nt_differences(
                        ancestral_sequence,
                        observed_sequence
                    )
                else:
                    nt_diff = 0

                differences.append(nt_diff)

                if nt_diff > 0:
                    descendants_diff_list.append(f"{tip_name}:{nt_diff}")

            if isinstance(ancestral_sequence, str):
                diff_positions = self.ancestor_diff_positions_all(
                    ancestral_sequence,
                    compared_sequences
                )
            else:
                diff_positions = set()

            diff_positions_1based = sorted([x + 1 for x in diff_positions])

            compared_counts.append(len(compared_sequences))
            missing_counts.append(missing_count)

            if len(differences) > 0:
                min_differences.append(min(differences))
                max_differences.append(max(differences))
            else:
                min_differences.append(pd.NA)
                max_differences.append(pd.NA)

            maddog_diff_counts.append(len(diff_positions))

            if len(diff_positions_1based) > 0:
                maddog_diff_positions.append(",".join(map(str, diff_positions_1based)))
            else:
                maddog_diff_positions.append(pd.NA)

            has_descendant_differences.append(len(diff_positions) > 0)

            if len(descendants_diff_list) > 0:
                descendants_with_differences.append(",".join(descendants_diff_list))
            else:
                descendants_with_differences.append(pd.NA)

        df["n_descendants_compared"] = compared_counts
        df["n_descendants_missing_observed_sequence"] = missing_counts
        df["min_descendant_nt_differences"] = min_differences
        df["max_descendant_nt_differences"] = max_differences
        df["maddog_diff_count"] = maddog_diff_counts
        df["maddog_diff_positions_1based"] = maddog_diff_positions
        df["has_descendant_nt_difference"] = has_descendant_differences
        df["descendants_with_difference"] = descendants_with_differences

        return df


    def filter_results(self, df, require_descendant_difference=True):
        result_df = df.copy()

        result_df = result_df[result_df["node_type"] == "internal"].copy()

        result_df = result_df[
            result_df["bootstrap"].notna()
        ].copy()

        result_df = result_df[
            result_df["bootstrap"] >= self.min_bootstrap
        ].copy()

        result_df = result_df[
            result_df["descendant_tip_count"] >= self.min_tips
        ].copy()

        if require_descendant_difference:
            if "has_descendant_nt_difference" in result_df.columns:
                result_df = result_df[
                    result_df["has_descendant_nt_difference"] == True
                ].copy()

        return result_df


    def debug_output_path(self, output):
        output_path = Path(output)

        return output_path.with_name(
            f"{output_path.stem}_debug.tsv"
        )


    def write_table(self, df, output):
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if str(output_path).endswith(".csv"):
            df.to_csv(output_path, index=False, na_rep="NA")
        else:
            df.to_csv(output_path, sep="\t", index=False, na_rep="NA")


    def run(self, output, observed_fasta=None, require_descendant_difference=True):
        raw_df = self.node_table()

        raw_df = self.add_ancestral_sequence(raw_df)

        if observed_fasta is not None:
            raw_df = self.add_descendant_difference(raw_df, observed_fasta)

        result_df = raw_df.copy()

        result_df = self.filter_results(
            result_df,
            require_descendant_difference=require_descendant_difference
        )

        debug_output = self.debug_output_path(output)

        self.write_table(result_df, output)
        self.write_table(raw_df, debug_output)

        print(f"Wrote filtered results to: {output}")
        print(f"Wrote raw debug results to: {debug_output}")


    def filter_and_save_debug(
        self,
        df,
        output_path,
        observed_fasta=None,
        require_descendant_difference=True
    ):
        raw_df = df.copy()

        # Add ancestral sequence for NODE_* rows
        raw_df = self.add_ancestral_sequence(raw_df)

        # NodeInfo uses the ancestral FASTA dictionary for NODE_* and tip IDs.
        # If observed_fasta is not provided, use ancestral_fasta as the descendant-tip source.
        fasta_for_descendants = observed_fasta or self.ancestral_fasta

        if fasta_for_descendants is not None:
            raw_df = self.add_descendant_difference(raw_df, fasta_for_descendants)

        if "maddog_diff_count" in raw_df.columns:
            raw_df["diff_count"] = raw_df["maddog_diff_count"]
            raw_df["n_diff_ancestor"] = raw_df["maddog_diff_count"]

        debug_path = Path(output_path).with_name(
            f"{Path(output_path).stem}_debug.tsv"
        )

        raw_df.to_csv(debug_path, sep="\t", index=False)

        result_df = raw_df.copy()

        result_df["bootstrap_numeric"] = pd.to_numeric(
            result_df["bootstrap"],
            errors="coerce"
        )

        result_df = result_df[
            result_df["node_type"] == "internal"
        ].copy()

        result_df = result_df[
            result_df["bootstrap_numeric"].notna()
        ].copy()

        result_df = result_df[
            result_df["bootstrap_numeric"] >= self.min_bootstrap
        ].copy()

        result_df = result_df[
            result_df["descendant_tip_count"] >= self.min_tips
        ].copy()

        # NodeInfo-like difference filter: keep only nodes with diff_count != 0
        if require_descendant_difference:
            if "maddog_diff_count" not in result_df.columns:
                raise ValueError(
                    "maddog_diff_count column is missing. "
                    "Provide --ancestral_fasta containing NODE_* and tip sequences, "
                    "or provide --observed_fasta."
                )

            result_df["maddog_diff_count_numeric"] = pd.to_numeric(
                result_df["maddog_diff_count"],
                errors="coerce"
            ).fillna(0)

            result_df = result_df[
                result_df["maddog_diff_count_numeric"] != 0
            ].copy()

        # overlap handling
        result_df = self.add_overlap_information(result_df)

        overlap_debug_path = Path(output_path).with_name(
            f"{Path(output_path).stem}_overlap_debug.tsv"
        )
        result_df.to_csv(overlap_debug_path, sep="\t", index=False)

        # MADDOG-like step:
        # assign tips to the lowest/deepest qualifying node,
        # then remove nodes with fewer than 2 uniquely assigned tips.
        result_df = self.remove_nodes_with_few_exclusive_tips(
        result_df,
        min_exclusive_tips=self.min_exclusive_tips
        )

        exclusive_debug_path = Path(output_path).with_name(
            f"{Path(output_path).stem}_exclusive_tip_debug.tsv"
        )
        result_df.to_csv(exclusive_debug_path, sep="\t", index=False)

        # Now remove near-identical parent-child overlaps, using your existing logic.
        result_df = self.remove_nearly_identical_overlaps(result_df)

        # Recompute overlap/parent columns after removing near-identical child nodes.
        result_df = self.add_overlap_information(result_df)

        # Recompute final exclusive tip counts after parent-child filtering.
        result_df = self.add_exclusive_tip_assignment(result_df)

        result_df = self.assign_hierarchical_lineages(result_df)

        result_df = result_df.drop(
            columns=["bootstrap_numeric", "maddog_diff_count_numeric"],
            errors="ignore"
        )

        return result_df
    

    # -----------------------------------------------------------
    # Dealing with overlaps -------------------------------------
    # -----------------------------------------------------------

    def _tip_set(self, value):
        if pd.isna(value) or str(value).strip() == "":
            return set()

        return set(
            tip.strip()
            for tip in str(value).split(",")
            if tip.strip()
        )


    def add_overlap_information(self, df):
        """
        NodeInfo-like overlap logic.

        overlaps = number of selected internal descendant nodes.
        direct_parent_overlap_node = closest selected ancestral internal node.
        parent_child_tip_difference = parent n_tips - child n_tips.
        """

        df = df.copy()

        internal_df = df[df["node_type"] == "internal"].copy()

        if "Node" not in internal_df.columns:
            raise ValueError(
                "Node column is missing. Update node_table() to add numeric Node values."
            )

        internal_df["Node_numeric"] = pd.to_numeric(
            internal_df["Node"],
            errors="coerce"
        )

        internal_df = internal_df[
            internal_df["Node_numeric"].notna()
        ].copy()

        internal_df["Node_numeric"] = internal_df["Node_numeric"].astype(int)

        selected_nodes = set(internal_df["Node_numeric"].tolist())

        node_to_id = dict(
            zip(
                internal_df["Node_numeric"],
                internal_df["node_id"]
            )
        )

        node_to_tip_count = dict(
            zip(
                internal_df["Node_numeric"],
                internal_df["descendant_tip_count"]
            )
        )

        overlap_child_count = {}
        overlapping_child_nodes = {}
        overlapping_child_numeric_nodes = {}
        direct_parent_overlap_node = {}
        direct_parent_overlap_numeric_node = {}
        parent_child_tip_difference = {}

        for node_number in selected_nodes:

            descendants = self._internal_descendants_ape(
                node_number,
                include_self=False
            )

            selected_descendants = [
                x for x in descendants
                if x in selected_nodes
            ]

            overlap_child_count[node_number] = len(selected_descendants)

            overlapping_child_numeric_nodes[node_number] = ",".join(
                str(x) for x in selected_descendants
            )

            overlapping_child_nodes[node_number] = ",".join(
                node_to_id.get(x, str(x))
                for x in selected_descendants
            )

            closest_parent = ""

            for ancestor in self._internal_ancestors_ape(node_number):
                if ancestor in selected_nodes:
                    closest_parent = ancestor
                    break

            if closest_parent == "":
                direct_parent_overlap_numeric_node[node_number] = pd.NA
                direct_parent_overlap_node[node_number] = ""
                parent_child_tip_difference[node_number] = pd.NA
            else:
                direct_parent_overlap_numeric_node[node_number] = closest_parent
                direct_parent_overlap_node[node_number] = node_to_id.get(
                    closest_parent,
                    str(closest_parent)
                )

                parent_tips = int(node_to_tip_count[closest_parent])
                child_tips = int(node_to_tip_count[node_number])

                parent_child_tip_difference[node_number] = parent_tips - child_tips

        df["Node_numeric"] = pd.to_numeric(df["Node"], errors="coerce")

        df["overlaps"] = df["Node_numeric"].map(overlap_child_count).fillna(0).astype(int)
        df["overlap_child_count"] = df["overlaps"]

        df["overlapping_child_nodes"] = df["Node_numeric"].map(
            overlapping_child_nodes
        ).fillna("")

        df["overlapping_child_numeric_nodes"] = df["Node_numeric"].map(
            overlapping_child_numeric_nodes
        ).fillna("")

        df["direct_parent_overlap_node"] = df["Node_numeric"].map(
            direct_parent_overlap_node
        ).fillna("")

        df["direct_parent_overlap_numeric_node"] = df["Node_numeric"].map(
            direct_parent_overlap_numeric_node
        )

        df["parent_child_tip_difference"] = df["Node_numeric"].map(
            parent_child_tip_difference
        )

        df = df.drop(columns=["Node_numeric"], errors="ignore")

        return df


    def remove_nearly_identical_overlaps(self, df):
        """
        Removes nested child nodes that are too similar to their parent.

        Example:
            Parent = 291 tips
            Child  = 290 tips
            Difference = 1

        If min_parent_tip_diff = 5, this child is removed.
        """

        df = df.copy()

        if "parent_child_tip_difference" not in df.columns:
            df = self.add_overlap_information(df)

        keep_rows = []

        for _, row in df.iterrows():
            node_type = row.get("node_type", "")

            if node_type != "internal":
                keep_rows.append(True)
                continue

            parent_node = row.get("direct_parent_overlap_node", "")
            tip_diff = row.get("parent_child_tip_difference", pd.NA)

            # No overlapping parent, keep it
            if parent_node == "" or pd.isna(tip_diff):
                keep_rows.append(True)
                continue

            # Child is meaningfully different from parent
            if int(tip_diff) >= self.min_parent_tip_diff:
                keep_rows.append(True)
            else:
                keep_rows.append(False)

        return df[keep_rows].copy()


    def assign_hierarchical_lineages(self, df, root_prefix="A"):
        """
        Assigns simple hierarchical names after overlap filtering.

        Parent node: A1
        Child node:  A1.1
        Child node:  A1.2
        Next parent: A2
        """

        df = df.copy()

        sort_cols = []

        if "overlaps" in df.columns:
            sort_cols.append("overlaps")

        if "descendant_tip_count" in df.columns:
            sort_cols.append("descendant_tip_count")

        if "Node" in df.columns:
            sort_cols.append("Node")

        if sort_cols:
            ascending = []

            for col in sort_cols:
                if col in ["overlaps", "descendant_tip_count"]:
                    ascending.append(False)
                else:
                    ascending.append(True)

            df = df.sort_values(
                by=sort_cols,
                ascending=ascending,
                kind="mergesort"
            ).reset_index(drop=True)


        if "direct_parent_overlap_node" not in df.columns:
            df = self.add_overlap_information(df)

        df["lineage"] = ""

        node_ids = df["node_id"].tolist()
        node_set = set(node_ids)

        children_by_parent = {}

        for _, row in df.iterrows():
            node_id = row["node_id"]
            parent_id = row.get("direct_parent_overlap_node", "")

            if parent_id in node_set:
                children_by_parent.setdefault(parent_id, []).append(node_id)

        top_nodes = []

        for _, row in df.iterrows():
            node_id = row["node_id"]
            parent_id = row.get("direct_parent_overlap_node", "")

            if parent_id == "" or parent_id not in node_set:
                top_nodes.append(node_id)

        lineage_map = {}

        for i, node_id in enumerate(top_nodes, start=1):
            lineage_map[node_id] = f"{root_prefix}{i}"

            stack = [(node_id, lineage_map[node_id])]

            while stack:
                parent_id, parent_lineage = stack.pop()

                children = children_by_parent.get(parent_id, [])

                for child_index, child_id in enumerate(children, start=1):
                    child_lineage = f"{parent_lineage}.{child_index}"
                    lineage_map[child_id] = child_lineage
                    stack.append((child_id, child_lineage))

        df["lineage"] = df["node_id"].map(lineage_map).fillna("")

        return df
    
    def _ape_parent_map(self):
        tree = self.load_tree()
        ntips = len(tree.get_terminals())
        internals = self._internals_ape()

        clade_to_ape = {
            id(clade): ntips + i + 1
            for i, clade in enumerate(internals)
        }

        ape_to_clade = {
            ntips + i + 1: clade
            for i, clade in enumerate(internals)
        }

        parent_by_id = {}

        for parent in tree.find_clades(order="preorder"):
            for child in parent.clades:
                parent_by_id[id(child)] = parent

        ape_parent = {}

        for ape_node, clade in ape_to_clade.items():
            parent = parent_by_id.get(id(clade))

            while parent is not None and id(parent) not in clade_to_ape:
                parent = parent_by_id.get(id(parent))

            if parent is None:
                ape_parent[ape_node] = None
            else:
                ape_parent[ape_node] = clade_to_ape[id(parent)]

        return ape_parent


    def _internal_descendants_ape(self, ape_node, include_self=False):
        tree = self.load_tree()
        ntips = len(tree.get_terminals())
        internals = self._internals_ape()

        internal_pos = int(ape_node) - ntips

        if internal_pos < 1 or internal_pos > len(internals):
            return []

        clade = internals[internal_pos - 1]

        clade_to_ape = {
            id(clade_obj): ntips + i + 1
            for i, clade_obj in enumerate(internals)
        }

        descendants = []

        for i, descendant in enumerate(clade.get_nonterminals(order="preorder")):
            if i == 0 and not include_self:
                continue

            descendant_node = clade_to_ape.get(id(descendant))

            if descendant_node is not None:
                descendants.append(descendant_node)

        return descendants


    def _internal_ancestors_ape(self, ape_node):
        ape_parent = self._ape_parent_map()

        ancestors = []
        current = int(ape_node)

        while True:
            parent = ape_parent.get(current)

            if parent is None:
                break

            ancestors.append(parent)
            current = parent

        return ancestors

    def add_exclusive_tip_assignment(self, df):
        """
        MADDOG-like exclusive tip assignment.

        Candidate nodes are ordered by:
            1. overlaps, highest first
            2. descendant_tip_count, highest first
            3. Node number, lowest first

        Tips are assigned repeatedly in this order.
        Later/deeper candidate nodes overwrite earlier/parent assignments.

        This gives the number of tips uniquely assigned to each candidate node.
        """

        df = df.copy()

        if "overlaps" not in df.columns:
            df = self.add_overlap_information(df)

        if df.empty:
            df["exclusive_tip_count"] = []
            df["exclusive_tip_labels"] = []
            return df

        ordered_df = df.copy()

        ordered_df["Node_numeric_sort"] = pd.to_numeric(
            ordered_df["Node"],
            errors="coerce"
        )

        ordered_df = ordered_df.sort_values(
            by=["overlaps", "descendant_tip_count", "Node_numeric_sort"],
            ascending=[False, False, True],
            kind="mergesort"
        ).reset_index(drop=True)

        tip_to_node = {}

        for _, row in ordered_df.iterrows():
            node_id = row["node_id"]
            descendant_tips = self._tip_set(row.get("descendant_tip_labels", ""))

            for tip in descendant_tips:
                tip_to_node[tip] = node_id

        exclusive_tip_counts = []
        exclusive_tip_labels = []

        for _, row in df.iterrows():
            node_id = row["node_id"]

            assigned_tips = sorted(
                tip for tip, assigned_node in tip_to_node.items()
                if assigned_node == node_id
            )

            exclusive_tip_counts.append(len(assigned_tips))

            if len(assigned_tips) > 0:
                exclusive_tip_labels.append(",".join(assigned_tips))
            else:
                exclusive_tip_labels.append("")

        df["exclusive_tip_count"] = exclusive_tip_counts
        df["exclusive_tip_labels"] = exclusive_tip_labels

        return df


    def remove_nodes_with_few_exclusive_tips(
        self,
        df,
        min_exclusive_tips=2,
        max_iterations=100
    ):
        """
        MADDOG-like iterative removal of candidate nodes with too few
        uniquely assigned tips.

        Original MADDOG repeats the exclusive assignment until no selected
        cluster has fewer than 2 uniquely assigned sequences.
        """

        result_df = df.copy()

        iteration = 0

        while True:
            if result_df.empty:
                return result_df

            result_df = self.add_overlap_information(result_df)
            result_df = self.add_exclusive_tip_assignment(result_df)

            too_small = result_df["exclusive_tip_count"] < min_exclusive_tips

            if not too_small.any():
                return result_df

            result_df = result_df.loc[~too_small].copy()

            iteration += 1

            if iteration > max_iterations:
                raise RuntimeError(
                    "Too many iterations while removing nodes with few exclusive tips. "
                    "Check overlap logic or input tree."
                )

if __name__ == "__main__":
    parser = ArgumentParser(description="Link ancestral node details to tree")

    parser.add_argument( "--tree", "--contree", dest="tree", required=True, help="IQ-TREE Newick tree file, for example .treefile or .contree.",)
    parser.add_argument( "--ancestral_fasta", default=None, help="Optional ancestral sequences FASTA. IDs usually look like NODE_0000000.",)
    parser.add_argument("--base_dir", default="tmp", help="Base directory where the output directory or files are created")
    parser.add_argument("--output",default="NodeAssignment",help="Output table path. Use .csv for CSV; otherwise TSV is written.",)
    parser.add_argument("--ancestral_prefix",default="NODE_",help="Prefix used by ancestral node FASTA IDs. Default: NODE_",)
    parser.add_argument("--ancestral_digits",type=int,default=7,help="Number of zero-padding digits in ancestral node IDs. Default: 7",)
    parser.add_argument("--ancestral_index_base",type=int,default=0,help="Base index for ancestral node IDs. Default 0 maps first internal node to NODE_0000000.",)
    parser.add_argument("--support_mode",choices=["ufboot", "second", "shalrt", "alrt", "first"],default="ufboot",help="How to parse labels like SH-aLRT/UFBoot. Default: ufboot/second value.",)
    parser.add_argument("--internal_order",choices=["preorder", "postorder", "level"],default="preorder",help="Internal node traversal order. Keep preorder to match the referenced MADDOG node_info.py logic.",)
    parser.add_argument( "--no_tip_labels",action="store_true",help="Do not write the descendant_tip_labels column. Useful for very large trees.",)
    parser.add_argument("--min_bootstrap",type=int,default=95.0,help="Minimum bootstrap/UFBoot value required to keep an internal node. Default: 95.",)
    parser.add_argument("--min_tips",type=int,default=10,help="Minimum number of descendant tips required to keep an internal node. Default: 10, matching the MADDOG concept.",)
    parser.add_argument("--observed_fasta",default=None,help="Observed aligned FASTA used for descendant difference checking.")
    parser.add_argument("--no_descendant_difference_filter",action="store_true",help="Do not filter by descendant ancestral sequence difference.")
    parser.add_argument("--min_parent_tip_diff",type=int,default=5,help="Minimum tip difference between a parent and child node to retain the child. Default: 5")
    parser.add_argument("--min_exclusive_tips",type=int,default=2,help="Minimum number of uniquely assigned tips required to keep a candidate node. Default: 2, matching MADDOG overlap handling.")
    
    args = parser.parse_args()

    checker = NodeAssigner(
        args.tree,
        args.ancestral_fasta,
        args.base_dir,
        args.ancestral_prefix,
        args.ancestral_digits,
        args.ancestral_index_base,
        args.support_mode,
        args.internal_order,
        args.no_tip_labels,
        args.min_bootstrap,
        args.min_tips,
        args.min_parent_tip_diff,
        args.min_exclusive_tips
    )

    df = checker.node_table()
    
    #df = checker.filter_and_save_debug(df,Path(args.base_dir) / args.output)

    df = checker.filter_and_save_debug(df,Path(args.base_dir) / args.output, observed_fasta=args.observed_fasta, require_descendant_difference=not args.no_descendant_difference_filter)

    output_path = Path(args.base_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, sep="\t", index=False)