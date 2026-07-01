from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from Bio import Phylo, SeqIO
from argparse import ArgumentParser
from Bio.Phylo.BaseTree import Clade, Tree

class NodeAssigner:
    def __init__(self, tree, ancestral_fasta, base_dir, ancestral_prefix, ancestral_digits, ancestral_index_base, support_mode, internal_order, no_tip_labels, min_bootstrap, min_tips):
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

    def load_tree(self):
        return Phylo.read(self.tree, "newick")

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

        internal_node_names = {}

        internal_nodes = tree.get_nonterminals(order=self.internal_order)

        for index, node in enumerate(internal_nodes, start=self.ancestral_index_base):
            node_name = f"{self.ancestral_prefix}{index:0{self.ancestral_digits}d}"
            internal_node_names[node] = node_name

        parents = {}

        for parent in tree.find_clades(order="level"):
            for child in parent.clades:
                parents[child] = parent

        rows = []

        for node in tree.find_clades(order=self.internal_order):

            if node.is_terminal():
                node_id = node.name
                node_type = "terminal"
            else:
                node_id = internal_node_names[node]
                node_type = "internal"

            parent = parents.get(node)

            if parent is None:
                parent_id = ""
            elif parent.is_terminal():
                parent_id = parent.name
            else:
                parent_id = internal_node_names[parent]
            
            #adding additional values
            descendant_tips = [tip.name for tip in node.get_terminals()]
            bootstrap_value = self.bootstrap(node)

            rows.append({
                "node_id": node_id,
                "parent_node_id": parent_id,
                "node_type": node_type,
                "bootstrap": bootstrap_value,

                "original_label": node.name,
                "branch_length": node.branch_length,
                "descendant_tip_count": len(node.get_terminals()),
                "descendant_tip_labels": ",".join(descendant_tips),
                
            })

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


    def filter_and_save_debug(self, df, output_path):
        raw_df = df.copy()

        debug_path = Path(output_path).with_name(
            f"{Path(output_path).stem}_debug.tsv"
        )

        raw_df.to_csv(debug_path, sep="\t", index=False)

        result_df = df.copy()

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

        result_df = result_df.drop(columns=["bootstrap_numeric"])

        return result_df

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
        args.min_tips
    )

    df = checker.node_table()
    
    df = checker.filter_and_save_debug(df,Path(args.base_dir) / args.output)

    output_path = Path(args.base_dir) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, sep="\t", index=False)