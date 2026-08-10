# MADDOG-V2

MADDOG-V2 is a phylogenetic lineage assignment and analysis workflow.

## Installation

Create the Conda environment:

```bash
conda env create -f env.yml
```

Activate the environment:

```bash
conda activate maddog_v2
```

Install MADDOG-V2 from the wheel package:

```bash
pip install dist/maddog-2.0.0-py3-none-any.whl
```

## Check installation

After installation, check the available MADDOG commands:

```bash
maddog-assignment --help
maddog-phylogeny --help
maddog-reroot --help
maddog-assign-nodes --help
maddog-lineage-assignment --help
maddog-reduce-lineages --help
```

## Running the pipeline

The MADDOG-V2 workflow can be run step-by-step using the commands below.

### 1. Clade assignment

Assign query sequences to major and minor clades:

```bash
maddog-assignment \
    --ref-aln reference/rabv/reference_sequences/ref.fa \
    --ref-tree reference/rabv/tree/ref.treefile \
    --query test_data/sea4_test/sea4.fa \
    --threads 5 \
    --taxon-major reference/rabv/epa_clades/ref_major_clades.tsv \
    --taxon-minor reference/rabv/epa_clades/ref_minor_clades.tsv
```

### 2. Phylogeny reconstruction

Generate clade-specific phylogenetic trees using the assignment output:

```bash
maddog-phylogeny \
    --assignment tmp/CladeAssignment/clade_annotation.tsv \
    --input-alignment tmp/CladeAssignment/input_seqs_with_ref_alignment.fa \
    --clade-tree-dir clade_based_trees/
```

### 3. Re-root the phylogenetic tree

Re-root the generated tree and order the nodes:

```bash
maddog-reroot \
    --input_tree tmp/tree/Asian/Asian.iqtree.treefile \
    --output tmp/tree/Asian/Asian.iqtree.rooted.treefile \
    --order_node decrease
```

### 4. Assign node details

Assign node information using the rooted tree and ancestral sequences:

```bash
maddog-assign-nodes \
    --tree tmp/tree/Asian/Asian.iqtree.rooted.treefile \
    --ancestral_fasta tmp/tree/Asian/Asian.treetime_ancestral/ancestral_sequences.fasta \
    --base_dir tmp \
    --output NodesAssigned.tsv \
    --reference_lineages clade_based_trees/Asian/metadata.tsv
```

### 5. Lineage assignment

Assign lineage labels to the tips using the phylogenetic tree and node assignment information:

```bash
maddog-lineage-assignment \
    --tree tmp/tree/Asian/Asian.iqtree.treefile \
    --node_assignment tmp/NodesAssigned.tsv \
    --out_prefix sea4_test
```

### 6. Reduce lineage assignments

Reduce closely related lineage assignments using lineage size and MRCA-based information:

```bash
maddog-reduce-lineages \
    --assignments tmp/lineage_assignment/sea4_test.tip_lineage_assignments.tsv \
    --output test_sea4.reduced.tip_lineage_assignments.tsv \
    --mapping-output test_sea4.lineage_reduction_map.tsv \
    --summary-output test_sea4.reduced.lineage_summary.tsv \
    --min-lineage-size 20 \
    --tree tmp/tree/Asian/Asian.iqtree.rooted.treefile \
    --mrca-output mrca
```

## Version

MADDOG-V2
Version 2.0.0
