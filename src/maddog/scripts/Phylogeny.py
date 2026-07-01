#!/usr/bin/env python3
"""
Run clade-wise IQ-TREE using a clade assignment TSV and pre-existing clade trees.

For each EPA_major_clade:
  1. Extract only query sequence IDs from --input-alignment using --assignment.
  2. Write <clade>.query.aligned.fa.
  3. Find the matching clade directory under --clade-tree-dir.
  4. Find the clade reference alignment and existing tree in that directory.
  5. Merge clade reference alignment + extracted query alignment.
  6. Run IQ-TREE using the existing tree as a constraint/start tree.

Default outputs:
  tmp/tree/<clade>/<clade>.query.aligned.fa
  tmp/tree/<clade>/<clade>.merged.aligned.fa
  tmp/tree/<clade>/<clade>.iqtree.*
  tmp/tree/phylogeny_summary.tsv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import logging
import re
import shlex
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FASTA_EXTENSIONS = (".fa", ".fasta", ".fas", ".aln", ".fna")
TREE_EXTENSIONS = (".treefile", ".nwk", ".newick", ".tre", ".tree")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "unknown_clade"


def parse_aliases(alias_args: Optional[Sequence[str]]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    if not alias_args:
        return aliases

    for item in alias_args:
        if "=" not in item:
            raise ValueError(
                f"Bad --clade-dir-alias value '{item}'. Expected: AssignmentClade=DirectoryName"
            )
        source, target = item.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(
                f"Bad --clade-dir-alias value '{item}'. Expected: AssignmentClade=DirectoryName"
            )
        aliases[source] = target

    return aliases


def read_assignment(
    assignment_path: Path,
    clade_column: str,
    sequence_column: str,
) -> "OrderedDict[str, List[str]]":
    if not assignment_path.exists():
        raise FileNotFoundError(f"Assignment file not found: {assignment_path}")

    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    seen_per_clade: Dict[str, set] = {}

    with assignment_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Assignment file has no header: {assignment_path}")

        missing = [col for col in (sequence_column, clade_column) if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Assignment file is missing required column(s): {', '.join(missing)}. "
                f"Available columns: {', '.join(reader.fieldnames)}"
            )

        for line_number, row in enumerate(reader, start=2):
            seq_id = (row.get(sequence_column) or "").strip()
            clade = (row.get(clade_column) or "").strip()

            if not seq_id:
                logging.warning("Skipping line %s because %s is empty", line_number, sequence_column)
                continue

            if not clade or clade.upper() == "NULL":
                logging.warning("Skipping %s because %s is empty/NULL", seq_id, clade_column)
                continue

            groups.setdefault(clade, [])
            seen_per_clade.setdefault(clade, set())

            if seq_id not in seen_per_clade[clade]:
                groups[clade].append(seq_id)
                seen_per_clade[clade].add(seq_id)

    if not groups:
        raise ValueError(f"No usable clade assignments found in {assignment_path}")

    return groups


def read_fasta(path: Path) -> "OrderedDict[str, Tuple[str, str]]":
    """
    Return OrderedDict:
      key   = first token in FASTA header
      value = (full header without '>', sequence)
    """
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()
    header: Optional[str] = None
    chunks: List[str] = []

    def commit() -> None:
        nonlocal header, chunks
        if header is None:
            return

        seq_id = header.split()[0]
        sequence = "".join(chunks).replace(" ", "").replace("\t", "")
        if not sequence:
            logging.warning("Ignoring empty FASTA record: %s", header)
        elif seq_id in records:
            raise ValueError(f"Duplicate FASTA ID '{seq_id}' found in {path}")
        else:
            records[seq_id] = (header, sequence)

        header = None
        chunks = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                commit()
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Blank FASTA header found in {path}")
            else:
                chunks.append(line.strip())

    commit()

    if not records:
        raise ValueError(f"No FASTA records found in {path}")

    return records


def write_fasta(records: Iterable[Tuple[str, str]], output_path: Path, line_width: int = 80) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for i in range(0, len(sequence), line_width):
                handle.write(sequence[i : i + line_width] + "\n")
            count += 1

    if count == 0:
        raise ValueError(f"No records were written to {output_path}")

    return count


def extract_records(
    fasta_records: "OrderedDict[str, Tuple[str, str]]",
    wanted_ids: Sequence[str],
    source_description: str,
) -> List[Tuple[str, str]]:
    extracted: List[Tuple[str, str]] = []
    missing: List[str] = []

    for seq_id in wanted_ids:
        if seq_id in fasta_records:
            extracted.append(fasta_records[seq_id])
        else:
            missing.append(seq_id)

    if missing:
        preview = ", ".join(missing[:10])
        extra = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
        raise ValueError(
            f"{len(missing)} assignment sequence(s) were not found in {source_description}: "
            f"{preview}{extra}"
        )

    return extracted


def find_clade_directory(
    clade: str,
    clade_tree_dir: Path,
    aliases: Dict[str, str],
    strict: bool = False,
) -> Path:
    if not clade_tree_dir.exists():
        raise FileNotFoundError(f"Clade tree directory not found: {clade_tree_dir}")

    requested = aliases.get(clade, clade)
    direct = clade_tree_dir / requested
    if direct.is_dir():
        return direct

    directories = [path for path in clade_tree_dir.iterdir() if path.is_dir()]
    by_lower = {path.name.lower(): path for path in directories}
    if requested.lower() in by_lower:
        return by_lower[requested.lower()]

    by_normalised = {normalise_name(path.name): path for path in directories}
    norm_requested = normalise_name(requested)
    if norm_requested in by_normalised:
        return by_normalised[norm_requested]

    if not strict:
        close = difflib.get_close_matches(norm_requested, list(by_normalised.keys()), n=1, cutoff=0.82)
        if close:
            matched = by_normalised[close[0]]
            logging.warning(
                "Using closest clade directory for '%s': %s. "
                "To make this explicit, use: --clade-dir-alias %s=%s",
                clade,
                matched,
                clade,
                matched.name,
            )
            return matched

    available = ", ".join(sorted(path.name for path in directories))
    raise FileNotFoundError(
        f"No clade directory found for '{clade}' under {clade_tree_dir}. "
        f"Available directories: {available}"
    )


def score_tree_candidate(path: Path, clade: str) -> Tuple[int, str]:
    clade_norm = normalise_name(clade)
    stem_norm = normalise_name(path.stem)

    ext_rank = {
        ".treefile": 0,
        ".nwk": 1,
        ".newick": 2,
        ".tre": 3,
        ".tree": 4,
    }.get(path.suffix.lower(), 10)

    clade_bonus = 0 if clade_norm and clade_norm in stem_norm else 1
    return (ext_rank + clade_bonus, str(path))


def find_tree_file(clade_dir: Path, clade: str, tree_file: Optional[Path] = None) -> Path:
    if tree_file is not None:
        resolved = tree_file if tree_file.is_absolute() else clade_dir / tree_file
        if not resolved.exists():
            raise FileNotFoundError(f"Tree file not found: {resolved}")
        return resolved

    candidates: List[Path] = []
    for ext in TREE_EXTENSIONS:
        candidates.extend(clade_dir.glob(f"*{ext}"))

    if not candidates:
        for ext in TREE_EXTENSIONS:
            candidates.extend(clade_dir.rglob(f"*{ext}"))

    if not candidates:
        raise FileNotFoundError(
            f"No tree file found in {clade_dir}. Expected one of: {', '.join(TREE_EXTENSIONS)}"
        )

    candidates = sorted(set(candidates), key=lambda path: score_tree_candidate(path, clade))
    if len(candidates) > 1:
        logging.info("Tree candidates for %s: %s", clade, ", ".join(str(p) for p in candidates))
        logging.info("Selected tree for %s: %s", clade, candidates[0])

    return candidates[0]


def score_alignment_candidate(path: Path, clade: str) -> Tuple[int, str]:
    name = path.name.lower()
    clade_norm = normalise_name(clade)
    stem_norm = normalise_name(path.stem)

    bad_terms = ("query", "merged", "iqtree", "consensus", "bootstrap", "ufboot")
    bad_penalty = 10 if any(term in name for term in bad_terms) else 0
    clade_bonus = 0 if clade_norm and clade_norm in stem_norm else 1
    ext_rank = {
        ".fa": 0,
        ".fasta": 1,
        ".fas": 2,
        ".aln": 3,
        ".fna": 4,
    }.get(path.suffix.lower(), 9)

    return (bad_penalty + clade_bonus + ext_rank, str(path))


def find_reference_alignment(
    clade_dir: Path,
    clade: str,
    ref_alignment: Optional[Path] = None,
) -> Path:
    if ref_alignment is not None:
        resolved = ref_alignment if ref_alignment.is_absolute() else clade_dir / ref_alignment
        if not resolved.exists():
            raise FileNotFoundError(f"Reference alignment not found: {resolved}")
        return resolved

    candidates: List[Path] = []
    for ext in FASTA_EXTENSIONS:
        candidates.extend(clade_dir.glob(f"*{ext}"))

    if not candidates:
        for ext in FASTA_EXTENSIONS:
            candidates.extend(clade_dir.rglob(f"*{ext}"))

    if not candidates:
        raise FileNotFoundError(
            f"No reference alignment found in {clade_dir}. "
            f"Expected one of: {', '.join(FASTA_EXTENSIONS)}"
        )

    candidates = sorted(set(candidates), key=lambda path: score_alignment_candidate(path, clade))
    if len(candidates) > 1:
        logging.info(
            "Reference alignment candidates for %s: %s",
            clade,
            ", ".join(str(p) for p in candidates),
        )
        logging.info("Selected reference alignment for %s: %s", clade, candidates[0])

    return candidates[0]


def validate_alignment_lengths(records: Sequence[Tuple[str, str]], path_for_error: Path) -> int:
    lengths: "OrderedDict[int, int]" = OrderedDict()
    for _, sequence in records:
        lengths[len(sequence)] = lengths.get(len(sequence), 0) + 1

    if len(lengths) != 1:
        details = ", ".join(f"{length} bp: {count} seqs" for length, count in lengths.items())
        raise ValueError(f"Alignment has unequal sequence lengths near {path_for_error}: {details}")

    return next(iter(lengths))


def build_merged_alignment(
    reference_alignment_path: Path,
    query_records: Sequence[Tuple[str, str]],
    merged_alignment_path: Path,
) -> Tuple[int, int]:
    reference_records_map = read_fasta(reference_alignment_path)
    reference_records = list(reference_records_map.values())

    ref_ids = set(reference_records_map.keys())
    query_ids = {header.split()[0] for header, _ in query_records}
    overlap = sorted(ref_ids & query_ids)
    if overlap:
        raise ValueError(
            f"Query ID(s) already exist in reference alignment {reference_alignment_path}: "
            f"{', '.join(overlap[:10])}"
        )

    merged_records = reference_records + list(query_records)
    validate_alignment_lengths(merged_records, merged_alignment_path)
    write_fasta(merged_records, merged_alignment_path)

    return len(reference_records), len(query_records)


def build_iqtree_command(
    iqtree_bin: str,
    alignment_path: Path,
    tree_path: Path,
    prefix: Path,
    model: str,
    threads: str,
    tree_mode: str,
    bootstrap_method: str,
    bootstrap_replicates: int,
    extra_args: Optional[str],
    redo: bool,
) -> List[str]:
    command = [
        iqtree_bin,
        "-s",
        str(alignment_path),
        "-m",
        model,
        "-nt",
        str(threads),
        "-pre",
        str(prefix),
    ]

    if tree_mode == "constraint":
        command.extend(["-g", str(tree_path)])
    elif tree_mode == "starting":
        command.extend(["-t", str(tree_path)])
    elif tree_mode == "fixed":
        command.extend(["-te", str(tree_path)])
    else:
        raise ValueError(f"Unsupported tree mode: {tree_mode}")

    if bootstrap_replicates > 0:
        if bootstrap_method == "ufboot":
            command.extend(["-bb", str(bootstrap_replicates)])
        elif bootstrap_method == "bootstrap":
            command.extend(["-b", str(bootstrap_replicates)])
        elif bootstrap_method == "none":
            pass
        else:
            raise ValueError(f"Unsupported bootstrap method: {bootstrap_method}")

    if redo:
        command.append("-redo")

    if extra_args:
        command.extend(shlex.split(extra_args))

    return command


def build_treetime_ancestral_command(
    treetime_bin: str,
    alignment_path: Path,
    tree_path: Path,
    outdir: Path,
    extra_args: Optional[str] = None,
) -> List[str]:
    command = [
        treetime_bin,
        "ancestral",
        "--aln",
        str(alignment_path),
        "--tree",
        str(tree_path),
        "--outdir",
        str(outdir),
    ]

    if extra_args:
        command.extend(shlex.split(extra_args))

    return command

def run_command(command: Sequence[str], dry_run: bool = False) -> None:
    logging.info("Running: %s", " ".join(shlex.quote(str(part)) for part in command))
    if dry_run:
        return

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"IQ-TREE failed with exit code {completed.returncode}")


def write_summary(summary_path: Path, rows: Sequence[Dict[str, str]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "clade",
        "query_count",
        "reference_count",
        "clade_dir",
        "reference_alignment",
        "tree_file",
        "query_alignment",
        "merged_alignment",
        "iqtree_prefix",
        "tree_mode",
        "treetime_outdir",
        "treetime_tree",
        "treetime_ancestral_sequences",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clade-wise IQ-TREE using assignment TSV and pre-existing clade trees."
    )

    parser.add_argument(
        "--assignment",
        required=True,
        type=Path,
        help="Clade assignment TSV with sequence_id and EPA_major_clade columns.",
    )
    parser.add_argument(
        "--input-alignment",
        required=True,
        type=Path,
        help="Aligned FASTA containing query sequences, e.g. input_seqs_with_ref_alignment.fa.",
    )
    parser.add_argument(
        "--clade-tree-dir",
        required=True,
        type=Path,
        help="Directory containing clade subdirectories, e.g. clade_based_trees.",
    )
    parser.add_argument(
        "--base-dir",
        default=Path("tmp"),
        type=Path,
        help="Base output directory. Default: tmp",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("tree"),
        type=Path,
        help="Output directory relative to --base-dir unless absolute. Default: tree",
    )

    parser.add_argument(
        "--sequence-column",
        default="sequence_id",
        help="Assignment TSV sequence column. Default: sequence_id",
    )
    parser.add_argument(
        "--clade-column",
        default="EPA_major_clade",
        help="Assignment TSV clade column. Default: EPA_major_clade",
    )

    parser.add_argument(
        "--reference-alignment",
        type=Path,
        default=None,
        help=(
            "Optional reference alignment filename/path for every clade directory. "
            "If omitted, the script auto-detects a FASTA inside each clade directory."
        ),
    )
    parser.add_argument(
        "--tree-file",
        type=Path,
        default=None,
        help=(
            "Optional tree filename/path for every clade directory. "
            "If omitted, the script auto-detects .treefile/.nwk/.newick/.tre/.tree."
        ),
    )

    parser.add_argument(
        "--tree-mode",
        choices=("constraint", "starting", "fixed"),
        default="constraint",
        help=(
            "How to use existing tree. constraint = -g, starting = -t, fixed = -te. "
            "Use constraint or starting when adding query taxa. Default: constraint"
        ),
    )
    parser.add_argument(
        "--model",
        default="GTR+G",
        help="IQ-TREE model. Default: GTR+G",
    )
    parser.add_argument(
        "--threads",
        default="AUTO",
        help="Threads passed to IQ-TREE -nt. Default: AUTO",
    )
    parser.add_argument(
        "--iqtree-bin",
        default="iqtree2",
        help="IQ-TREE executable. Default: iqtree2",
    )
    parser.add_argument(
        "--bootstrap-method",
        choices=("none", "ufboot", "bootstrap"),
        default="none",
        help="Bootstrap method. ufboot = -bb, bootstrap = -b. Default: none",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=0,
        help="Bootstrap replicates. Default: 0",
    )
    parser.add_argument(
        "--extra-iqtree-args",
        default=None,
        help='Extra IQ-TREE arguments as a quoted string, e.g. "--alrt 1000 -safe".',
    )

    parser.add_argument(
        "--skip-treetime",
        action="store_true",
        help="Skip TreeTime ancestral reconstruction step.",
    )
    parser.add_argument(
        "--treetime-bin",
        default="treetime",
        help="TreeTime executable. Default: treetime",
    )
    parser.add_argument(
        "--extra-treetime-args",
        default=None,
        help='Extra TreeTime arguments as a quoted string, e.g. "--gtr infer".',
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Pass -redo to IQ-TREE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write files and print IQ-TREE commands but do not execute IQ-TREE.",
    )

    parser.add_argument(
        "--clade-dir-alias",
        action="append",
        default=[],
        help=(
            "Map assignment clade name to directory name. Can be repeated. "
            "Example: --clade-dir-alias Indian-Sub=India-Sub"
        ),
    )
    parser.add_argument(
        "--strict-clade-dir",
        action="store_true",
        help="Disable fuzzy clade directory matching.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args()



def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if args.bootstrap_method != "none" and args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be > 0 when --bootstrap-method is not none")

    if args.tree_mode == "fixed":
        logging.warning(
            "--tree-mode fixed uses -te. This is usually NOT suitable for adding new query taxa, "
            "because the tree normally must already contain all taxa in the alignment."
        )

    output_root = args.output_dir if args.output_dir.is_absolute() else args.base_dir / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    aliases = parse_aliases(args.clade_dir_alias)
    groups = read_assignment(args.assignment, args.clade_column, args.sequence_column)
    input_records = read_fasta(args.input_alignment)

    logging.info("Found %d clade group(s): %s", len(groups), ", ".join(groups.keys()))
    logging.info("Read %d sequences from %s", len(input_records), args.input_alignment)

    summary_rows: List[Dict[str, str]] = []

    for clade, query_ids in groups.items():
        clade_safe = safe_name(clade)
        clade_out_dir = output_root / clade_safe
        clade_out_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Processing clade '%s' with %d assigned query sequence(s)", clade, len(query_ids))

        query_records = extract_records(
            input_records,
            query_ids,
            source_description=str(args.input_alignment),
        )

        query_alignment_path = clade_out_dir / f"{clade_safe}.query.aligned.fa"
        write_fasta(query_records, query_alignment_path)

        clade_dir = find_clade_directory(
            clade,
            args.clade_tree_dir,
            aliases=aliases,
            strict=args.strict_clade_dir,
        )
        reference_alignment_path = find_reference_alignment(
            clade_dir,
            clade,
            ref_alignment=args.reference_alignment,
        )
        tree_path = find_tree_file(clade_dir, clade, tree_file=args.tree_file)

        merged_alignment_path = clade_out_dir / f"{clade_safe}.merged.aligned.fa"
        reference_count, query_count = build_merged_alignment(
            reference_alignment_path,
            query_records,
            merged_alignment_path,
        )

        iqtree_prefix = clade_out_dir / f"{clade_safe}.iqtree"
        command = build_iqtree_command(
            iqtree_bin=args.iqtree_bin,
            alignment_path=merged_alignment_path,
            tree_path=tree_path,
            prefix=iqtree_prefix,
            model=args.model,
            threads=args.threads,
            tree_mode=args.tree_mode,
            bootstrap_method=args.bootstrap_method,
            bootstrap_replicates=args.bootstrap_replicates,
            extra_args=args.extra_iqtree_args,
            redo=args.redo,
        )

        run_command(command, dry_run=args.dry_run)
        iqtree_treefile = Path(str(iqtree_prefix) + ".treefile")
        treetime_outdir = clade_out_dir / f"{clade_safe}.treetime_ancestral"

        treetime_tree = treetime_outdir / "annotated_tree.nexus"
        treetime_ancestral_sequences = treetime_outdir / "ancestral_sequences.fasta"

        if not args.skip_treetime:
            if not args.dry_run and not iqtree_treefile.exists():
                raise FileNotFoundError(
                    f"Expected IQ-TREE output tree not found: {iqtree_treefile}"
                )

            treetime_command = build_treetime_ancestral_command(
                treetime_bin=args.treetime_bin,
                alignment_path=merged_alignment_path,
                tree_path=iqtree_treefile,
                outdir=treetime_outdir,
                extra_args=args.extra_treetime_args,
            )

            run_command(treetime_command, dry_run=args.dry_run)

        summary_rows.append(
            {
                "clade": clade,
                "query_count": str(query_count),
                "reference_count": str(reference_count),
                "clade_dir": str(clade_dir),
                "reference_alignment": str(reference_alignment_path),
                "tree_file": str(tree_path),
                "query_alignment": str(query_alignment_path),
                "merged_alignment": str(merged_alignment_path),
                "iqtree_prefix": str(iqtree_prefix),
                "tree_mode": args.tree_mode,
                "treetime_outdir": str(treetime_outdir),
                "treetime_tree": str(treetime_tree),
                "treetime_ancestral_sequences": str(treetime_ancestral_sequences),

            }
        )

    summary_path = output_root / "phylogeny_summary.tsv"
    write_summary(summary_path, summary_rows)

    logging.info("Wrote summary: %s", summary_path)
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.error(str(exc))
        raise SystemExit(1)
