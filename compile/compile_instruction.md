After any script modifications do the following to re-build the package

python -m pip install build

python -m build

output should looklike
dist/
├── maddog-0.1.0-py3-none-any.whl
└── maddog-0.1.0.tar.gz

install the new package

pip install dist/maddog-0.1.0-py3-none-any.whl

testing if all passes 

maddog-assignment --help
maddog-phylogeny --help
maddog-reroot --help
maddog-assign-nodes --help
maddog-lineage-assignment --help
maddog-reduce-lineages --help


