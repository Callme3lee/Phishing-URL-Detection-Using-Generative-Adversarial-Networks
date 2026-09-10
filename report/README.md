# New Graduation Report

This is the reference-based graduation report source. It currently contains
the University front pages and Chapters 1--9, including the experiments,
browser extension, user-evaluation findings, conclusion, and future work.

Build the PDF from this directory with:

```bash
make
```

The main content files are:

- `metadata.tex` -- project and student details;
- `frontmatter/` -- title, declarations, abstract, and acknowledgement;
- `chapters/` -- report chapters;
- `assets/figures/` and `assets/graphs/` -- report visuals;
- `supporting/generate_report_graphs.py` -- reproducible graph generation;
- `references.bib` -- cited sources.
