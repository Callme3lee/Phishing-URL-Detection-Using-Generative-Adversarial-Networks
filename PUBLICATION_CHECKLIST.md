# Publication checklist

Complete these items before making the repository public.

- [ ] Choose a software licence with the project owner and supervisor. Do not assume that source availability automatically grants reuse rights.
- [ ] Decide whether the student ID on the report front matter should remain public.
- [ ] Confirm redistribution rights for every dataset or publish acquisition instructions instead.
- [ ] Deposit permitted datasets and selected final artifacts in Zenodo, Hugging Face, or approved institutional storage.
- [ ] Replace the `null` download URLs in `data/artifacts_manifest.json` with stable public links.
- [ ] Verify every external artifact using `python3 scripts/verify_artifacts.py`.
- [ ] Confirm that no participant browsing export, private hostname, email address, credential, query value, or identifier is included.
- [ ] Keep live phishing URLs out of screenshots and prose; use defanged forms where examples are necessary.
- [ ] Run `python3 scripts/check_repository.py` and resolve every reported error.
- [ ] Run the extension test suite with `cd extension && npm ci && npm test`.
- [ ] Create a tagged GitHub release containing the final report and install-only extension package.
- [ ] Add the final repository URL and archival DOI to `CITATION.cff` and the README.
