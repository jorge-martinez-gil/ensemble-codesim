# Growth Playbook: turning this repository into a citation magnet

This is a concrete, ordered plan for making `ensemble-codesim` a work that
other researchers find, use, and cite. The theme running through it is simple:
citations follow from three things at once, usefulness, reproducibility, and
discoverability. Each step below moves at least one of them.

## 1. Mint a citable software DOI on Zenodo

A paper has a DOI; the code should have one too, so people can cite the software
release itself. This is the single highest-impact step.

1. Sign in to [Zenodo](https://zenodo.org) with your GitHub account.
2. Open Settings, then GitHub, and flip the switch next to
   `jorge-martinez-gil/ensemble-codesim` to ON.
3. Back on GitHub, draft a release (for instance tag `v0.1.0`). Zenodo archives
   that release and issues a DOI within a minute or two.
4. Zenodo reads the `.zenodo.json` in this repo for the title, authors, license,
   and keywords, so the archive is described correctly from the first release.
5. Copy the "concept DOI" (the one that always points to the newest version) and
   add its badge to the top of the README.

Every future release gets a fresh version DOI automatically, and the concept DOI
keeps resolving to the latest.

## 2. Turn on the GitHub "Cite this repository" button

The `CITATION.cff` in this repo activates a Cite button on the repository home
page and lets reference managers import the citation in one click. Confirm it
renders (GitHub shows a "Cite this repository" link on the right sidebar once the
file is present). Keep the version and `date-released` fields current with each
release, and add your ORCID where the file notes it.

## 3. Keep the preprint fresh and linked both ways

The arXiv preprint (2405.02095) and the SWQD 2025 paper are the anchor
citations. Make the link bidirectional: the paper and preprint should point to
this repository, and the repository should point to them (the README and the
`CITATION.cff` already do). When the survey draft in `paper/` is ready, post it
to arXiv as well and cross-link it here, since a survey attracts a different and
often larger citing audience than a method paper.

## 4. Make the code trivially runnable, then say so

Reviewers and readers cite what they can run in one minute.

- The `CodeSim` package installs with `pip install -e .` and its tests pass in
  continuous integration (see the tests badge). Keep both green.
- Publish the package on PyPI so anyone can `pip install codesim`. A tool that
  people install is a tool they cite. Steps: `pip install build twine`, then
  `python -m build`, then `twine upload dist/*` from an account you control.
- Keep the quickstart in `examples/` and the notebooks in `tutorials/` working
  against the current code, so the first thing a visitor runs succeeds.

## 5. Optimize discoverability on GitHub

- Set the repository "About" text to one clear sentence with the words people
  search: "source code clone detection", "code similarity", "ensemble".
- Add GitHub Topics: `code-clone-detection`, `source-code-similarity`,
  `clone-detection`, `software-engineering`, `bigclonebench`, `plagiarism-detection`,
  `ensemble-learning`, `python`. Topics feed GitHub search and the topic pages.
- Pin the repository on your GitHub profile.

## 6. Get listed where practitioners look

- Submit the method and results to [Papers with Code](https://paperswithcode.com)
  and attach this repository, linking the BigCloneBench task.
- Propose the repo to relevant curated lists (for instance an "awesome code
  clone detection" or "awesome software engineering" list) through a pull
  request.
- Add the repository link to your Google Scholar profile and your institutional
  page.

## 7. Lower the friction to cite

- Keep a short, copy-ready BibTeX block near the top of the README (done).
- Repeat a one-line "How to cite" at the end of each tutorial notebook, so a
  student who learns from the notebook knows how to credit it.
- When someone opens an issue describing their use, thank them and point to the
  citation. Users often become citers when reminded once.

## 8. Signal that the project is alive

Maintenance is a trust signal, and trust drives adoption.

- Cut tagged releases with short changelogs.
- Answer issues and pull requests promptly.
- Add a `CONTRIBUTING.md` with a two-minute setup so outside contributions are
  easy. Contributors cite what they help build.

## 9. Publish supporting content

- A short blog post or a thread that walks through one figure from the study
  (for instance, why a naive average of all measures loses to a learned
  ensemble) draws readers who then find the paper.
- A five-minute screencast of the toolkit running on a real clone pair gives
  talks and courses something to reuse, and course use produces citations.

## 10. Track and iterate

- Set a Google Scholar alert for the paper title and for "BigCloneBench
  ensemble" so you see new citing work early.
- Add a citations badge to the README and watch it move.
- When a new paper cites this work, read what they needed and file it as a
  feature idea. The features that citers ask for are the ones that attract the
  next round of citers.

## A realistic note

No single trick manufactures citations. What compounds is a useful tool that
runs the first time, a paper and a survey that frame the area clearly, and a
repository that is easy to find and easy to credit. The steps above put all
three in place; the rest is steady upkeep.
