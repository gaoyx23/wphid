# Third-party attribution

The project license covers first-party code and associated documentation. It does not override licenses of dependencies or external sources. No publisher article, external figure, original ADAM notebook or WENDy source is newly copied into the tracked repository candidate.

- NumPy and SciPy: installed scientific dependencies; BSD-style licenses remain with their distributions.
- PyTorch: dependency used for the transparent conditional Adam adapter; its existing license remains with the distribution.
- Matplotlib, Python-docx, ReportLab, Pillow, lxml, latex2mathml and PyMuPDF: manuscript/plotting tools remain dependencies with their own distribution licenses; they are not vendored as project code.
- ADAM-SINDy: Siva Viknesh, Younes Tatari, Chase Christenson and Amirhossein Arzani. Public method/code review uses the source commit and file identities recorded in `reproducibility/final_candidate_v1/official-source-lock.json`. The project's conditional adapters are independently implemented and differ from native discovery; the source lock explicitly records that reviewed external code was not copied or executed.
- WENDy: Nic Rummel, Daniel A. Messenger, Stephen Becker, Vanja Dukic and David M. Bortz. The Gaussian comparator is a first-order PDE quasi-likelihood implementation, not official WENDy source or runtime results.
- NASA, ESA and DLR sources motivate physical assumptions and are cited in the archived source catalogue. Synthetic constitutive/kinetic settings are not represented as experimentally calibrated by these sources. External documents are linked rather than reissued under MIT.

The preserved dependency archives retain their original source identities and notices. This notice records source inspection and attribution; it is not a relicensing of third-party works.
