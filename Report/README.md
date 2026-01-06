# Project Report - Build Instructions

## Required Files

To compile the LaTeX report, you need the following files in the `Report/` directory:

1. `project_report.tex` - Main report file (provided)
2. `references.bib` - Bibliography file (provided)
3. `siamart171218.cls` - SIAM article class file (download required)
4. `siamplain.bst` - SIAM bibliography style (download required)

## Download SIAM Style Files

Download the SIAM LaTeX macro package from:
https://www.siam.org/publications/journals/about-siam-journals/information-for-authors

Or directly from the SIAM FTP:
- `siamart171218.cls`: The main document class
- `siamplain.bst`: The bibliography style

Place these files in the same directory as `project_report.tex`.

## Compilation

To compile the document, run the following commands in order:

```bash
pdflatex project_report
bibtex project_report
pdflatex project_report
pdflatex project_report
```

Or use a LaTeX editor like Overleaf, TeXstudio, or VS Code with LaTeX Workshop.

## Overleaf (Recommended)

For easiest compilation, upload all files to Overleaf.com:
1. Create a new project
2. Upload `project_report.tex`, `references.bib`
3. Search for "SIAM" in the templates or upload the SIAM style files
4. Set the main document to `project_report.tex`
5. Compile

## Expected Output

The compiled PDF should be approximately 8-10 pages and include:
- Abstract
- Introduction
- Research Question and Literature Review
- Methodology (HRP + Regime Detection)
- Data Description
- Implementation
- Results
- Conclusion
- Appendix (AI Tools)
- References

## Figures

If you want to include figures from the `Super_Agent_Output/` directory:
1. Copy the PNG files to the `Report/` directory
2. Use `\includegraphics{filename}` in the LaTeX file
3. Ensure you have the `graphicx` package loaded (already included)
