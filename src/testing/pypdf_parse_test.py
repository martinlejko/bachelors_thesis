from pypdf import PdfReader

reader_less = PdfReader("/Users/martinlejko/desktop/tabulka1.pdf")
reader_more = PdfReader("/Users/martinlejko/desktop/tabulka2.pdf")

with open('/Users/martinlejko/Repos/github.com/martinlejko/bachelors_thesis/src/pypdf_tabulka1.md', 'w') as f:
    f.write(reader_less.pages[0].extract_text())

with open('/Users/martinlejko/Repos/github.com/martinlejko/bachelors_thesis/src/pypdf_tabulka2.md', 'w') as f:
    f.write(reader_more.pages[0].extract_text())