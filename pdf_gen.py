from pylatex import Document, Section, Package, NoEscape

# Constants
OUTPUT_PATH = "output.pdf"

# Add characters to the PDF by their coordinates
# Below is the legend of styles:
# 'H': Section header. Large, bold font.
# 'N': Normal text.
def add_content(doc, chars, coords, styles):
    for char, (x, y), style in zip(chars, coords, styles):
        if style == 'H':
            doc.append(NoEscape(fr"\node[font=\bfseries\Large] at ({x},{y}) {{{char}}};"))
        elif style == 'N':
            doc.append(NoEscape(fr"\node at ({x},{y}) {{{char}}};"))


# Performs an initial setup for the LaTeX PDF, including the title and other settings
def setup_doc(title):
    doc = Document(documentclass='article')

    # Use the TikZ package to place characters in specific coordinates
    doc.packages.append(Package("tikz"))
    doc.preamble.append(NoEscape(r"\usepackage[margin=1in]{geometry}"))
    # doc.preamble.append(NoEscape(r'\renewcommand{\familydefault}{\ttdefault}'))
    doc.append(NoEscape(r"\begin{tikzpicture}[scale=0.5]"))

    return doc


def render_pdf(characters, coordinates, styles):
    doc = setup_doc("Auto-generated Test From Blackboard")
    add_content(doc, characters, coordinates, styles)

    doc.append(NoEscape(r"\end{tikzpicture}"))
    doc.generate_pdf("test.pdf", clean_tex=False, compiler="pdfLaTeX")


if __name__ == "__main__":

    # Test data
    characters = ['O', 'C', 'R', 'P', 'R', 'O', 'J', 'E', 'C', 'T',
                  'H', 'E', 'R', 'E', 'I', 'S', 'S', 'O', 'M', 'E', 'T', 'E', 'X', 'T']
    coordinates = [(1, 2.5), (1.8, 2.5), (2.6, 2.5), (4.2, 2.5), (5, 2.5), (5.8, 2.5), (6.6, 2.5), (7.4, 2.5), (8.2, 2.5), (9, 2.5), 
                   (1, 1), (1.5, 1), (2, 1), (2.5, 1), (3.5, 1), (4, 1), (5, 1), (5.5, 1), (6, 1), (6.5, 1), (7.5, 1), (8, 1), (8.5, 1), (9, 1)]
    styles = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
              'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
    render_pdf(characters, coordinates, styles)
