from pylatex import Document, Package, NoEscape
import numpy as np

# Constants
OUTPUT_PATH = "output.pdf"
PAGE_WIDTH = 246.2
PAGE_HEIGHT = 159.2
MM_TO_PT = 2.835

def get_style(box):
    if(box[1] - box[3] > 100):
        return 'H'
    else:
        return 'N'

# Add characters to the PDF by their coordinates
# Below is the legend of styles:
# 'H': Section header. Large, bold font.
# 'N': Normal text.
def add_content(doc, pdfData, Bwidth, Bheight):
    # for char, (x, y), style in zip(chars, coords, styles):
    #     if style == 'H':
    #         doc.append(NoEscape(fr"\node[font=\bfseries\Large] at ({x},{y}) {{{char}}};"))
    #     elif style == 'N':
    #         doc.append(NoEscape(fr"\node at ({x},{y}) {{{char}}};"))
    #data[0] is the label, data[1], data[2] is the maxY and maxX, data[3] and data[4] is the minY and minX, data[5] is size
    for data in pdfData:
        style = get_style(data)
        char =' '
        if(data[0] > 9):
            char = chr(data[0] - 9 + 64)
        else:
            char = chr(data[0] + 48)

        
        if style == 'H':
            doc.append(NoEscape(fr"\node[font=\bfseries\Large] at ({data[4] / Bwidth * PAGE_WIDTH},{data[3] / Bheight * PAGE_HEIGHT}) {{{char}}};"))
        elif style == 'N':
            doc.append(NoEscape(fr"\node at ({data[4] / Bwidth * PAGE_WIDTH},{data[3] / Bheight * PAGE_HEIGHT}) {{{char}}};"))




# Performs an initial setup for the LaTeX PDF, including the title and other settings
def setup_doc(title):
    doc = Document(documentclass='article')

    # Use the TikZ package to place characters in specific coordinates
    doc.packages.append(Package("tikz"))
    doc.packages.append(Package('geometry', options=['a4paper', 'landscape']))
    doc.preamble.append(NoEscape(r"\usepackage[margin=1in]{geometry}"))
    # doc.preamble.append(NoEscape(r'\renewcommand{\familydefault}{\ttdefault}'))
    doc.append(NoEscape(r"\begin{tikzpicture}[scale=0.1, yscale = -1]"))
    return doc


def render_pdf(pdfData, Bwidth, Bheight):
    doc = setup_doc("Auto-generated Test From Blackboard")
    add_content(doc, pdfData, Bwidth, Bheight)

    doc.append(NoEscape(r"\end{tikzpicture}"))
    doc.generate_pdf("test.pdf", clean_tex=False, compiler="pdfLaTeX")


# if __name__ == "__main__":

#     # Test data
#     characters = ['O', 'C', 'R', 'P', 'R', 'O', 'J', 'E', 'C', 'T',
#                   'H', 'E', 'R', 'E', 'I', 'S', 'S', 'O', 'M', 'E', 'T', 'E', 'X', 'T']
#     coordinates = [(1, 1), (1.8, 1), (2.6, 1), (4.2, 1), (5, 1), (5.8, 1), (6.6, 1), (7.4, 1), (8.2, 1), (9, 1), 
#                    (1, 2.5), (1.5, 2.5), (2, 2.5), (2.5, 2.5), (3.5, 2.5), (4, 2.5), (5, 2.5), (5.5, 2.5), (6, 2.5), (6.5, 2.5), (7.5, 2.5), (8, 2.5), (8.5, 2.5), (9, 2.5)]
#     styles = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
#               'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
#     render_pdf(characters, coordinates, styles)