from pylatex import Document, Package, NoEscape
import numpy as np

# Constants
OUTPUT_PATH = "output.pdf"
PAGE_WIDTH = 246.2
PAGE_HEIGHT = 159.2
MM_TO_PT = 2.835
HEADER_MIN_HEIGHT = 250
LINE_TOLERANCE = 50

# Global variables
doc = Document()
firstPageAdded = False

def get_style(box):
    if(box[1] - box[3] > HEADER_MIN_HEIGHT):
        return 'H'
    else:
        return 'N'
    

# Removes slight differences in y-values of characters, to preserve alignment of characters on the PDF
# pdfData: Each box is in the form [label, maxY, maxX, minY, minX, size]
# y_tolerance: The maximum difference in y-values for characters to be considered "on the same line"
def realign_chars(pdfData, y_tolerance):

    # Sort the chars by maxY in ascending order
    pdfData = pdfData[np.argsort(pdfData[:, 1])]

    # Separate the chars into lines
    lines = []
    prev_char = pdfData[0]
    curr_line = [pdfData[0]]
    for i in range(1, len(pdfData)):
        if abs(pdfData[i, 1] - prev_char[1]) < y_tolerance:
            curr_line.append(pdfData[i])
        else:
            lines.append(curr_line)
            curr_line = [pdfData[i]]
        prev_char = pdfData[i]
    lines.append(curr_line) # Add the last line

    # Set the maxY of each character in the line to the average
    for line in lines:
        y_avg = np.mean([char[1] for char in line])
        for char in line:
            char[1] = y_avg

    return pdfData


# Performs an initial setup for the LaTeX PDF, including the title and other settings
def setup_pdf():
    global doc
    doc = Document(documentclass='article', geometry_options={'a4paper', 'landscape', 'margin=1in'})

    # Use the TikZ package to place characters in specific coordinates
    doc.packages.append(Package("tikz"))
    doc.preamble.append(NoEscape(r"\pagenumbering{gobble}"))
    # doc.preamble.append(NoEscape(r'\renewcommand{\familydefault}{\ttdefault}'))
    return doc


# Add characters onto a new page of the PDF by their coordinates
# Below is the legend of styles:
# 'H': Section header. Large, bold font.
# 'N': Normal text.
def add_page(pdfData, Bwidth, Bheight):
    global doc
    global firstPageAdded
    pdfData = realign_chars(pdfData, LINE_TOLERANCE)

    # Make a new page if necessary, then start a new TikZ picture on that page
    if not firstPageAdded:
        firstPageAdded = True
    else:
        doc.append(NoEscape(r"\newpage"))
    doc.append(NoEscape(r"\begin{tikzpicture}[scale=0.04, yscale = -1]"))

    #data[0] is the label, data[1], data[2] is the maxY and maxX, data[3] and data[4] is the minY and minX, data[5] is size
    for data in pdfData:
        style = get_style(data)
        char =' '
        if(data[0] > 9):
            char = chr(data[0] - 9 + 64)
        else:
            char = chr(data[0] + 48)
        
        if style == 'H':
            doc.append(NoEscape(fr"\node[font=\bfseries\huge] at ({data[4] / Bwidth * PAGE_WIDTH},{data[1] / Bheight * PAGE_HEIGHT}) {{{char}}};"))
        elif style == 'N':
            doc.append(NoEscape(fr"\node[font=\Large] at ({data[4] / Bwidth * PAGE_WIDTH},{data[1] / Bheight * PAGE_HEIGHT}) {{{char}}};"))

    doc.append(NoEscape(r"\end{tikzpicture}"))


def render_pdf():
    global doc
    doc.generate_pdf("output/output", clean_tex=False, compiler="pdfLaTeX")
    # doc.generate_pdf("output", clean_tex=False, compiler="latexmk", compiler_args=['-c'])


# if __name__ == "__main__":

#     # Test data
#     characters = ['O', 'C', 'R', 'P', 'R', 'O', 'J', 'E', 'C', 'T',
#                   'H', 'E', 'R', 'E', 'I', 'S', 'S', 'O', 'M', 'E', 'T', 'E', 'X', 'T']
#     coordinates = [(1, 1), (1.8, 1), (2.6, 1), (4.2, 1), (5, 1), (5.8, 1), (6.6, 1), (7.4, 1), (8.2, 1), (9, 1), 
#                    (1, 2.5), (1.5, 2.5), (2, 2.5), (2.5, 2.5), (3.5, 2.5), (4, 2.5), (5, 2.5), (5.5, 2.5), (6, 2.5), (6.5, 2.5), (7.5, 2.5), (8, 2.5), (8.5, 2.5), (9, 2.5)]
#     styles = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
#               'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
#     render_pdf(characters, coordinates, styles)
