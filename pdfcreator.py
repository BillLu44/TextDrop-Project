from decimal import Decimal

from borb.pdf import Alignment
from borb.pdf import Document
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import Paragraph
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.page.page_size import PageSize


def renderPDF(pdfData, Twidth, Theight):
    # create Document
    doc: Document = Document()

    # create Page
    page: Page = Page(
        width=PageSize.A4_LANDSCAPE.value[0], height=PageSize.A4_LANDSCAPE.value[1]
    )

    pageWidth = PageSize.A4_LANDSCAPE.value[0]
    pageHeight = PageSize.A4_LANDSCAPE.value[1]

    # add Page to Document
    doc.add_page(page)

    # define layout rectangle
    # fmt: off
    for data in pdfData:
        r: Rectangle = Rectangle(
            Decimal(data[4] * pageWidth / Twidth),                # x: 0 + page_margin
            Decimal(pageHeight - data[1] * pageHeight / Theight),    # y: page_height - page_margin - height_of_textbox
            Decimal((data[2] - data[4]) * pageWidth / Twidth),      # width: page_width - 2 * page_margin
            Decimal((data[1] - data[3]) * pageHeight / Theight),               # height
        )
        # fmt: o
        # the next line of code uses absolute positioning
        if data[0] > 9:
            Paragraph(chr(data[0] - 9 + 64), vertical_alignment=Alignment.TOP, font="Courier").paint(page, r)
        else:
            Paragraph(chr(data[0] + 48), vertical_alignment=Alignment.TOP, font="Courier").paint(page, r)

    # store
    with open("output.pdf", "wb") as pdf_file_handle:
        PDF.dumps(pdf_file_handle, doc)


# if __name__ == "__main__":
#     test()