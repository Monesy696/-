import csv
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas


SOURCE_PDF = Path("A题.pdf")
OUTPUT_PDF = Path("output/pdf/A题_表1至表6已填.pdf")
TABLE_DIR = Path("outputs/tables")

TABLE_SPECS = {
    1: {
        "page": 0,
        "csv": "table_01_q1_temperature.csv",
        "row_centers_top": [457.03, 473.11, 489.19, 505.27, 521.41, 537.55, 553.63],
    },
    2: {
        "page": 0,
        "csv": "table_02_q1_moisture.csv",
        "row_centers_top": [633.64, 649.78, 665.86, 681.94, 698.02, 714.10, 730.24],
    },
    3: {
        "page": 1,
        "csv": "table_03_q2_temperature.csv",
        "row_centers_top": [221.42, 237.56, 253.70, 269.80, 285.89, 301.97],
    },
    4: {
        "page": 1,
        "csv": "table_04_q2_moisture.csv",
        "row_centers_top": [382.01, 398.09, 414.17, 430.25, 446.40, 462.55],
    },
    5: {
        "page": 1,
        "csv": "table_05_q3_moisture.csv",
        "row_centers_top": [620.73, 636.88, 653.02, 669.10, 685.18],
        "row_indices": [0, 1, 2, None, 9],
    },
    6: {
        "page": 2,
        "csv": "table_06_q4_moisture.csv",
        "row_centers_top": [174.86, 191.00, 207.14, 223.22, 239.30],
        "row_indices": [0, 1, 2, None, 8],
        "column_indices": [0, 1, 5],
        "column_centers": [231.11, 310.98, 470.85],
    },
}

COLUMN_CENTERS = [223.13, 287.04, 350.95, 414.86, 478.84]
FONT_NAME = "Times-Roman"
FONT_SIZE = 8.2


def read_table(filename):
    path = TABLE_DIR / filename
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows[0]) < 6:
        raise ValueError(f"{path} 至少应包含时间列和5个空间位置列")
    return [
        [None if value == "" else float(value) for value in row[1:]]
        for row in rows[1:]
    ]


def create_overlay(page_width, page_height, tables):
    stream = BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(page_width, page_height))
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont(FONT_NAME, FONT_SIZE)

    for table in tables:
        values = read_table(table["csv"])
        row_centers = table["row_centers_top"]
        row_indices = table.get("row_indices", list(range(len(values))))
        column_indices = table.get("column_indices", list(range(len(values[0]))))
        column_centers = table.get("column_centers", COLUMN_CENTERS)
        if len(row_indices) != len(row_centers):
            raise ValueError(f"{table['csv']} 的行映射与题面表格不一致")
        if len(column_indices) != len(column_centers):
            raise ValueError(f"{table['csv']} 的列映射与题面表格不一致")
        for row_index, row_center_top in zip(row_indices, row_centers):
            if row_index is None:
                continue
            row_values = values[row_index]
            baseline_y = page_height - row_center_top - 0.34 * FONT_SIZE
            for column_index, center_x in zip(column_indices, column_centers):
                value = row_values[column_index]
                if value is None:
                    continue
                pdf.drawCentredString(center_x, baseline_y, f"{value:.4f}")

    pdf.save()
    stream.seek(0)
    return PdfReader(stream).pages[0]


def main():
    source = PdfReader(SOURCE_PDF)
    writer = PdfWriter()

    for page_index, source_page in enumerate(source.pages):
        page_tables = [
            spec for spec in TABLE_SPECS.values() if spec["page"] == page_index
        ]
        if page_tables:
            width = float(source_page.mediabox.width)
            height = float(source_page.mediabox.height)
            overlay = create_overlay(width, height, page_tables)
            source_page.merge_page(overlay)
        writer.add_page(source_page)

    if source.metadata:
        writer.add_metadata(
            {key: str(value) for key, value in source.metadata.items() if value is not None}
        )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PDF.open("wb") as file:
        writer.write(file)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
