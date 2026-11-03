import pymupdf
from pathlib import Path
import re
import statistics



def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file page by page.

    Returns a list of dictionaries containing:
    - Page number
    - Extracted text
    """


    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )
    
    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")
    
    extracted_pages = []

    try:
        with pymupdf.open(file_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                page_text = page.get_text("text").strip()

                extracted_pages.append({
                    "page_number": page_number,
                    "text": page_text
                })

    except pymupdf.FileDataError as error:
        raise ValueError(
            f"Could not read PDF file: {error}"
        ) from error
    
    return extracted_pages




def extract_text_blocks_from_pdf(file_path):
    """
    Extract individual text blocks from every PDF page.

    Each block contains:
    - Page number
    - Block number
    - Extracted text
    - Bounding box coordinates (x0, y0, x1, y1)
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_blocks = []

    try:
        with pymupdf.open(file_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                blocks = page.get_text("blocks")

                for block_number, block in enumerate(blocks, start=1):

                    x0, y0, x1, y1, text, *_ = block

                    text = text.strip()

                    if not text:
                        continue

                    extracted_blocks.append({
                        "page_number": page_number,
                        "block_number": block_number,
                        "text": text,
                        "coordinates": {
                            "x0": x0,
                            "y0": y0,
                            "x1": x1,
                            "y1": y1
                        }
                    })

    except pymupdf.FileDataError as error:
        raise ValueError(
            f"Could not read PDF file: {error}"
        ) from error

    return extracted_blocks




def extract_text_with_formatting(file_path):
    """
    Extract text with font and layout information.

    Each text span contains:
    - Page number
    - Text
    - Font name
    - Font size
    - Bold/italic flags
    - Bounding box coordinates
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_spans = []

    try:
        with pymupdf.open(file_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                page_data = page.get_text("dict")

                for block in page_data["blocks"]:

                    if block.get("type") != 0:
                        continue

                    for line in block["lines"]:

                        for span in line["spans"]:

                            text = span["text"].strip()

                            if not text:
                                continue

                            font_name = span["font"]
                            font_size = span["size"]
                            font_flags = span["flags"]

                            extracted_spans.append({
                                "page_number": page_number,
                                "text": text,
                                "font_name": font_name,
                                "font_size": font_size,
                                "is_bold": bool(font_flags & 16),
                                "is_italic": bool(font_flags & 2),
                                "coordinates": {
                                    "x0": span["bbox"][0],
                                    "y0": span["bbox"][1],
                                    "x1": span["bbox"][2],
                                    "y1": span["bbox"][3]
                                }
                            })

    except pymupdf.FileDataError as error:
        raise ValueError(
            f"Could not read PDF file: {error}"
        ) from error

    return extracted_spans



def extract_headings_from_pdf(file_path):
    """
    Extract structural headings from a research paper.

    Detection rules:
    - Recognizes Abstract, Acknowledgements and References.
    - Recognizes numbered sections and subsections.
    - Recognizes short, bold section headings.
    - Ignores author details and affiliations before Abstract.
    - Ignores bibliography entries after References.
    - Avoids treating long algorithm instructions as headings.

    Returns a list containing:
    - Page number
    - Heading text
    - Font size
    - Bold status
    - Bounding box coordinates
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_headings = []
    lines = []

    try:
        with pymupdf.open(file_path) as pdf:

            # Step 1: Extract lines and formatting
            for page_number, page in enumerate(pdf, start=1):

                page_data = page.get_text("dict")

                for block in page_data["blocks"]:

                    if block.get("type") != 0:
                        continue

                    for line in block["lines"]:

                        spans = line["spans"]

                        text = "".join(
                            span["text"] for span in spans
                        ).strip()

                        if not text:
                            continue

                        font_sizes = [
                            span["size"]
                            for span in spans
                            if span["text"].strip()
                        ]

                        if not font_sizes:
                            continue

                        average_font_size = statistics.mean(
                            font_sizes
                        )

                        is_bold = any(
                            span["flags"] & 16
                            for span in spans
                            if span["text"].strip()
                        )

                        lines.append({
                            "page_number": page_number,
                            "text": text,
                            "font_size": average_font_size,
                            "is_bold": is_bold,
                            "coordinates": {
                                "x0": line["bbox"][0],
                                "y0": line["bbox"][1],
                                "x1": line["bbox"][2],
                                "y1": line["bbox"][3]
                            }
                        })

            if not lines:
                return extracted_headings

            # Step 2: Estimate body font size
            body_font_sizes = [
                line["font_size"]
                for line in lines
                if len(line["text"]) > 80
            ]

            if body_font_sizes:
                body_font_size = statistics.median(
                    body_font_sizes
                )
            else:
                body_font_size = statistics.median(
                    line["font_size"] for line in lines
                )

            # Step 3: Define heading patterns
            numbered_heading_pattern = re.compile(
                r"^\d+(?:\.\d+)*\.?\s+[A-Z]"
            )

            special_headings = {
                "abstract",
                "acknowledgements",
                "acknowledgments",
                "references",
                "bibliography",
            }

            # Step 4: Identify Abstract
            abstract_index = None

            for index, line in enumerate(lines):

                if line["text"].strip().lower() == "abstract":
                    abstract_index = index
                    break

            # If Abstract is not found, process the document
            # from its beginning.
            start_index = (
                abstract_index
                if abstract_index is not None
                else 0
            )

            # Step 5: Detect structural headings
            inside_references = False

            for line in lines[start_index:]:

                text = line["text"].strip()
                normalized_text = text.lower()

                # Stop processing after References
                if inside_references:
                    continue

                # Explicitly recognized headings
                if normalized_text in special_headings:

                    extracted_headings.append({
                        "page_number": line["page_number"],
                        "text": text,
                        "font_size": line["font_size"],
                        "is_bold": line["is_bold"],
                        "coordinates": line["coordinates"]
                    })

                    if normalized_text in {
                        "references",
                        "bibliography"
                    }:
                        inside_references = True

                    continue

                # Ignore long lines
                if len(text) > 100:
                    continue

                # Detect numbered sections and subsections
                is_numbered_heading = (
                    bool(numbered_heading_pattern.match(text))
                    and line["is_bold"]
                )

                # Detect short, bold, unnumbered headings
                is_formatted_heading = (
                    line["is_bold"]
                    and line["font_size"] >= body_font_size
                    and len(text) <= 80
                    and not text.endswith((".", ",", ";", ":"))
                )

                if (
                    is_numbered_heading
                    or is_formatted_heading
                ):

                    extracted_headings.append({
                        "page_number": line["page_number"],
                        "text": text,
                        "font_size": line["font_size"],
                        "is_bold": line["is_bold"],
                        "coordinates": line["coordinates"]
                    })

    except pymupdf.FileDataError as error:
        raise ValueError(
            f"Could not read PDF file: {error}"
        ) from error

    return extracted_headings


def normalize_table_cell(value):
    """
    Normalize whitespace inside a table cell.

    - Replace line breaks with spaces.
    - Collapse repeated whitespace.
    - Remove unnecessary spaces before punctuation.
    - Preserve the original characters and numerical values.
    """

    if value is None:
        return None

    text = str(value)

    # Replace line breaks and tabs with spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()



def extract_tables_from_pdf(file_path):
    """
    Extract tables from every page of a PDF.

    Returns:
    - Page number
    - Table number
    - Row count
    - Column count
    - Original extracted table data
    - Normalized table data
    - Bounding box coordinates
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_tables = []

    try:
        with pymupdf.open(file_path) as pdf:

            for page_number, page in enumerate(pdf, start=1):

                table_finder = page.find_tables()

                for table_number, table in enumerate(
                    table_finder.tables,
                    start=1
                ):

                    original_data = table.extract()

                    if not original_data:
                        continue

                    # Normalize cell text without changing its content
                    normalized_data = [
                        [
                            normalize_table_cell(cell)
                            for cell in row
                        ]
                        for row in original_data
                    ]

                    extracted_tables.append({
                        "page_number": page_number,
                        "table_number": table_number,

                        "row_count": len(normalized_data),

                        "column_count": max(
                            (len(row) for row in normalized_data),
                            default=0
                        ),

                        "original_data": original_data,

                        "data": normalized_data,

                        "coordinates": {
                            "x0": table.bbox[0],
                            "y0": table.bbox[1],
                            "x1": table.bbox[2],
                            "y1": table.bbox[3]
                        }
                    })

    except pymupdf.FileDataError as error:
        raise ValueError(
            f"Could not read PDF file: {error}"
        ) from error

    return extracted_tables




def process_pdf(file_path):
    """
    Process a PDF and extract all available information.

    Extracts:
    1. Page-wise text
    2. Individual text blocks
    3. Text with formatting information
    4. Document headings
    5. Tables

    Returns a dictionary containing all extracted information.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    # Extract all PDF components
    pages = extract_text_from_pdf(file_path)

    text_blocks = extract_text_blocks_from_pdf(file_path)

    formatted_spans = extract_text_with_formatting(file_path)

    headings = extract_headings_from_pdf(file_path)

    tables = extract_tables_from_pdf(file_path)

    # Combine all extracted information
    return {
        "file_name": file_path.name,

        "total_pages": len(pages),

        "pages": pages,

        "text_blocks": text_blocks,

        "formatted_spans": formatted_spans,

        "headings": headings,

        "tables": tables,

        "statistics": {
            "total_pages": len(pages),
            "total_text_blocks": len(text_blocks),
            "total_formatted_spans": len(formatted_spans),
            "total_headings": len(headings),
            "total_tables": len(tables)
        }
    }