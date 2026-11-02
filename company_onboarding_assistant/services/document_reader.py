import io
import os
from typing import Any

import easyocr
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from pptx import Presentation
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

with console.status("[bold green]Loading EasyOCR Model (English)..."):
    reader = easyocr.Reader(["en"])


def process_txt(file_path: str) -> str:
    """Reads a plain text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def process_pdf(file_path: str) -> str:
    """Converts PDF to images and applies EasyOCR."""
    extracted_text: list[str] = []

    with console.status(f"[bold cyan]Converting {os.path.basename(file_path)} to images..."):
        pages = convert_from_path(file_path)

    for i, page in enumerate(pages):
        with console.status(f"[bold yellow]Running OCR on PDF page {i + 1}/{len(pages)}..."):
            img_array = np.array(page)
            result = reader.readtext(img_array, detail=0)
            extracted_text.append(f"--- Page {i + 1} ---")
            extracted_text.append(" ".join(result))

    return "\n".join(extracted_text)


def process_pptx(file_path: str) -> str:
    """Extracts text from slides and applies OCR to embedded images."""
    prs = Presentation(file_path)
    extracted_text: list[str] = []

    for i, slide in enumerate(prs.slides):
        slide_text = [f"--- Slide {i + 1} ---"]

        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text.append(shape.text)

            if shape.shape_type == 13:
                with console.status(f"[bold magenta]Running OCR on image found in Slide {i + 1}..."):
                    image_bytes = shape.image.blob
                    img = Image.open(io.BytesIO(image_bytes))
                    img_array = np.array(img.convert("RGB"))
                    ocr_result = reader.readtext(img_array, detail=0)
                    if ocr_result:
                        slide_text.append("[Image OCR Text]: " + " ".join(ocr_result))

        extracted_text.append("\n".join(slide_text))

    return "\n\n".join(extracted_text)


def read_document(file_path: str) -> str | None:
    """Routes the file to the correct processor and prints it beautifully."""
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/] File '{file_path}' not found.")
        return None

    ext = os.path.splitext(file_path)[1].lower()
    content = ""

    try:
        if ext == ".txt":
            content = process_txt(file_path)
        elif ext == ".pdf":
            content = process_pdf(file_path)
        elif ext == ".pptx":
            content = process_pptx(file_path)
        elif ext == ".ppt":
            console.print("[bold red]Warning:[/] Please save your .ppt file as a newer .pptx file first.")
            return None
        else:
            console.print(f"[bold red]Unsupported file format:[/] {ext}")
            return None

        panel = Panel(
            Text(content, justify="left", style="white"),
            title=f"[bold cyan]Document Content: {os.path.basename(file_path)}[/]",
            border_style="green",
            padding=(1, 2),
        )
        console.print(panel)
        return content
    except Exception as exc:  # pragma: no cover - CLI utility
        console.print(f"[bold red]An error occurred while processing {file_path}:[/] {exc}")
        return None


if __name__ == "__main__":
    sample_file = "/Users/raghavpatidar/Portfolio/blog-agent/documents/company_onboarding_handbook.pdf"
    read_document(sample_file)
