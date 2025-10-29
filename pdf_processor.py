import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytesseract
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

load_dotenv()

client = OpenAI()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")


def _call_openai(messages: List[Dict[str, str]]) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
        )
    except Exception as exc:  # pragma: no cover - surface API issues to user
        tqdm.write(f"OpenAI API error: {exc}")
        return ""

    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped.splitlines()[1:-1]
        return "\n".join(inner).strip()
    return stripped


def _parse_layout_response(raw_text: str) -> List[Dict[str, Any]]:
    if not raw_text:
        return []

    cleaned = _strip_code_fences(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            payload = json.loads(cleaned.replace("'", '"'))
        except json.JSONDecodeError:
            return []

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    parsed: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue

        candidate_type = str(entry.get("type", "paragraph")).lower()
        if candidate_type not in {"paragraph", "table", "header"}:
            candidate_type = "paragraph"

        line_numbers: List[int] = []
        for raw_value in entry.get("lines", []) or []:
            try:
                value = int(float(raw_value))
            except (TypeError, ValueError):
                continue
            if value > 0:
                line_numbers.append(value)

        if not line_numbers:
            continue

        text_field = entry.get("text") if isinstance(entry.get("text"), str) else ""
        unique_lines = list(dict.fromkeys(sorted(line_numbers)))
        parsed.append({
            "type": candidate_type,
            "lines": unique_lines,
            "text": text_field,
        })

    return parsed


def llm_agent_layout_segmenter(page_lines: List[str]) -> List[Dict[str, Any]]:
    if not page_lines:
        return []

    enumerated_lines = "\n".join(f"{idx + 1}: {line}" for idx, line in enumerate(page_lines))
    system_msg = (
        "You are a precise document layout analyst. Respond ONLY with JSON. "
        "Each array item MUST include keys: \"type\" (paragraph|table|header) and \"lines\" "
        "(array of line numbers). Use every line exactly once; if a line is unclear, place it in "
        "a paragraph segment."
    )
    user_msg = (
        "Group the OCR lines below into ordered segments. Mark tabular regions as type \"table\", "
        "short titles as \"header\", and everything else as \"paragraph\". Ensure every line "
        "appears in exactly one segment. Return a JSON array.\n\n"
        + enumerated_lines
    )
    raw_response = _call_openai([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])
    return _parse_layout_response(raw_response)


def llm_agent_table_parser(table_text: str) -> str:
    system_msg = (
        "You convert OCR tables into well-formatted Markdown tables. "
        "Preserve every row and column, normalise spacing, and avoid dropping data."
    )
    user_msg = (
        "Reformat the raw table text below into a Markdown table. "
        "If a cell spans multiple lines, join the lines with spaces."
        " Do not add commentary outside the table itself.\n\n"
        + table_text
    )
    cleaned = _call_openai([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]).strip()

    cleaned = _strip_code_fences(cleaned)

    if not cleaned:
        cleaned = table_text.strip()

    raw_block = table_text.strip()
    if raw_block:
        cleaned = cleaned.rstrip()
        cleaned += (
            "\n\n<details>\n<summary>Raw OCR Table</summary>\n\n````\n"
            + raw_block
            + "\n````\n</details>"
        )

    return cleaned


def llm_agent_prose_cleaner(prose_text: str) -> str:
    system_msg = (
        "You polish OCR prose into clear Markdown while preserving meaning. "
        "Use headings and bullet lists only when helpful, and keep all factual details. "
        "Do not add commentary, apologies, or invented content."
    )
    user_msg = (
        "Clean the OCR text below so it reads well. "
        "Return Markdown with concise paragraphs and optional bullet lists. "
        "Keep every factual detail and avoid wrapping the answer in code fences.\n\n"
        + prose_text
    )
    cleaned = _call_openai([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]).strip()
    cleaned = _strip_code_fences(cleaned)
    return cleaned or prose_text.strip()


def get_ocr_data(page_image: Image.Image) -> Dict[str, List[Any]]:
    return pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)


def extract_lines(ocr_dict: Dict[str, List[Any]]) -> List[str]:
    entries: List[Tuple[int, int, str]] = []

    for idx, token in enumerate(ocr_dict.get("text", [])):
        word = token.strip()
        if not word:
            continue

        top = ocr_dict.get("top", [0])[idx]
        left = ocr_dict.get("left", [0])[idx]
        entries.append((top, left, word))

    if not entries:
        return []

    entries.sort(key=lambda item: (item[0], item[1]))

    line_threshold = 20
    lines: List[List[Tuple[int, str]]] = []
    current_line: List[Tuple[int, str]] = []
    last_top: int | None = None

    for top, left, text in entries:
        if last_top is None or abs(top - last_top) <= line_threshold:
            current_line.append((left, text))
        else:
            lines.append(sorted(current_line, key=lambda part: part[0]))
            current_line = [(left, text)]
        last_top = top

    if current_line:
        lines.append(sorted(current_line, key=lambda part: part[0]))

    return [" ".join(piece for _, piece in line).strip() for line in lines if line]


def _append_fallback(page_sections: List[str], fallback_source: str) -> None:
    if not fallback_source.strip():
        return
    recovered = llm_agent_prose_cleaner(fallback_source)
    if recovered:
        page_sections.append(recovered)


def process_pdf(pdf_path: Path, pages_pbar: tqdm, output_dir: Path) -> None:
    if not pdf_path.exists():
        tqdm.write(f"Error: PDF file not found at {pdf_path}")
        return

    try:
        images = convert_from_path(str(pdf_path))
    except Exception as exc:
        tqdm.write(f"Error converting {pdf_path.name} to images: {exc}")
        tqdm.write("Ensure Poppler is installed and available on PATH.")
        return

    if not images:
        tqdm.write(f"No pages found in {pdf_path.name}.")
        return

    pages_pbar.reset(total=len(images))
    assembled_pages: List[str] = []

    for page_index, page_image in enumerate(images, start=1):
        pages_pbar.set_description(f"Page {page_index}/{len(images)}")

        sections: List[str] = [f"# Page {page_index}"]
        ocr_dict = get_ocr_data(page_image)
        page_lines = extract_lines(ocr_dict)
        plain_text = "\n".join(page_lines)

        if not page_lines:
            _append_fallback(sections, plain_text)
            assembled_pages.append("\n\n".join(sections))
            pages_pbar.update(1)
            continue

        layout = llm_agent_layout_segmenter(page_lines)
        content_elements = layout or []

        content_pbar = tqdm(
            total=len(content_elements),
            desc="Extracting Content",
            leave=False,
            position=pages_pbar.pos + 1,
        )

        pieces: List[Dict[str, Any]] = []
        used_line_numbers: Set[int] = set()

        for element in content_elements:
            line_numbers: List[int] = []
            for raw_value in element.get("lines", []):
                try:
                    value = int(float(raw_value))
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    line_numbers.append(value)
            segment_lines = [
                page_lines[line_num - 1]
                for line_num in line_numbers
                if 0 < line_num <= len(page_lines)
            ]
            element_text = "\n".join(segment_lines).strip()

            if not element_text:
                content_pbar.update(1)
                continue

            element_type = str(element.get("type", "paragraph")).lower()
            if element_type == "table":
                parsed = llm_agent_table_parser(element_text)
            elif element_type == "header":
                parsed = f"### {element_text.strip()}"
            else:
                parsed = llm_agent_prose_cleaner(element_text)

            first_line = min(line_numbers) if line_numbers else len(page_lines) + 1
            pieces.append({"order": first_line, "content": parsed.strip()})
            used_line_numbers.update(line_numbers)
            content_pbar.update(1)

        content_pbar.close()

        if not pieces:
            _append_fallback(sections, plain_text)
        else:
            for piece in sorted(pieces, key=lambda entry: entry["order"]):
                sections.append(piece["content"])

            missing_lines = [
                page_lines[idx]
                for idx in range(len(page_lines))
                if (idx + 1) not in used_line_numbers and page_lines[idx]
            ]
            if missing_lines:
                sections.append(
                    "<details>\n"
                    "<summary>Additional OCR Lines</summary>\n\n"
                    "```text\n"
                    + "\n".join(missing_lines)
                    + "\n```\n"
                    "</details>"
                )

        assembled_pages.append("\n\n".join(sections))
        pages_pbar.update(1)

    output_path = output_dir / pdf_path.with_suffix(".md").name
    output_path.write_text("\n\n".join(assembled_pages) + "\n", encoding="utf-8")
    tqdm.write(f"✔ Saved markdown to {output_path.name}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    pdf_dir = script_dir / "pdf"

    if not pdf_dir.is_dir():
        print(f"Error: 'pdf' directory not found in {script_dir}")
        print("Create the folder and place PDF files inside before running again.")
        return

    pdf_files = sorted(path for path in pdf_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_files:
        print("No PDF files found in the 'pdf' directory.")
        return

    files_pbar = tqdm(total=len(pdf_files), desc="Overall Progress", position=0)
    pages_pbar = tqdm(total=0, desc="Pages", position=1, leave=False)

    for pdf_path in pdf_files:
        files_pbar.set_description(f"Processing {pdf_path.name}")
        try:
            process_pdf(pdf_path, pages_pbar, script_dir)
        except Exception as exc:
            tqdm.write(f"Unexpected error while processing {pdf_path.name}: {exc}")
        files_pbar.update(1)

    pages_pbar.close()
    files_pbar.close()
    print("\nAll PDF files processed.")


if __name__ == "__main__":
    main()
