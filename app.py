from __future__ import annotations
import json
import os
import re
import unicodedata
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "App Diario de Obra - RDO Digital"
MODEL_VENDOR = "Update Digital Tecnologia da Informacao LTDA"

SECTION_ALIASES = {
    "labor": ("mao de obra",),
    "equipment": ("equipamentos",),
    "activities": ("atividades",),
    "occurrences": ("ocorrencias",),
    "comments": ("comentarios",),
    "photos": ("fotos",),
}

NEXT_SECTION_NAMES = (
    "horario de trabalho",
    "condicao climatica",
    "mao de obra",
    "equipamentos",
    "atividades",
    "ocorrencias",
    "controle de material",
    "comentarios",
    "fotos",
    "galeria de fotos",
    "videos",
    "anexos",
    "assinaturas",
    "assinatura",
    "criado por",
)

SUMMARY_LABOR_HINTS = (
    "mao de obra direta",
    "mao de obra indireta",
    "direta",
    "indireta",
)

OCCURRENCE_TAG_HINTS = (
    "solicitacoes do cliente",
    "solicitacao do cliente",
    "cliente",
    "chuva",
    "clima",
    "seguranca",
    "interferencia",
    "impedimento",
    "paralisacao",
    "material",
    "equipamento",
    "projeto",
)

STATUS_APPROVAL_PATTERNS = (
    "aguardando aprovacao",
    "aprovado",
    "reprovado",
    "revisar relatorio",
    "preenchendo relatorio",
)


# ============================================================
# GENERAL HELPERS
# ============================================================


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_cell(value)).strip()


def norm(value: Any) -> str:
    text = one_line(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("\u00ba", "o").replace("\u00b0", "o")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def source_name(source: Any) -> str:
    if isinstance(source, (str, os.PathLike)):
        return os.path.basename(os.fspath(source))
    return getattr(source, "name", "rdo.pdf")


def read_source_bytes(source: Any) -> bytes:
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as handle:
            return handle.read()

    try:
        source.seek(0)
    except Exception:
        pass

    data = source.read()

    try:
        source.seek(0)
    except Exception:
        pass

    return data


def extract_full_text(pdf_bytes: bytes) -> Tuple[str, List[str]]:
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text", sort=True) or ""
            text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
            text = text.replace("\u00a0", " ").replace("\r", "\n")
            text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
            text = re.sub(r"[ \t]+", " ", text)
            pages.append(text.strip())
    return "\n".join(pages), pages


def extract_tables(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    found = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, table in enumerate(page.extract_tables() or []):
                if not table:
                    continue
                cleaned = []
                for row in table:
                    if row is None:
                        continue
                    cleaned.append([clean_cell(cell) for cell in row])
                if cleaned:
                    found.append({
                        "page": page_number,
                        "table_index": table_index,
                        "rows": cleaned,
                    })
    return found


def first_nonempty(cells: Iterable[Any]) -> str:
    for cell in cells:
        value = one_line(cell)
        if value:
            return value
    return ""


def table_signature(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    preview = []
    for row in rows[:2]:
        preview.extend(cell for cell in row if one_line(cell))
    return norm(" | ".join(preview))


def section_count(text: str) -> Optional[int]:
    match = re.search(r"\((\d+)\)", text)
    return int(match.group(1)) if match else None


def parse_number(value: str) -> Optional[float]:
    text = one_line(value)
    match = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def dedupe_records(records: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for record in records:
        key = tuple(str(record.get(k, "")) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


# ============================================================
# MODEL DETECTION
# ============================================================


def detect_model(text: str) -> Dict[str, Any]:
    n = norm(text)
    signatures = {
        "relatorio diario de obra (rdo)": "relatorio diario de obra (rdo)" in n,
        "data do relatorio": "data do relatorio" in n,
        "dia da semana": "dia da semana" in n,
        "prazo contratual": "prazo contratual" in n,
        "mao de obra": "mao de obra" in n,
        "equipamentos": "equipamentos" in n,
        "atividades": "atividades" in n,
        "criado por": "criado por:" in n,
        "ultima modificacao": "ultima modificacao:" in n,
    }
    score = sum(signatures.values())
    total = len(signatures)
    confidence = score / total if total else 0.0

    if score >= 6:
        identified = MODEL_NAME
    elif score >= 4:
        identified = MODEL_NAME + " (modelo provavel)"
    else:
        identified = "Modelo nao identificado"

    return {
        "software": identified,
        "fabricante": MODEL_VENDOR if score >= 4 else "",
        "confianca": round(confidence, 3),
        "assinaturas": signatures,
    }


# ============================================================
# HEADER AND CLIMATE
# ============================================================

HEADER_LABELS = {
    "relatorio n": "numero_rdo",
    "relatorio no": "numero_rdo",
    "data do relatorio": "data_rdo",
    "dia da semana": "dia_semana",
    "contrato": "contrato",
    "obra": "obra",
    "local": "local",
    "cliente": "cliente",
    "contratante": "cliente",
    "responsavel": "responsavel",
    "prazo contratual": "prazo_contratual",
    "prazo decorrido": "prazo_decorrido",
    "prazo a vencer": "prazo_a_vencer",
}


def _next_nonempty_right(row: List[str], start_index: int) -> str:
    for index in range(start_index + 1, len(row)):
        value = one_line(row[index])
        if value:
            return value
    return ""


def extract_header(tables: List[Dict[str, Any]], first_page_text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "numero_rdo": "",
        "data_rdo": "",
        "dia_semana": "",
        "contrato": "",
        "obra": "",
        "periodo_contratual": "",
        "local": "",
        "cliente": "",
        "responsavel": "",
        "prazo_contratual": "",
        "prazo_decorrido": "",
        "prazo_a_vencer": "",
    }

    for item in tables:
        if item["page"] != 1:
            continue
        rows = item["rows"]
        sig = table_signature(rows)
        if "relatorio diario de obra" not in sig and not any(
            "relatorio diario de obra" in norm(" ".join(row)) for row in rows
        ):
            continue

        for row in rows:
            for idx, cell in enumerate(row):
                label = norm(cell).rstrip(":")
                if label in HEADER_LABELS:
                    value = _next_nonempty_right(row, idx)
                    if value:
                        data[HEADER_LABELS[label]] = value

    # Split work name and contractual period when they share the same cell.
    obra_flat = one_line(data["obra"])
    period_match = re.search(
        r"(\d{2}/\d{2}/\d{4})\s+(?:ate|at\S*)\s+(\d{2}/\d{2}/\d{4})$",
        norm(obra_flat),
    )
    if period_match:
        # Use the position of the first date in the original flattened value.
        first_date = period_match.group(1)
        pos = obra_flat.find(first_date)
        if pos >= 0:
            data["periodo_contratual"] = obra_flat[pos:].strip()
            data["obra"] = obra_flat[:pos].strip()
        else:
            data["obra"] = obra_flat
    else:
        data["obra"] = obra_flat

    # Regex fallbacks for PDFs where the header grid is not extracted correctly.
    fallback_patterns = {
        "numero_rdo": [
            r"Relat[o\u00f3]rio(?:\s+\d{2}/\d{2}/\d{4})?\s+n[\u00b0\u00ba\u00bfo]?\s*(\d+)",
            r"Relat[o\u00f3]rio\s+n[\u00b0\u00ba]?\s*(\d+)",
        ],
        "data_rdo": [r"Data\s+do\s+relat[o\u00f3]rio\s*(\d{2}/\d{2}/\d{4})"],
        "dia_semana": [r"Dia\s+da\s+semana\s*([A-Za-z\u00c0-\u00ff-]+)"],
    }
    for field, patterns in fallback_patterns.items():
        if data[field]:
            continue
        for pattern in patterns:
            match = re.search(pattern, first_page_text, flags=re.I)
            if match:
                data[field] = one_line(match.group(1))
                break

    return data


def extract_climate(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for item in tables:
        rows = item["rows"]
        sig = table_signature(rows)
        if "condicao climatica" not in sig:
            continue
        for row in rows[1:]:
            cells = [one_line(cell) for cell in row]
            if not any(cells):
                continue
            records.append({
                "turno": cells[0] if len(cells) > 0 else "",
                "tempo": cells[1] if len(cells) > 1 else "",
                "condicao": cells[2] if len(cells) > 2 else "",
                "pagina": item["page"],
            })
    return records


# ============================================================
# LABOR AND EQUIPMENT
# ============================================================


def parse_resource_cell(cell: str) -> Optional[Tuple[str, int, bool]]:
    lines = [one_line(line) for line in clean_cell(cell).splitlines() if one_line(line)]
    if not lines:
        return None

    joined = " ".join(lines)

    # Summary/group cell, e.g. "EnerMais (2)" or "Mao de Obra Direta (27)".
    paren_match = re.match(r"^(.*?)\s*\((\d+)\)\s*$", joined)
    if paren_match:
        return one_line(paren_match.group(1)), int(paren_match.group(2)), True

    # Regular cell: last extracted line is the quantity.
    if re.fullmatch(r"\d+", lines[-1]):
        description = one_line(" ".join(lines[:-1]))
        if description:
            return description, int(lines[-1]), False

    # Fallback if line breaks were lost.
    match = re.match(r"^(.*?)\s+(\d+)\s*$", joined)
    if match:
        description = one_line(match.group(1))
        quantity = int(match.group(2))
        return description, quantity, False

    return None


def extract_resources(
    tables: List[Dict[str, Any]],
    section: str,
    file_name: str,
    header: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    aliases = SECTION_ALIASES[section]
    records: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for item in tables:
        rows = item["rows"]
        sig = table_signature(rows)
        if not any(alias in sig for alias in aliases):
            continue

        heading = first_nonempty(rows[0])
        declared_total = section_count(heading)

        for row in rows[1:]:
            for cell in row:
                parsed = parse_resource_cell(cell)
                if not parsed:
                    continue
                description, quantity, is_summary = parsed
                dnorm = norm(description)

                # Protect against accidentally parsing section headers.
                if any(alias == dnorm for alias in aliases):
                    continue

                base = {
                    "Arquivo": file_name,
                    "N RDO": header.get("numero_rdo", ""),
                    "Data": header.get("data_rdo", ""),
                    "Descricao": description,
                    "Quantidade": quantity,
                    "Pagina": item["page"],
                    "Total declarado da secao": declared_total,
                }

                if section == "labor" and (is_summary or any(hint in dnorm for hint in SUMMARY_LABOR_HINTS)):
                    summaries.append({**base, "Tipo resumo": "Grupo/Resumo"})
                elif section == "labor" and is_summary:
                    summaries.append({**base, "Tipo resumo": "Grupo/Resumo"})
                elif section == "equipment" and is_summary:
                    # Equipment cells normally have a regular trailing quantity. Keep unusual
                    # parenthesized cells as summaries instead of inventing an equipment item.
                    summaries.append({**base, "Tipo resumo": "Grupo/Resumo"})
                else:
                    records.append(base)

    records = dedupe_records(records, ["Arquivo", "N RDO", "Descricao", "Quantidade", "Pagina"])
    summaries = dedupe_records(summaries, ["Arquivo", "N RDO", "Descricao", "Quantidade", "Pagina"])
    return records, summaries


# ============================================================
# ACTIVITIES
# ============================================================

CODE_LINE_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*[-\u2013]\s*(.+)$")


def parse_quantity_unit(text: str) -> Tuple[Optional[float], str, str]:
    raw = one_line(text)
    if not raw:
        return None, "", ""
    match = re.match(r"^(-?\d+(?:[.,]\d+)?)\s*(.*)$", raw)
    if not match:
        return None, "", raw
    value = float(match.group(1).replace(",", "."))
    unit = one_line(match.group(2))
    return value, unit, raw


def parse_activity_status(text: str) -> Dict[str, Any]:
    raw = clean_cell(text)
    flat = one_line(raw)
    n = norm(flat)

    start = ""
    end = ""
    hours = None
    percent = None
    status = ""

    match_time = re.search(r"(\d{1,2}:\d{2})\s*(?:ate|a)\s*(\d{1,2}:\d{2})", n)
    if match_time:
        start, end = match_time.group(1), match_time.group(2)

    match_hours = re.search(r"\((\d+(?:[.,]\d+)?)\s*h\)", flat, flags=re.I)
    if match_hours:
        hours = float(match_hours.group(1).replace(",", "."))

    match_percent = re.search(r"(\d+(?:[.,]\d+)?)\s*%", flat)
    if match_percent:
        percent = float(match_percent.group(1).replace(",", "."))
        after = flat[match_percent.end():]
        after = re.sub(r"^[\s\-\u2013:]+", "", after).strip()
        status = after
    elif flat:
        # Keep a short status-only value if present.
        status_lines = [one_line(line) for line in raw.splitlines() if one_line(line)]
        for line in reversed(status_lines):
            if any(word in norm(line) for word in ("andamento", "conclu", "paralis", "nao inici")):
                status = line
                break

    return {
        "Hora inicio": start,
        "Hora fim": end,
        "Horas": hours,
        "Percentual": percent,
        "Status": status,
        "Status bruto": flat,
    }


def parse_activity_description(text: str) -> Dict[str, str]:
    lines = [one_line(line) for line in clean_cell(text).splitlines() if one_line(line)]
    if not lines:
        return {
            "Codigo etapa": "",
            "Etapa": "",
            "Codigo atividade": "",
            "Atividade": "",
            "Descricao executada": "",
            "Texto bruto": "",
        }

    if norm(lines[0]) == "qssma":
        comments = [re.sub(r"^[\-\u2013]\s*", "", line).strip() for line in lines[1:]]
        return {
            "Codigo etapa": "",
            "Etapa": "QSSMA",
            "Codigo atividade": "",
            "Atividade": "QSSMA",
            "Descricao executada": one_line(" ".join(comments)),
            "Texto bruto": " | ".join(lines),
        }

    coded = []
    comments = []
    for line in lines:
        match = CODE_LINE_RE.match(line)
        if match:
            coded.append((match.group(1), one_line(match.group(2))))
        else:
            comments.append(re.sub(r"^[\-\u2013]\s*", "", line).strip())

    if coded:
        etapa_code, etapa = coded[0]
        if len(coded) >= 2:
            activity_code, activity = coded[1]
        else:
            activity_code, activity = coded[0]
    else:
        etapa_code = ""
        etapa = ""
        activity_code = ""
        activity = lines[0]
        comments = lines[1:]

    return {
        "Codigo etapa": etapa_code,
        "Etapa": etapa,
        "Codigo atividade": activity_code,
        "Atividade": activity,
        "Descricao executada": one_line(" ".join(comments)),
        "Texto bruto": " | ".join(lines),
    }


def extract_activities(
    tables: List[Dict[str, Any]],
    file_name: str,
    header: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records = []
    order = 0
    for item in tables:
        rows = item["rows"]
        sig = table_signature(rows)
        if not any(alias in sig for alias in SECTION_ALIASES["activities"]):
            continue

        declared_total = section_count(first_nonempty(rows[0]))

        for row in rows[1:]:
            if not row or not one_line(row[0] if len(row) > 0 else ""):
                continue
            order += 1
            desc = parse_activity_description(row[0])
            qty, unit, qty_raw = parse_quantity_unit(row[1] if len(row) > 1 else "")
            status = parse_activity_status(row[2] if len(row) > 2 else "")

            records.append({
                "Arquivo": file_name,
                "N RDO": header.get("numero_rdo", ""),
                "Data": header.get("data_rdo", ""),
                "Ordem": order,
                **desc,
                "Quantidade": qty,
                "Unidade": unit,
                "Quantidade bruta": qty_raw,
                **status,
                "Pagina": item["page"],
                "Total declarado da secao": declared_total,
            })

    return dedupe_records(records, ["Arquivo", "N RDO", "Ordem", "Texto bruto", "Pagina"])


# ============================================================
# OCCURRENCES
# ============================================================


def _looks_like_occurrence_tag(line: str) -> bool:
    n = norm(line)
    if not n or len(n) > 80:
        return False
    return any(hint in n for hint in OCCURRENCE_TAG_HINTS)


def parse_occurrence_cell(text: str) -> Tuple[str, str]:
    lines = [one_line(line) for line in clean_cell(text).splitlines() if one_line(line)]
    if not lines:
        return "", ""
    tags = []
    body = lines[:]
    while len(body) > 1 and _looks_like_occurrence_tag(body[-1]):
        tags.insert(0, body.pop())
    return one_line(" ".join(body)), "; ".join(tags)


def extract_occurrences(
    tables: List[Dict[str, Any]],
    file_name: str,
    header: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records = []
    order = 0

    for item in tables:
        rows = item["rows"]
        sig = table_signature(rows)
        if not any(alias in sig for alias in SECTION_ALIASES["occurrences"]):
            continue

        declared_total = section_count(first_nonempty(rows[0]))
        for row in rows[1:]:
            cells = [clean_cell(cell) for cell in row if one_line(cell)]
            if not cells:
                continue

            duration = ""
            text_cells = []
            for cell in cells:
                if re.fullmatch(r"\d+(?:[.,]\d+)?\s*h", one_line(cell), flags=re.I):
                    duration = one_line(cell)
                else:
                    text_cells.append(cell)

            raw_text = "\n".join(text_cells).strip()
            description, tags = parse_occurrence_cell(raw_text)
            if not description:
                continue

            order += 1
            records.append({
                "Arquivo": file_name,
                "N RDO": header.get("numero_rdo", ""),
                "Data": header.get("data_rdo", ""),
                "Ordem": order,
                "Ocorrencia": description,
                "Tags": tags,
                "Duracao": duration,
                "Horas impactadas": parse_number(duration),
                "Pagina": item["page"],
                "Total declarado da secao": declared_total,
            })

    return dedupe_records(records, ["Arquivo", "N RDO", "Ocorrencia", "Duracao", "Pagina"])


# ============================================================
# COMMENTS
# ============================================================

TIMESTAMP_RE = re.compile(r"\b(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})(?::\d{2})?\b")


def extract_comments_from_tables(
    tables: List[Dict[str, Any]],
    file_name: str,
    header: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records = []
    order = 0

    for item in tables:
        rows = item["rows"]
        sig = table_signature(rows)
        if not any(alias in sig for alias in SECTION_ALIASES["comments"]):
            continue

        declared_total = section_count(first_nonempty(rows[0]))
        current: Optional[Dict[str, Any]] = None

        def flush() -> None:
            nonlocal current, order
            if not current:
                return
            text = one_line(" ".join(current.pop("_parts", [])))
            if text:
                order += 1
                current["Ordem"] = order
                current["Comentario"] = text
                records.append(current)
            current = None

        for row in rows[1:]:
            cells = [one_line(cell) for cell in row if one_line(cell)]
            if not cells:
                continue

            row_text = " | ".join(cells)
            timestamp_match = TIMESTAMP_RE.search(row_text)

            if timestamp_match:
                flush()
                timestamp = f"{timestamp_match.group(1)} {timestamp_match.group(2)}"
                author_parts = []
                text_parts = []

                for cell in cells:
                    if TIMESTAMP_RE.search(cell):
                        cleaned = TIMESTAMP_RE.sub("", cell).strip(" -|")
                        if cleaned:
                            text_parts.append(cleaned)
                    else:
                        author_parts.append(cell)

                author = one_line(" ".join(author_parts))
                current = {
                    "Arquivo": file_name,
                    "N RDO": header.get("numero_rdo", ""),
                    "Data": header.get("data_rdo", ""),
                    "Responsavel pelo comentario": author,
                    "Data/hora comentario": timestamp,
                    "Pagina": item["page"],
                    "Total declarado da secao": declared_total,
                    "_parts": text_parts,
                }
            else:
                if current is None:
                    current = {
                        "Arquivo": file_name,
                        "N RDO": header.get("numero_rdo", ""),
                        "Data": header.get("data_rdo", ""),
                        "Responsavel pelo comentario": "",
                        "Data/hora comentario": "",
                        "Pagina": item["page"],
                        "Total declarado da secao": declared_total,
                        "_parts": [],
                    }
                current["_parts"].extend(cells)

        flush()

    return records


def _extract_text_section(full_text: str, start_name: str) -> str:
    start_re = re.compile(rf"^\s*{start_name}\s*\(\s*\d+\s*\)\s*$", re.I | re.M)
    start = start_re.search(full_text)
    if not start:
        return ""

    rest = full_text[start.end():]
    end_patterns = [name for name in NEXT_SECTION_NAMES if norm(name) != norm(start_name)]
    end_re = re.compile(
        r"^\s*(?:" + "|".join(re.escape(name) for name in end_patterns) + r")\s*(?:\(\s*\d+\s*\))?\s*$",
        re.I | re.M,
    )
    end = end_re.search(rest)
    if end:
        rest = rest[:end.start()]
    return rest.strip()


def extract_comments_fallback(
    full_text: str,
    file_name: str,
    header: Dict[str, Any],
) -> List[Dict[str, Any]]:
    section = _extract_text_section(full_text, "Coment[a\u00e1]rios")
    if not section:
        return []

    lines = [one_line(line) for line in section.splitlines() if one_line(line)]
    records = []
    current = None
    order = 0

    for line in lines:
        match = TIMESTAMP_RE.search(line)
        if match:
            if current and current["_parts"]:
                order += 1
                current["Ordem"] = order
                current["Comentario"] = one_line(" ".join(current.pop("_parts")))
                records.append(current)

            author = one_line(line[:match.start()])
            current = {
                "Arquivo": file_name,
                "N RDO": header.get("numero_rdo", ""),
                "Data": header.get("data_rdo", ""),
                "Responsavel pelo comentario": author,
                "Data/hora comentario": f"{match.group(1)} {match.group(2)}",
                "Pagina": "",
                "Total declarado da secao": None,
                "_parts": [],
            }
        elif current:
            current["_parts"].append(line)

    if current and current["_parts"]:
        order += 1
        current["Ordem"] = order
        current["Comentario"] = one_line(" ".join(current.pop("_parts")))
        records.append(current)

    return records


# ============================================================
# APPROVALS AND AUDIT TRAIL
# ============================================================


def extract_approvals(pdf_bytes: bytes, file_name: str, header: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = []
    email_re = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            blocks = []
            for block in page.get_text("blocks", sort=True):
                x0, y0, x1, y1, text = block[:5]
                text = clean_cell(text.replace("\ufb01", "fi").replace("\ufb02", "fl"))
                if not text:
                    continue
                blocks.append({
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "cx": (x0 + x1) / 2.0,
                    "text": text,
                })

            status_blocks = []
            for block in blocks:
                first_line = one_line(block["text"].splitlines()[0])
                nline = norm(first_line)
                if any(nline.startswith(candidate) for candidate in STATUS_APPROVAL_PATTERNS):
                    status_blocks.append(block)

            if not status_blocks:
                continue

            status_blocks.sort(key=lambda b: b["cx"])
            centers = [b["cx"] for b in status_blocks]

            for idx, status_block in enumerate(status_blocks):
                left = -float("inf") if idx == 0 else (centers[idx - 1] + centers[idx]) / 2.0
                right = float("inf") if idx == len(centers) - 1 else (centers[idx] + centers[idx + 1]) / 2.0

                payload_lines = []
                for block in blocks:
                    if block is status_block:
                        continue
                    if block["y0"] < status_block["y1"] - 1:
                        continue
                    if block["y0"] > status_block["y1"] + 55:
                        continue
                    if not (left <= block["cx"] < right):
                        continue
                    for line in block["text"].splitlines():
                        line = one_line(line)
                        if not line or re.fullmatch(r"\d+\s*/\s*\d+", line):
                            continue
                        if norm(line).startswith("criado por:") or norm(line).startswith("ultima modificacao:"):
                            continue
                        payload_lines.append(line)

                email_index = next((i for i, value in enumerate(payload_lines) if email_re.match(value)), None)
                if email_index is None:
                    continue

                name = payload_lines[email_index - 1] if email_index >= 1 else ""
                email = payload_lines[email_index]
                role = payload_lines[email_index + 1] if email_index + 1 < len(payload_lines) else ""

                status_line = one_line(status_block["text"])
                timestamp_match = TIMESTAMP_RE.search(status_line)
                timestamp = ""
                if timestamp_match:
                    timestamp = f"{timestamp_match.group(1)} {timestamp_match.group(2)}"

                records.append({
                    "Arquivo": file_name,
                    "N RDO": header.get("numero_rdo", ""),
                    "Data": header.get("data_rdo", ""),
                    "Status": status_line,
                    "Data/hora aprovacao": timestamp,
                    "Nome": name,
                    "Email": email,
                    "Funcao": role,
                    "Pagina": page_number,
                })

    return dedupe_records(records, ["Arquivo", "N RDO", "Status", "Nome", "Email"])

def extract_audit(full_text: str) -> Dict[str, str]:
    result = {
        "criado_por": "",
        "criado_em": "",
        "ultima_modificacao_por": "",
        "ultima_modificacao_em": "",
    }

    created = re.search(
        r"Criado\s+por:\s*(.*?)\s*\((\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})\)",
        full_text,
        flags=re.I,
    )
    if created:
        result["criado_por"] = one_line(created.group(1))
        result["criado_em"] = one_line(created.group(2))

    modified = re.search(
        r"(?:Ultima|\u00daltima)\s+modifica(?:cao|\u00e7\u00e3o):\s*(.*?)\s*\((\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})\)",
        full_text,
        flags=re.I,
    )
    if modified:
        result["ultima_modificacao_por"] = one_line(modified.group(1))
        result["ultima_modificacao_em"] = one_line(modified.group(2))

    return result


# ============================================================
# MAIN PARSER
# ============================================================


def parse_rdo(source: Any) -> Dict[str, Any]:
    file_name = source_name(source)
    pdf_bytes = read_source_bytes(source)
    full_text, page_texts = extract_full_text(pdf_bytes)
    tables = extract_tables(pdf_bytes)
    first_page_text = page_texts[0] if page_texts else ""

    model = detect_model(full_text)
    header = extract_header(tables, first_page_text)
    audit = extract_audit(full_text)

    climate = extract_climate(tables)
    labor, labor_summaries = extract_resources(tables, "labor", file_name, header)
    equipment, equipment_summaries = extract_resources(tables, "equipment", file_name, header)
    activities = extract_activities(tables, file_name, header)
    occurrences = extract_occurrences(tables, file_name, header)
    comments = extract_comments_from_tables(tables, file_name, header)
    if not comments:
        comments = extract_comments_fallback(full_text, file_name, header)
    approvals = extract_approvals(pdf_bytes, file_name, header)

    summary = {
        "Arquivo": file_name,
        "Software detectado": model["software"],
        "Fabricante": model["fabricante"],
        "Confianca modelo": model["confianca"],
        "N RDO": header.get("numero_rdo", ""),
        "Data": header.get("data_rdo", ""),
        "Dia da semana": header.get("dia_semana", ""),
        "Contrato": header.get("contrato", ""),
        "Obra": header.get("obra", ""),
        "Periodo contratual": header.get("periodo_contratual", ""),
        "Local": header.get("local", ""),
        "Cliente": header.get("cliente", ""),
        "Responsavel": header.get("responsavel", ""),
        "Prazo contratual": header.get("prazo_contratual", ""),
        "Prazo decorrido": header.get("prazo_decorrido", ""),
        "Prazo a vencer": header.get("prazo_a_vencer", ""),
        "Criado por": audit["criado_por"],
        "Criado em": audit["criado_em"],
        "Ultima modificacao por": audit["ultima_modificacao_por"],
        "Ultima modificacao em": audit["ultima_modificacao_em"],
        "Qtd MO extraida": sum(int(row["Quantidade"]) for row in labor),
        "Qtd equipamentos extraida": sum(int(row["Quantidade"]) for row in equipment),
        "Qtd atividades extraidas": len(activities),
        "Qtd ocorrencias extraidas": len(occurrences),
        "Qtd comentarios extraidos": len(comments),
        "Qtd aprovacoes extraidas": len(approvals),
    }

    return {
        "resumo": [summary],
        "clima": [
            {
                "Arquivo": file_name,
                "N RDO": header.get("numero_rdo", ""),
                "Data": header.get("data_rdo", ""),
                **row,
            }
            for row in climate
        ],
        "mao_obra": labor,
        "resumos_mao_obra": labor_summaries,
        "equipamentos": equipment,
        "resumos_equipamentos": equipment_summaries,
        "atividades": activities,
        "ocorrencias": occurrences,
        "comentarios": comments,
        "aprovacoes": approvals,
        "debug": {
            "Arquivo": file_name,
            "modelo": model,
            "header": header,
            "audit": audit,
            "texto": full_text,
        },
    }


def parse_many(sources: Iterable[Any]) -> Dict[str, pd.DataFrame]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "Resumo": [],
        "Clima": [],
        "Mao_Obra": [],
        "Resumo_MO": [],
        "Equipamentos": [],
        "Resumo_EQP": [],
        "Atividades": [],
        "Ocorrencias": [],
        "Comentarios": [],
        "Aprovacoes": [],
    }

    debug_rows = []

    for source in sources:
        parsed = parse_rdo(source)
        buckets["Resumo"].extend(parsed["resumo"])
        buckets["Clima"].extend(parsed["clima"])
        buckets["Mao_Obra"].extend(parsed["mao_obra"])
        buckets["Resumo_MO"].extend(parsed["resumos_mao_obra"])
        buckets["Equipamentos"].extend(parsed["equipamentos"])
        buckets["Resumo_EQP"].extend(parsed["resumos_equipamentos"])
        buckets["Atividades"].extend(parsed["atividades"])
        buckets["Ocorrencias"].extend(parsed["ocorrencias"])
        buckets["Comentarios"].extend(parsed["comentarios"])
        buckets["Aprovacoes"].extend(parsed["aprovacoes"])
        debug_rows.append({
            "Arquivo": parsed["debug"]["Arquivo"],
            "Modelo": parsed["debug"]["modelo"]["software"],
            "Confianca": parsed["debug"]["modelo"]["confianca"],
            "Cabecalho": json.dumps(parsed["debug"]["header"], ensure_ascii=False),
            "Auditoria": json.dumps(parsed["debug"]["audit"], ensure_ascii=False),
        })

    frames = {name: pd.DataFrame(rows) for name, rows in buckets.items()}
    frames["Debug"] = pd.DataFrame(debug_rows)
    return frames


def frames_to_excel(frames: Dict[str, pd.DataFrame]) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in frames.items():
            safe_name = re.sub(r"[\[\]:*?/\\]", "", sheet_name)[:31]
            if df.empty:
                pd.DataFrame({"Info": ["Nenhum registro extraido"]}).to_excel(
                    writer, index=False, sheet_name=safe_name
                )
            else:
                df.to_excel(writer, index=False, sheet_name=safe_name)
    output.seek(0)
    return output


# ============================================================
# BACKWARD-COMPATIBLE WRAPPERS
# ============================================================


def extrair_mao_obra_e_equipamentos(arquivos_pdf: Iterable[Any]) -> pd.DataFrame:
    frames = parse_many(arquivos_pdf)
    mo = frames["Mao_Obra"].copy()
    eqp = frames["Equipamentos"].copy()
    if not mo.empty:
        mo.insert(3, "Tipo", "MAO DE OBRA")
    if not eqp.empty:
        eqp.insert(3, "Tipo", "EQUIPAMENTO")
    return pd.concat([mo, eqp], ignore_index=True, sort=False)


def extrair_atividades(arquivos_pdf: Iterable[Any]) -> pd.DataFrame:
    return parse_many(arquivos_pdf)["Atividades"]


def extrair_comentarios_rdo(arquivos_pdf: Iterable[Any]) -> pd.DataFrame:
    frames = parse_many(arquivos_pdf)
    comentarios = frames["Comentarios"].copy()
    ocorrencias = frames["Ocorrencias"].copy()

    rows = []
    if not ocorrencias.empty:
        for _, row in ocorrencias.iterrows():
            rows.append({
                "N RDO": row.get("N RDO", ""),
                "Data": row.get("Data", ""),
                "Responsavel pelo Comentario": "",
                "Classificacao": "IMPACTO",
                "Comentario": row.get("Ocorrencia", ""),
                "Tags": row.get("Tags", ""),
                "Duracao": row.get("Duracao", ""),
                "Nome do arquivo": row.get("Arquivo", ""),
            })

    if not comentarios.empty:
        for _, row in comentarios.iterrows():
            rows.append({
                "N RDO": row.get("N RDO", ""),
                "Data": row.get("Data", ""),
                "Responsavel pelo Comentario": row.get("Responsavel pelo comentario", ""),
                "Classificacao": "INFORMACAO",
                "Comentario": row.get("Comentario", ""),
                "Tags": "",
                "Duracao": "",
                "Nome do arquivo": row.get("Arquivo", ""),
            })

    return pd.DataFrame(rows)


# ============================================================
# STREAMLIT UI
# ============================================================


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run: pip install streamlit")
    st.set_page_config(page_title="Parser RDO - App Diario de Obra", layout="wide")
    st.title("Parser de RDO - App Diario de Obra / RDO Digital")
    st.caption(
        "Extrai cabecalho, clima, mao de obra, equipamentos, atividades, ocorrencias, "
        "comentarios, aprovacoes e trilha de auditoria."
    )

    files = st.file_uploader(
        "Selecione um ou mais PDFs de RDO",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not files:
        st.info("Envie ao menos um PDF para iniciar.")
        return

    if not st.button("Processar RDOs", type="primary"):
        return

    with st.spinner("Lendo e estruturando os RDOs..."):
        try:
            frames = parse_many(files)
        except Exception as exc:
            st.exception(exc)
            return

    summary = frames["Resumo"]
    if summary.empty:
        st.warning("Nenhum RDO foi reconhecido.")
        return

    st.subheader("Identificacao do modelo")
    st.dataframe(
        summary[["Arquivo", "Software detectado", "Fabricante", "Confianca modelo", "N RDO", "Data"]],
        use_container_width=True,
        hide_index=True,
    )

    tab_names = [
        "Resumo",
        "Clima",
        "Mao_Obra",
        "Resumo_MO",
        "Equipamentos",
        "Atividades",
        "Ocorrencias",
        "Comentarios",
        "Aprovacoes",
        "Debug",
    ]
    tabs = st.tabs(tab_names)
    for tab, name in zip(tabs, tab_names):
        with tab:
            df = frames[name]
            if df.empty:
                st.info("Nenhum registro extraido nesta secao.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)

    excel = frames_to_excel(frames)
    st.download_button(
        "Baixar consolidado em Excel",
        data=excel,
        file_name="consolidado_rdo_app_diario_obra.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
