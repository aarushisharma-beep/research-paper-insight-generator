import os
import re
import pandas as pd

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = os.environ.get(
    "INPUT_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "input"
    )
)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output"
    )
)
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "research_paper_insights.csv"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):

    doc = fitz.open(pdf_path)

    pages_text = []

    for page in doc:

        text = page.get_text()

        # Preserve hyphens when words are split across lines
        text = re.sub(r'-\s*\n\s*', '-', text)

        # Keep enough structure for metadata extraction
        text = re.sub(r'\n+', '\n', text)

        # Remove excessive spaces
        text = re.sub(r'[ \t]+', ' ', text).strip()

        pages_text.append(text)

    full_text = "\n".join(pages_text)

    return full_text, len(doc)


# ============================================================
# ABSTRACT EXTRACTION
# ============================================================

def extract_abstract(text):

    match = re.search(
        r'\bABSTRACT\b\s*:?\s*(.*?)(?=\n\s*(?:1\.?\s+INTRODUCTION|INTRODUCTION|Keywords?|Index Terms?|CCS CONCEPTS)\b)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        abstract = match.group(1).strip()

        abstract = re.sub(r'\s+', ' ', abstract)

        if len(abstract) > 100:
            return abstract

    # Fallback
    match = re.search(
        r'\bABSTRACT\b\s*:?\s*(.*)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        abstract = match.group(1).strip()

        abstract = re.split(
            r'\n\s*(?:Keywords?|Index Terms?|1\.?\s+INTRODUCTION|INTRODUCTION)\b',
            abstract,
            flags=re.IGNORECASE
        )[0]

        abstract = re.sub(
            r'\s+',
            ' ',
            abstract
        ).strip()

        if len(abstract) > 100:
            return abstract

    return "Not available"


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(text):

    match = re.search(
        r'(?:Index Terms?|Keywords?)\s*[:\-]?\s*(.*?)(?=\n\s*(?:1\.?\s+INTRODUCTION|INTRODUCTION|I\.?\s+INTRODUCTION)\b)',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:

        keywords = match.group(1).strip()

        keywords = re.sub(
            r'\s+',
            ' ',
            keywords
        )

        keywords = keywords.strip(" .;:")

        if keywords:
            return keywords

    return "Not provided"


# ============================================================
# EXTRACTIVE SUMMARY
# ============================================================

def generate_summary(text, num_sentences=3):

    clean_text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    sentences = re.split(
        r'(?<=[.!?])\s+',
        clean_text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip().split()) >= 8
    ]

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        matrix = vectorizer.fit_transform(sentences)

        document_vector = matrix.mean(axis=0)

        similarities = cosine_similarity(
            matrix,
            document_vector
        ).flatten()

        top_indices = similarities.argsort()[-num_sentences:]

        top_indices = sorted(top_indices)

        summary = " ".join(
            sentences[i]
            for i in top_indices
        )

        return summary

    except Exception:

        return " ".join(
            sentences[:num_sentences]
        )


# ============================================================
# TF-IDF KEYWORDS
# ============================================================

def extract_tfidf_keywords(text, top_n=8):

    clean_text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    if not clean_text:
        return "Not available"

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        matrix = vectorizer.fit_transform(
            [clean_text]
        )

        scores = matrix.toarray()[0]

        terms = vectorizer.get_feature_names_out()

        ranked_indices = scores.argsort()[::-1]

        keywords = [
            terms[i]
            for i in ranked_indices[:top_n]
            if scores[i] > 0
        ]

        return ", ".join(keywords)

    except Exception:

        return "Not available"


# ============================================================
# TITLE + AUTHOR EXTRACTION
# ============================================================

AFFIL_WORDS = {
    "university",
    "universität",
    "université",
    "institute",
    "institut",
    "department",
    "dept",
    "school",
    "laboratory",
    "lab",
    "college",
    "academy",
    "center",
    "centre",
    "faculty",
    "cnrs",
    "corp",
    "inc",
    "ltd",
    "gmbh",
    "research",
    "foundation",
    "hospital",
    "technology",
    "technical",
    "sciences",
    "group"
}

PARTICLES = {
    "de",
    "van",
    "von",
    "der",
    "den",
    "la",
    "le",
    "da",
    "di",
    "del",
    "bin",
    "al",
    "el",
    "ben"
}

ACRONYMS = {
    "ai",
    "rl",
    "llm",
    "llms",
    "nlp",
    "ml",
    "gpu",
    "t2i",
    "mis",
    "sar"
}

NAME_TOKEN = re.compile(
    r"^(?:[A-ZÀ-ÖØ-Þ][A-Za-zà-öø-ÿ'’\-]*[a-zà-öø-ÿ]|[A-ZÀ-ÖØ-Þ]\.?)$"
)


def _line_text(line, max_size):

    output = []

    previous_x1 = None

    for span in line["spans"]:

        chars = span["chars"]

        span_text = "".join(
            char["c"]
            for char in chars
        )

        is_superscript = (
            bool(span["flags"] & 1)
            or (
                span["size"] < 0.72 * max_size
                and len(span_text.strip()) <= 4
            )
        )

        if is_superscript:

            if previous_x1 is not None:
                previous_x1 += 999

            continue

        for char in chars:

            c = char["c"]

            if c == " ":

                output.append(" ")
                previous_x1 = char["bbox"][2]
                continue

            if (
                previous_x1 is not None
                and char["bbox"][0] - previous_x1
                > 0.15 * span["size"]
                and output
                and output[-1] != " "
            ):
                output.append(" ")

            output.append(c)

            previous_x1 = char["bbox"][2]

    return re.sub(
        r"\s+",
        " ",
        "".join(output)
    ).strip()


def get_first_page_lines(pdf_path):

    page = fitz.open(pdf_path)[0]

    height = page.rect.height

    raw = []

    for block in page.get_text("rawdict")["blocks"]:

        if block.get("type") != 0:
            continue

        for line in block["lines"]:

            if abs(line["dir"][0]) < 0.99:
                continue

            spans = [
                span
                for span in line["spans"]
                if "".join(
                    char["c"]
                    for char in span["chars"]
                ).strip()
            ]

            if not spans:
                continue

            if line["bbox"][1] > 0.6 * height:
                continue

            max_size = max(
                span["size"]
                for span in spans
            )

            text = _line_text(
                line,
                max_size
            )

            if text:

                raw.append({
                    "text": text,
                    "size": max_size,
                    "y0": line["bbox"][1],
                    "y1": line["bbox"][3],
                    "x0": line["bbox"][0]
                })

    raw.sort(
        key=lambda item: item["y0"]
    )

    rows = []

    for line in raw:

        if rows:

            row = rows[-1]

            overlap = (
                min(row["y1"], line["y1"])
                - max(row["y0"], line["y0"])
            )

            if overlap >= 0.5 * min(
                row["y1"] - row["y0"],
                line["y1"] - line["y0"]
            ):

                row["parts"].append(line)

                row["y0"] = min(
                    row["y0"],
                    line["y0"]
                )

                row["y1"] = max(
                    row["y1"],
                    line["y1"]
                )

                continue

        rows.append({
            "y0": line["y0"],
            "y1": line["y1"],
            "parts": [line]
        })

    lines = []

    for row in rows:

        parts = sorted(
            row["parts"],
            key=lambda item: item["x0"]
        )

        lines.append({
            "text": " ".join(
                part["text"]
                for part in parts
            ),
            "size": max(
                part["size"]
                for part in parts
            ),
            "y0": row["y0"],
            "y1": row["y1"],
            "x0": parts[0]["x0"]
        })

    return lines


def find_title_lines(lines):

    groups = []

    current = []

    for line in lines:

        if current and (
            line["size"]
            >= 0.75 * max(
                item["size"]
                for item in current
            )
            and
            line["y0"] - current[-1]["y1"]
            < 0.6 * line["size"]
        ):

            current.append(line)

        else:

            if current:
                groups.append(current)

            current = [line]

    if current:
        groups.append(current)

    valid_groups = [
        group
        for group in groups
        if sum(
            len(line["text"].split())
            for line in group
        ) >= 3
        and not re.search(
            r"arxiv|preprint|under review",
            " ".join(
                line["text"]
                for line in group
            ),
            re.IGNORECASE
        )
    ]

    return max(
        valid_groups,
        key=lambda group: max(
            line["size"]
            for line in group
        ),
        default=[]
    )


def smart_title(text):

    output = []

    for index, word in enumerate(
        text.lower().split(" ")
    ):

        parts = []

        for part in word.split("-"):

            if part in ACRONYMS:
                parts.append(part.upper())

            elif index > 0 and part in {
                "a",
                "an",
                "the",
                "and",
                "or",
                "of",
                "for",
                "via",
                "in",
                "on",
                "to",
                "with",
                "by",
                "at",
                "vs",
                "from",
                "as"
            }:
                parts.append(part)

            else:
                parts.append(
                    part[:1].upper()
                    + part[1:]
                )

        output.append(
            "-".join(parts)
        )

    return " ".join(output)


def join_title(title_lines):

    output = ""

    for line in title_lines:

        text = line["text"]

        if output.endswith("-"):
            output += text
        elif output:
            output += " " + text
        else:
            output = text

    output = re.sub(
        r"\s+",
        " ",
        output
    ).strip()

    if output.isupper():
        return smart_title(output)

    return output


def split_names(text):

    text = re.sub(
        r"\[\s*\d{4}-\d{4}-\d{4}-\d{3}[\dXx]\s*\]",
        "",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    text = re.sub(
        r"[∗*†‡§¶#⋆¹²³⁴⁵⁶⁷⁸⁹⁰]",
        "",
        text
    )

    text = re.sub(
        r"(?<=[^\W\d_])\d+(?:\s*,\s*\d+)*",
        "",
        text
    )

    text = re.sub(
        r"\s+(?:and|&)\s+",
        ", ",
        text
    )

    return [
        part.strip(" ;")
        for part in text.split(",")
        if part.strip(" ;")
    ]


def is_name(part):

    tokens = part.split()

    if not 2 <= len(tokens) <= 5:
        return False

    if any(
        token.lower().strip(".,")
        in AFFIL_WORDS
        for token in tokens
    ):
        return False

    return all(
        NAME_TOKEN.match(token)
        or token.lower() in PARTICLES
        for token in tokens
    )


def extract_authors(lines, title_lines):

    after_title = [
        line
        for line in lines
        if line["y0"]
        >= title_lines[-1]["y1"] - 1
    ]

    block = ""
    previous = ""

    for line in after_title[:10]:

        text = line["text"]

        if re.match(
            r"(?i)^(abstract|keywords|index terms|1\.?\s*introduction|introduction)",
            text
        ):
            break

        continues = previous.endswith("-")

        has_name = any(
            is_name(part)
            for part in split_names(text)
        )

        if not (has_name or continues):

            if block:
                break

            continue

        if continues:

            block = block[:-1] + text

        else:

            block = (
                block + " " + text
            ).strip()

        previous = text

    names = [
        part
        for part in split_names(block)
        if is_name(part)
    ]

    return ", ".join(
        dict.fromkeys(names)
    ) or "Not available"


def extract_title_authors(pdf_path):

    lines = get_first_page_lines(
        pdf_path
    )

    title_lines = find_title_lines(
        lines
    )

    if title_lines:

        return (
            join_title(title_lines),
            extract_authors(
                lines,
                title_lines
            )
        )

    metadata = (
        fitz.open(pdf_path).metadata
        or {}
    )

    return (
        metadata.get("title")
        or "Not available",
        metadata.get("author")
        or "Not available"
    )


# ============================================================
# PREPRINT DATE
# ============================================================

def extract_preprint_date(text):

    pattern = (
        r"arXiv:\S+.*?"
        r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1)

    return "Not available"


# ============================================================
# PROCESS ONE PDF
# ============================================================

def process_pdf(pdf_path, paper_id):

    print(f"Processing {paper_id}...")

    text, page_count = extract_pdf_text(
        pdf_path
    )

    abstract = extract_abstract(
        text
    )

    keywords = extract_keywords(
        text
    )

    summary = generate_summary(
        abstract
    )

    tfidf_keywords = extract_tfidf_keywords(
        abstract
    )

    title, authors = extract_title_authors(
        pdf_path
    )

    preprint_date = extract_preprint_date(
        text
    )

    word_count = len(
        text.split()
    )

    return {
        "Paper_ID": paper_id,
        "File_Name": os.path.basename(
            pdf_path
        ),
        "Title": title,
        "Preprint_Date": preprint_date,
        "Publication_Year": None,
        "Abstract": abstract,
        "Keywords": keywords,
        "Pages": page_count,
        "Word_Count": word_count,
        "Authors": authors,
        "Summary": summary,
        "TFIDF_Keywords": tfidf_keywords
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if not os.path.exists(INPUT_DIR):

        print(
            f"Input folder not found: {INPUT_DIR}"
        )

        return

    pdf_paths = sorted(
        [
            os.path.join(
                INPUT_DIR,
                filename
            )
            for filename in os.listdir(
                INPUT_DIR
            )
            if filename.lower().endswith(".pdf")
        ]
    )

    if not pdf_paths:

        print(
            f"No PDF files found in {INPUT_DIR}"
        )

        return

    records = []

    for index, pdf_path in enumerate(
        pdf_paths,
        start=1
    ):

        paper_id = f"RP{index}"

        try:

            record = process_pdf(
                pdf_path,
                paper_id
            )

            records.append(record)

            print(
                f"✓ {paper_id} completed"
            )

        except Exception as error:

            print(
                f"✗ {paper_id} failed: {error}"
            )

    if not records:

        print(
            "No papers were processed successfully."
        )

        return

    papers_df = pd.DataFrame(
        records
    )

    # Basic quality checks
    print("\n===== DATA QUALITY CHECK =====")

    print(
        "Number of papers:",
        len(papers_df)
    )

    print(
        "Number of columns:",
        len(papers_df.columns)
    )

    print(
        "Duplicate Paper IDs:",
        papers_df["Paper_ID"].duplicated().sum()
    )

    print(
        "Empty abstracts:",
        (
            papers_df["Abstract"]
            .fillna("")
            .str.strip()
            == ""
        ).sum()
    )

    print(
        "Empty summaries:",
        (
            papers_df["Summary"]
            .fillna("")
            .str.strip()
            == ""
        ).sum()
    )

    # Save CSV
    papers_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== OUTPUT =====")

    print(
        "CSV saved successfully:"
    )

    print(
        OUTPUT_FILE
    )

    print("\nProcessed papers:")

    print(
        papers_df[
            [
                "Paper_ID",
                "File_Name",
                "Title",
                "Authors"
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()