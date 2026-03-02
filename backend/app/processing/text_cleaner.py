import re

def clean_text(text: str) -> str:
    """Full cleaning pipeline for extracted text."""
    if not text:
        return ""
    text = _remove_page_artifacts(text)
    text = _fix_hyphenation(text)
    text = _standardize_bullets(text)
    text = _normalize_whitespace(text)
    return text.strip()

def clean_transcription(text: str) -> str:
    """Clean audio transcription: remove fillers and annotations."""
    if not text:
        return ""
    text = re.sub(r'\b(um|uh|eh|ah|hmm|hm)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[.*?\]', '', text)  
    return clean_text(text)

def _remove_page_artifacts(text: str) -> str:
    """Remove PDF headers, footers, page numbers."""
    text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*(?:of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
    return text

def _fix_hyphenation(text: str) -> str:
    """Fix words broken across lines: back-\npropagation → backpropagation."""
    return re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

def _standardize_bullets(text: str) -> str:
    """Normalize bullet characters to dashes."""
    for char in '•●▪▸►':
        text = text.replace(char, '- ')
    for char in '◦○':
        text = text.replace(char, '  - ')
    return text

def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace, preserve paragraph breaks."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text