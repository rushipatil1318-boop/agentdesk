"""Tiny policy knowledge base: paragraph chunks + TF-IDF retrieval (no external services)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_ID_RE = re.compile(r"^\*\*([A-Z]+-\d+)\.\*\*\s*(.*)$", re.S)
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Domain synonyms folded into one term, so "return"/"money back" find the refund policy.
_SYNONYMS = {"return": "refund", "reimburse": "refund", "reimbursement": "refund", "automatic": "auto",
             "automatically": "auto", "days": "day"}


def _stem(token: str) -> str:
    token = _SYNONYMS.get(token, token)
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return _SYNONYMS.get(token, token)


def analyze(text: str) -> list[str]:
    """Lowercase, drop stop words, light suffix stemming (no external NLP dependency)."""
    tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]
    return [_stem(t) for t in tokens]


@dataclass(frozen=True)
class Passage:
    id: str
    source: str
    text: str


class PolicyKnowledgeBase:
    def __init__(self, directory: Path | str = DEFAULT_DIR) -> None:
        self.passages: list[Passage] = []
        for path in sorted(Path(directory).glob("*.md")):
            for block in path.read_text(encoding="utf-8").split("\n\n"):
                match = _ID_RE.match(block.strip())
                if match:
                    self.passages.append(Passage(match.group(1), path.name, match.group(2).strip()))
        if not self.passages:
            raise ValueError(f"No policy passages found in {directory}")
        texts = [p.text for p in self.passages]
        # Hybrid lexical retrieval: stemmed word TF-IDF + character n-gram TF-IDF (robust to word forms)
        self._word = TfidfVectorizer(analyzer=analyze)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self._word_matrix = self._word.fit_transform(texts)
        self._char_matrix = self._char.fit_transform(texts)

    def search(self, query: str, k: int = 3, min_score: float = 0.08) -> list[dict]:
        scores = 0.5 * cosine_similarity(self._word.transform([query]), self._word_matrix).ravel() \
            + 0.5 * cosine_similarity(self._char.transform([query]), self._char_matrix).ravel()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"id": self.passages[i].id, "source": self.passages[i].source,
             "text": self.passages[i].text, "score": round(float(scores[i]), 3)}
            for i in ranked
            if scores[i] >= min_score
        ]
