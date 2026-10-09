from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    id: int
    content: str
    section: str
    chapter: str
    url: str
    version: str
    score: float = 0.0

    def public(self) -> dict:
        return {
            "id": self.id,
            "section": self.section,
            "chapter": self.chapter,
            "url": self.url,
            "version": self.version,
            "score": round(self.score, 4),
            "snippet": self.content[:240],
        }