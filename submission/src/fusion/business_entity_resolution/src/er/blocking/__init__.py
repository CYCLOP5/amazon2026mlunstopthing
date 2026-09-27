'candidate generation methods'
from er.registry import Registry

BLOCKERS = Registry("blocker")

from er.blocking import skeleton_tfidf  # noqa: E402,F401  (registers "skeleton_tfidf")
