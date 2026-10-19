"""Human-in-the-loop production: an analyst agent proposes player insights, a producer approves them.

Event data only (no video, no tracking). See :mod:`.analyst`, :mod:`.queue` and :mod:`.session`.
"""

from .analyst import AnalystAgent, Proposal
from .queue import ApprovalQueue
from .session import LiveSession

__all__ = ["AnalystAgent", "ApprovalQueue", "LiveSession", "Proposal"]
