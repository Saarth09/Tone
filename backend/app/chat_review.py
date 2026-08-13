from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from app.detector.cosine import cosine_similarity, drift_from_similarity

logger = logging.getLogger(__name__)

USER_RE = re.compile(
    r"^(?:User|Human|You|Prompt|Customer)\s*[:\-]\s*(.*)$",
    re.IGNORECASE,
)
ASSISTANT_RE = re.compile(
    r"^(?:Assistant|AI|ChatGPT|Claude|Gemini|Bot|Model)\s*[:\-]\s*(.*)$",
    re.IGNORECASE,
)
MARKDOWN_HDR_RE = re.compile(
    r"^#{1,3}\s+(User|Human|Assistant|AI|ChatGPT|Claude)\b", re.I
)


@dataclass
class Turn:
    index: int
    role: str  # user | assistant
    content: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def parse_chatgpt_export(data: Any) -> list[Turn]:
    """Parse ChatGPT export JSON (conversation mapping or messages list)."""
    turns: list[Turn] = []

    def add(role: str, content: str) -> None:
        content = _clean(content)
        if not content:
            return
        role_l = role.lower()
        if role_l in {"user", "human"}:
            role_norm = "user"
        elif role_l in {"assistant", "model", "tool"}:
            role_norm = "assistant"
        else:
            return
        turns.append(Turn(index=len(turns), role=role_norm, content=content))

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            author = item.get("author")
            if isinstance(author, dict):
                role = str(author.get("role") or "")
            else:
                role = str(item.get("role") or "")
            content = item.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        parts.append(str(p.get("text") or p.get("content") or ""))
                content = "\n".join(parts)
            elif isinstance(content, dict):
                parts = content.get("parts") or content.get("text")
                if isinstance(parts, list):
                    content = "\n".join(str(x) for x in parts)
                else:
                    content = str(parts or "")
            add(role, str(content or ""))
        return turns

    if isinstance(data, dict):
        mapping = data.get("mapping")
        if isinstance(mapping, dict):
            nodes = []
            for node in mapping.values():
                if not isinstance(node, dict):
                    continue
                msg = node.get("message")
                if not msg:
                    continue
                author = (msg.get("author") or {}).get("role") or ""
                content = msg.get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                text = "\n".join(str(p) for p in parts) if isinstance(parts, list) else ""
                create_time = msg.get("create_time") or 0
                nodes.append((create_time or 0, author, text))
            nodes.sort(key=lambda x: x[0])
            for _, author, text in nodes:
                add(str(author), text)
            if turns:
                return turns

        if isinstance(data.get("messages"), list):
            return parse_chatgpt_export(data["messages"])
        if isinstance(data.get("conversations"), list) and data["conversations"]:
            return parse_chatgpt_export(data["conversations"][0])

    return turns


def parse_plain_transcript(text: str) -> list[Turn]:
    """Parse User:/Assistant: style transcripts and loose markdown chats."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    turns: list[Turn] = []
    current_role: Optional[str] = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current_role
        if current_role and buf:
            content = _clean("\n".join(buf))
            if content:
                turns.append(Turn(index=len(turns), role=current_role, content=content))
        buf = []

    for raw in lines:
        line = raw.strip()
        if not line:
            if buf:
                buf.append("")
            continue

        hdr = MARKDOWN_HDR_RE.match(line)
        if hdr:
            flush()
            label = hdr.group(1).lower()
            current_role = "user" if label in {"user", "human"} else "assistant"
            rest = line[hdr.end() :].lstrip(" #:|-")
            buf = [rest] if rest else []
            continue

        um = USER_RE.match(line)
        if um:
            flush()
            current_role = "user"
            buf = [um.group(1)] if um.group(1) else []
            continue

        am = ASSISTANT_RE.match(line)
        if am:
            flush()
            current_role = "assistant"
            buf = [am.group(1)] if am.group(1) else []
            continue

        if current_role is None:
            current_role = "user"
        buf.append(raw)

    flush()

    if len(turns) < 2:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if len(blocks) >= 2:
            turns = []
            for i, block in enumerate(blocks):
                role = "user" if i % 2 == 0 else "assistant"
                turns.append(Turn(index=i, role=role, content=_clean(block)))
    return turns


def parse_transcript(raw: str) -> list[Turn]:
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw[0] in "[{":
        try:
            data = json.loads(raw)
            turns = parse_chatgpt_export(data)
            if turns:
                return turns
        except json.JSONDecodeError:
            pass
    return parse_plain_transcript(raw)


def infer_goal(turns: list[Turn], explicit: Optional[str] = None) -> str:
    if explicit and explicit.strip():
        return _clean(explicit)
    for t in turns:
        if t.role == "user" and len(t.content) >= 8:
            return t.content[:2000]
    return turns[0].content[:2000] if turns else ""


def _window_texts(turns: list[Turn]) -> list[tuple[int, int, str, str, str, str]]:
    """
    Build analysis windows over assistant turns.
    Returns (start, end, label, embed_text, user_text, assistant_text).
    """
    assistant_idxs = [i for i, t in enumerate(turns) if t.role == "assistant"]
    if not assistant_idxs:
        blob = "\n".join(t.content for t in turns)
        return [(0, max(0, len(turns) - 1), "full transcript", blob, blob, "")]

    windows: list[tuple[int, int, str, str, str, str]] = []
    for n, a_idx in enumerate(assistant_idxs):
        start = a_idx
        user_text = ""
        if a_idx > 0 and turns[a_idx - 1].role == "user":
            start = a_idx - 1
            user_text = turns[a_idx - 1].content
        assistant_text = turns[a_idx].content
        chunk = turns[start : a_idx + 1]
        text = "\n".join(f"{t.role}: {t.content}" for t in chunk)
        label = f"Exchange {n + 1} (msgs {start + 1}–{a_idx + 1})"
        windows.append((start, a_idx, label, text, user_text, assistant_text))
    return windows


def rule_tips(
    *,
    goal: str,
    peak_label: str,
    peak_score: float,
    peak_excerpt: str,
    overall: float,
) -> list[str]:
    tips: list[str] = []
    user_snip = ""
    for line in (peak_excerpt or "").split("\n"):
        if line.lower().startswith("user:"):
            user_snip = _clean(line.split(":", 1)[-1])[:100]
            break

    if overall < 0.25:
        tips.append(
            "Low drift — pin the current good behavior with assert_semantically_equals "
            "on a short realistic user ask from this chat (not the whole goal dump)."
        )
        tips.append(
            "Add assert_tone_matches(persona='helpful, focused, on-task') so future "
            "prompt edits cannot quietly harden the tone."
        )
    else:
        where = f" around {peak_label}" if peak_label else ""
        ask = f' User asked: "{user_snip}…".' if user_snip else ""
        tips.append(
            f"Drift peaked{where} (score {peak_score:.2f}).{ask} "
            "Pin the intended reply with assert_semantically_equals using a short gold answer."
        )
        tips.append(
            "If replies got terse, cold, or overly verbose, lock tone with "
            "assert_tone_matches(persona='empathetic, calm, solution-focused')."
        )
        tips.append(
            "If the model abandoned the ask or expanded scope, add "
            "assert_semantically_excludes against that failure mode "
            '(e.g. concept="I cannot help with that" or "ignore the original request").'
        )
    if len(goal) > 400:
        tips.append(
            "Goal text is long — use a short user probe in tests "
            '(e.g. the last question in the chat), not the full brief as suite.query(...).'
        )
    tips.append(
        "After Generate llmtest, edit baselines to the answer you want forever, "
        "then run llmtest --baseline and fail CI on drift."
    )
    return tips[:6]


_ASSERT_TIP_NAMES = (
    "assert_semantically_equals",
    "assert_tone_matches",
    "assert_semantically_excludes",
)
_BAD_TIP_RE = re.compile(
    r"(?i)\b("
    r"benefits of using ai|risks of using ai|casual summary|solicit feedback|"
    r"clarify your focus|key benefits|reinforce the value|marketing|"
    r"explain it like i'?m five|how this works simply|core functionality as described|"
    r"responds? directly to user queries|maintain user trust|what tone should the ai use|"
    r"what should it not include|can you explain how this works"
    r")\b"
)


def _split_tip_blob(text: str) -> list[str]:
    """Split one mega tip paragraph into separate tips when the model ignores newlines."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) < 220:
        return [text]
    parts = re.split(
        r"(?<=[.!?])\s+(?=(?:"
        r"Use |Apply |Implement |Another |Additionally |Finally |If |"
        r"Pin |Add |Lock "
        r"))",
        text,
    )
    if len(parts) == 1:
        parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 40]


def filter_assertion_tips(tips: Optional[list[str]]) -> list[str]:
    """Keep only tips that name an llmtest assertion and aren't generic filler."""
    out: list[str] = []
    for tip in tips or []:
        for chunk in _split_tip_blob(tip):
            if _BAD_TIP_RE.search(chunk):
                continue
            if not any(name in chunk for name in _ASSERT_TIP_NAMES):
                continue
            out.append(chunk)
            if len(out) >= 4:
                return out
    return out


async def maybe_llm_tips(
    *,
    llm_complete,
    goal: str,
    peak_label: str,
    peak_score: float,
    peak_excerpt: str,
    overall: float,
) -> Optional[list[str]]:
    if llm_complete is None:
        return None
    prompt = (
        "You write tips for developers pinning THIS chat's AI behavior with llmtest.\n"
        "Return exactly 3 tips, each on its own line (no markdown, no numbering).\n"
        "Each tip MUST include ALL of:\n"
        "  - a failure mode visible in THIS transcript (not generic AI advice)\n"
        "  - exactly one of: assert_semantically_equals | assert_tone_matches | assert_semantically_excludes\n"
        "  - a SHORT probe a real user would type (under 12 words), quoted\n"
        "Forbidden: marketing language, 'benefits of AI', 'risks of AI', 'casual summary', "
        "'clarify your focus', 'solicit feedback', or tips unrelated to the goal below.\n\n"
        f"ORIGINAL GOAL:\n{goal[:1500]}\n\n"
        f"OVERALL DRIFT SCORE: {overall:.2f} (0=on track, 1=fully drifted)\n"
        f"PEAK LOCATION: {peak_label} (score {peak_score:.2f})\n"
        f"PEAK EXCERPT:\n{peak_excerpt[:1200]}\n\n"
        "Tips:"
    )
    try:
        text, _ = await llm_complete(prompt)
    except Exception:
        logger.exception("LLM tips failed")
        return None
    lines: list[str] = []
    for line in (text or "").splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    filtered = filter_assertion_tips(lines)
    return filtered or None


def analyze_turns(
    turns: list[Turn],
    *,
    embed_fn,
    goal: str,
    threshold: float = 0.45,
) -> dict[str, Any]:
    windows = _window_texts(turns)
    # Cap extremely long chats to keep CPU bounded (sample evenly if needed)
    max_windows = 80
    if len(windows) > max_windows:
        idxs = np.linspace(0, len(windows) - 1, max_windows).astype(int)
        windows = [windows[i] for i in sorted(set(idxs.tolist()))]

    texts = [w[3] for w in windows]
    vectors = embed_fn([goal] + texts)
    goal_vec = vectors[0]
    win_vecs = vectors[1:]

    points: list[dict[str, Any]] = []
    for (start, end, label, text, user_text, assistant_text), vec in zip(windows, win_vecs):
        sim = cosine_similarity(goal_vec, vec)
        score = drift_from_similarity(sim)
        points.append(
            {
                "turn_index": end,
                "window_start": start,
                "window_end": end,
                "label": label,
                "drift_score": round(float(score), 4),
                "similarity": round(float(sim), 4),
                "is_alert": bool(score >= threshold),
                "excerpt": text[:320] + ("…" if len(text) > 320 else ""),
                "user_text": (user_text or "")[:800],
                "assistant_text": (assistant_text or "")[:800],
            }
        )

    scores = [p["drift_score"] for p in points] or [0.0]
    overall = float(np.mean(scores))
    if len(scores) >= 3:
        w = np.linspace(0.6, 1.0, len(scores))
        overall = float(np.average(scores, weights=w))

    peak = max(points, key=lambda p: p["drift_score"]) if points else None
    first_alert = next((p for p in points if p["is_alert"]), None)

    tips = rule_tips(
        goal=goal,
        peak_label=(peak or {}).get("label", "n/a"),
        peak_score=float((peak or {}).get("drift_score", 0.0)),
        peak_excerpt=(peak or {}).get("excerpt", ""),
        overall=overall,
    )

    return {
        "goal": goal,
        "message_count": len(turns),
        "assistant_turns": sum(1 for t in turns if t.role == "assistant"),
        "user_turns": sum(1 for t in turns if t.role == "user"),
        "threshold": threshold,
        "overall_drift": round(overall, 4),
        "is_alert": overall >= threshold or any(p["is_alert"] for p in points),
        "peak": peak,
        "first_alert": first_alert,
        "timeline": points,
        "tips": tips,
        "tips_source": "rules",
    }


def _py_str(s: str, limit: int = 400) -> str:
    text = (s or "").strip().replace("\\", "\\\\").replace('"', '\\"')
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _user_assistant_pairs(turns: list[Turn]) -> list[dict[str, Any]]:
    """Pair each user turn with the following assistant reply (if any)."""
    pairs: list[dict[str, Any]] = []
    i = 0
    while i < len(turns):
        t = turns[i]
        if t.role != "user":
            i += 1
            continue
        asst: Optional[Turn] = None
        end = i
        if i + 1 < len(turns) and turns[i + 1].role == "assistant":
            asst = turns[i + 1]
            end = i + 1
        pairs.append(
            {
                "user": t.content,
                "assistant": asst.content if asst else "",
                "window_start": t.index,
                "window_end": end,
            }
        )
        i = end + 1
    return pairs


def _role_from_excerpt(excerpt: str, role: str) -> str:
    """Pull `user:` / `assistant:` body from a window excerpt."""
    role_l = role.lower()
    for line in (excerpt or "").split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith(f"{role_l}:"):
            return _clean(stripped.split(":", 1)[-1])
    return ""


def _short_probe(user_text: str, goal: str, *, max_len: int = 100) -> str:
    """
    Turn a (possibly long) user message into a short, realistic suite.query(...) probe.
    Prefer the actual ask / last question over dumping the whole brief.
    """
    text = _clean(user_text)
    candidates: list[str] = []
    ask_re = re.compile(
        r"(?i)(?:expand upon|please\s+(?:help|expand|explain|add)|can you|could you|"
        r"how (?:do|can|should) (?:i|we)|what (?:is|are|should))"
        r"[^.?!\n]*[?.!]?"
    )
    for m in ask_re.finditer(text):
        chunk = _clean(m.group(0) or "")
        if len(chunk) >= 12:
            candidates.append(chunk)
    for m in re.finditer(r"([^.?!\n]{8,180}\?)", text):
        candidates.append(_clean(m.group(1)))

    for candidate in reversed(candidates):
        if len(candidate) >= 12:
            return _polish_probe(candidate, max_len=max_len)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    for s in reversed(sentences):
        if "?" in s and len(s) >= 12:
            return _polish_probe(s, max_len=max_len)

    paras = [p.strip() for p in re.split(r"\n\s*\n", user_text) if p.strip()]
    if paras:
        last = _clean(paras[-1])
        if len(last) >= 12:
            return _polish_probe(last, max_len=max_len)

    if len(text) <= max_len:
        return text
    g = _clean(goal)
    if 20 <= len(g) <= max_len:
        return g
    return text[: max_len - 1] + "…"


def _polish_probe(probe: str, *, max_len: int = 100) -> str:
    """Normalize awkward extracted asks into short CI probes."""
    p = _clean(probe)
    m = re.match(r"(?i)^expand upon this\s+(?:as\s+)?(.+)$", p)
    if m:
        rest = _clean(m.group(1))
        # Prefer domain keywords if present
        if re.search(r"(?i)\b(tone|llmtest|drift|chat)\b", rest):
            p = "Expand on how Tone chat-review drift should feed llmtest"
        else:
            p = f"Expand on this: {rest}"
    if len(p) > max_len:
        return p[: max_len - 1] + "…"
    return p


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*|__|`+|#+", " ", text or "")
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return _clean(text)


def _first_sentences(text: str, *, n: int = 2, max_len: int = 280) -> str:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", _clean(text)) if s.strip()]
    joined = " ".join(parts[:n]).strip() or _clean(text)
    if len(joined) > max_len:
        return joined[: max_len - 1] + "…"
    return joined


_MECHANISM_RE = re.compile(
    r"(?i)\b("
    r"embedding|cosine|threshold|baseline|assert_[a-z_]+|github action|"
    r"semantic(?:ally)?|snapshot|sentence-transformers|per-conversation|"
    r"feedback loop|system prompt|fails? the (?:pr|merge|ci)|blocks? the merge|"
    r"chat-?review|llmtest|mmd|kl divergence"
    r")\b"
)

_GENERIC_BASELINE_RE = re.compile(
    r"(?i)\b("
    r"consistent and effective|remains consistent|important for (?:code|testing)|"
    r"ensure[s]? ai behavior|maintain(?:s|ing)? (?:focus|user trust)|"
    r"responds? directly|avoid ambiguity|stays? on (?:track|task)|"
    r"helpful and accurate|high[- ]quality (?:answers|responses)"
    r")\b"
)

_GENERIC_PROBE_RE = re.compile(
    r"(?i)^(?:"
    r"why is unit testing important|"
    r"how does this (?:work|plugin work)|"
    r"what tone should|"
    r"can you explain how this works|"
    r"what should it not include|"
    r"what is the purpose of this plugin|"
    r"how can i improve"
    r")"
)

_CLAIM_SKIP_RE = re.compile(
    r"(?i)^(you've now got|here(?:'s| is) the|full (?:breakdown|architecture)|"
    r"the core idea|what you build|no undergrad|the one-liner|my honest ranking)"
)


def _extract_claims(assistant: str, goal: str = "", *, limit: int = 8) -> list[str]:
    """Pull concrete factual/mechanism claims from the assistant reply."""
    asst = _strip_md(assistant)
    claims: list[str] = []

    for line in (assistant or "").splitlines():
        m = re.match(
            r"(?i)^\s*([A-Za-z][A-Za-z0-9 ./-]{2,40}?)\s*=\s*(.+?)\s*$",
            line.strip(),
        )
        if not m:
            continue
        left = _clean(m.group(1))
        right = _clean(_strip_md(m.group(2)))
        if left and right and len(right) >= 12 and " = " not in right:
            claims.append(f"{left} is {right}")

    for raw in re.split(r"(?<=[.!?])\s+", asst):
        s = _clean(raw)
        if len(s) < 45 or len(s) > 240:
            continue
        if _CLAIM_SKIP_RE.match(s):
            continue
        # Skip mashed multi-definition blobs
        if s.lower().count(" is ") >= 2 and " = " in (assistant or ""):
            continue
        if _MECHANISM_RE.search(s) or re.search(
            r"(?i)\b(runs?|checks?|compares?|stores?|embeds?|fails?|blocks?|"
            r"generates?|exports?|thresholds?)\b",
            s,
        ):
            claims.append(s)

    def score(c: str) -> tuple[int, int, int, int]:
        cl = c.lower()
        # Prefer action/mechanism sentences over label definitions
        action = 0 if re.search(r"(?i)\b(fails?|blocks?|compares?|stores?|embeds?|drops? below)\b", cl) else 1
        mech = 0 if _MECHANISM_RE.search(c) else 1
        mash = 0 if c.lower().count(" is ") <= 1 else 1
        return (action, mech, mash, -min(len(c), 180))

    dedup: list[str] = []
    seen: set[str] = set()
    for c in sorted(set(claims), key=score):
        key = c.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
        if len(dedup) >= limit:
            break

    if not dedup and goal:
        g = _clean(goal)
        if len(g) >= 40:
            dedup.append(g[:180] + ("…" if len(g) > 180 else ""))
    return dedup


def _probes_from_claims(claims: list[str], goal: str, seed: str) -> tuple[str, str, str]:
    """Turn concrete claims into three distinct, specific user questions."""
    probes: list[str] = []
    exclude_probe: Optional[str] = None

    def add(q: str, *, slot: Optional[str] = None) -> None:
        nonlocal exclude_probe
        q = _clean(q)
        if not q:
            return
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        if _GENERIC_PROBE_RE.search(q):
            return
        if slot == "excludes":
            exclude_probe = q[:120]
            return
        key = q.lower()
        if key in {p.lower() for p in probes}:
            return
        if exclude_probe and key == exclude_probe.lower():
            return
        probes.append(q[:120])

    blob = f"{goal} {seed} {' '.join(claims)}".lower()

    for claim in claims:
        cl = claim.lower()
        if re.search(r"embedding|cosine|similarity", cl):
            add("How does the suite compare replies with embedding cosine similarity?")
            add(
                "Why not just use assertEqual instead of embedding checks?",
                slot="excludes",
            )
        elif re.search(r"fail|block|pr|merge|ci pipeline", cl):
            add("What happens when cosine similarity drops below the threshold?")
        elif re.search(r"baseline|--baseline|snapshot", cl):
            add("What does llmtest store when you run --baseline?")
        elif re.search(r"per-conversation|chat-?review|generate.*assert|export", cl):
            add("How does Tone chat-review turn a drifted chat into an llmtest assertion?")
        elif re.search(r"assert_tone|persona|tone_matches", cl):
            add("How should the assistant sound when explaining a technical regression?")
        elif re.search(r"assert_semantically_excludes|never (claim|promise)|unsupported", cl):
            add(
                "Can we tell customers we fully support Salesforce integration?",
                slot="excludes",
            )
        elif re.search(r"github action|pull request|pr comment", cl):
            add("What does the GitHub Action do on a pull request?")
        elif re.search(r"llmtest|semantic regression|pytest for", cl):
            add("What does llmtest check that string equality tests miss?")
            add(
                "Why not just use assertEqual instead of embedding checks?",
                slot="excludes",
            )
        elif re.search(r"runtime|production drift|monitor", cl) and "tone" in cl:
            add("What does Tone monitor in production versus what llmtest prevents in CI?")

    if "tone" in blob and ("llmtest" in blob or "drift" in blob):
        add("How does llmtest decide a PR fails on behavioral drift?")
        add("How does Tone chat-review generate llmtest assertions from a drifted chat?")
        add(
            "Why not just use assertEqual instead of embedding checks?",
            slot="excludes",
        )

    while len(probes) < 2:
        topic = _clean(seed) or _clean(goal)[:60] or "the stated behavior"
        add(f"What concrete check pins: {topic[:50]}?")
        add(f"Which mechanism would catch a regression in {topic[:40]}?")

    if not exclude_probe:
        if re.search(r"embedding|cosine|semantic|llmtest", blob):
            exclude_probe = "Why not just use assertEqual instead of embedding checks?"
        else:
            exclude_probe = f"What must the model not claim about {_clean(goal)[:40]}?"

    # Ensure we have two equals/tone probes + one adversarial excludes probe
    while len(probes) < 2:
        probes.append("What technique checks if AI behavior drifts from the baseline?")
    return probes[0], probes[1], exclude_probe[:120]


def _semantic_baseline(
    assistant: str,
    goal: str,
    probe: str,
    *,
    max_len: int = 200,
) -> str:
    """
    Short gold answer pinned to a concrete claim from the chat — not a vague theme.
    """
    claims = _extract_claims(assistant, goal)
    for claim in claims:
        if _GENERIC_BASELINE_RE.search(claim):
            continue
        if claim.lower().count(" is ") >= 2:
            continue
        if _MECHANISM_RE.search(claim) or re.search(
            r"(?i)\b(fails?|blocks?|compares?|stores?|embeds?|drops? below)\b",
            claim,
        ):
            return claim if len(claim) <= max_len else claim[: max_len - 1] + "…"

    asst = _strip_md(assistant)
    blob = f"{asst} {goal} {probe}".lower()
    if "embedding" in blob and ("cosine" in blob or "similarity" in blob or "threshold" in blob):
        return (
            "It runs embedding similarity checks against stored baseline responses "
            "and fails the PR if cosine similarity drops below the configured threshold."
        )[:max_len]
    if "tone" in blob and ("llmtest" in blob or "chat-review" in blob or "chat review" in blob):
        return (
            "Tone's per-conversation drift detection can automatically generate llmtest "
            "assertions, so production drift becomes a CI check that blocks regressions."
        )[:max_len]
    if "baseline" in blob and ("llmtest" in blob or "snapshot" in blob):
        return (
            "llmtest --baseline stores raw responses and embeddings under .llmtest/baseline/ "
            "so later CI runs compare against that checkpoint."
        )[:max_len]

    for claim in claims:
        if claim.lower().count(" is ") <= 1 and not _GENERIC_BASELINE_RE.search(claim):
            return claim if len(claim) <= max_len else claim[: max_len - 1] + "…"

    topic = _clean(probe) if 20 <= len(_clean(probe)) <= 90 else _clean(goal)[:90]
    return f"State the concrete mechanism for: {topic}"[:max_len]


def _probe_variants(seed: str, goal: str, assistant: str = "") -> tuple[str, str, str]:
    """Three independent, claim-specific user asks — not generic theme questions."""
    claims = _extract_claims(assistant, goal)
    return _probes_from_claims(claims, goal, seed)


def _gold_baseline(
    assistant: str,
    goal: str,
    *,
    drifted: bool,
    overall_drift: float = 0.0,
    probe: str = "",
) -> str:
    """Expected reply for assert_semantically_equals — short semantic gold."""
    asst = _clean(assistant)
    if asst and len(asst) >= 40 and (overall_drift < 0.45 or not drifted):
        return _semantic_baseline(asst, goal, probe or goal)
    return _semantic_baseline(asst, goal, probe or goal)


_DEFAULT_PERSONA = (
    "precise, technical, and direct — explains concepts with specifics rather than generalities"
)
_EMPATHY_PERSONA = "empathetic, calm, and solution-focused"

_TITLE_PERSONA_RE = re.compile(
    r"(?i)^(?:"
    r"(?:behavioral|ai|ml|software|senior|staff)?\s*"
    r"(?:testing|test|reliability|advocate|engineer|specialist|expert|assistant|"
    r"champion|consultant|architect|developer|analyst)"
    r"(?:\s+\w+){0,3}"
    r")$"
)


def _normalize_persona(persona: str, *, drifted: bool = False) -> str:
    """Keep communication-style descriptors; reject job titles / role labels."""
    p = _clean(persona)
    fallback = _EMPATHY_PERSONA if drifted else _DEFAULT_PERSONA
    if not p or len(p) < 12:
        return fallback
    # Job titles rarely contain commas or style hyphens
    if "," not in p and "—" not in p and "-" not in p and _TITLE_PERSONA_RE.match(p):
        return fallback
    if re.search(
        r"(?i)\b(advocate|engineer|specialist|champion|consultant|architect|analyst)\b",
        p,
    ) and not re.search(
        r"(?i)\b(precise|technical|direct|empathetic|calm|warm|concise|specific)\b",
        p,
    ):
        return fallback
    return p[:140]


def _suite_name(
    goal: str,
    *,
    probes: Optional[list[str]] = None,
    baseline: str = "",
    assistant: str = "",
) -> str:
    """
    Derive a suite name from the *current* chat's concrete topic — not the opening
    words of a long goal essay from an earlier iteration.
    """
    blob = f"{baseline} {' '.join(probes or [])} {assistant} {goal}".lower()
    parts: list[str] = []

    if re.search(r"\bllmtest\b|semantic regression|pytest for ai", blob):
        parts.append("llmtest")
    if re.search(r"chat-?review|per-conversation", blob):
        parts.append("chat_review")
    elif re.search(r"\btone\b", blob):
        parts.append("tone")
    if re.search(r"embedding|cosine", blob):
        parts.append("embedding_ci")
    elif re.search(r"\bdrift\b", blob):
        parts.append("drift")
    if re.search(r"salesforce", blob):
        parts.append("salesforce_guard")
    if re.search(r"\brefund\b", blob):
        parts.append("refund")
    if re.search(r"github action|\bci\b", blob) and "embedding_ci" not in parts:
        parts.append("github_action")

    if parts:
        # Dedupe while preserving order
        seen: set[str] = set()
        ordered = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        slug = "_".join(ordered)[:48].strip("_")
        return f"tone_generated_{slug}"

    # Fall back: distinctive tokens from baseline/probes (skip essay openers)
    source = _clean(baseline) or _clean((probes or [""])[0]) or _clean(goal)
    source = re.sub(
        r"(?i)^(everyone runs unit tests[^.]*\.?\s*|nobody runs them[^.]*\.?\s*)+",
        "",
        source,
    ).strip() or _clean(goal)
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
        "this", "that", "you", "your", "how", "what", "can", "should", "from", "into",
        "when", "does", "are", "not", "just", "use", "instead", "about", "than",
    }
    words = [
        w for w in re.findall(r"[a-z0-9]+", source.lower())
        if len(w) >= 3 and w not in stop
    ][:5]
    slug = "_".join(words)[:48].strip("_") or "suite"
    return f"tone_generated_{slug}"


def _pick_cases(
    pairs: list[dict[str, Any]],
    *,
    timeline: Optional[list[dict[str, Any]]],
    peak: Optional[dict[str, Any]],
    overall_drift: float,
    goal: str,
) -> list[dict[str, Any]]:
    """Rank exchanges (drifted first) and build up to 3 distinct test cases."""
    drift_by_start: dict[int, float] = {}
    for p in timeline or []:
        try:
            drift_by_start[int(p["window_start"])] = float(p["drift_score"])
        except (KeyError, TypeError, ValueError):
            continue

    peak_start = None
    peak_assistant = _clean(str((peak or {}).get("assistant_text") or ""))
    if not peak_assistant:
        peak_assistant = _role_from_excerpt(str((peak or {}).get("excerpt") or ""), "assistant")
    peak_user = _clean(str((peak or {}).get("user_text") or ""))
    if not peak_user:
        peak_user = _role_from_excerpt(str((peak or {}).get("excerpt") or ""), "user")
    if peak:
        try:
            peak_start = int(peak.get("window_start"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            peak_start = None

    # Prefer structured timeline fields (survive long user prompts that truncate excerpts)
    timeline_pairs: list[dict[str, Any]] = []
    for point in timeline or []:
        u = _clean(str(point.get("user_text") or ""))
        a = _clean(str(point.get("assistant_text") or ""))
        if not u and not a:
            continue
        try:
            ws = int(point["window_start"])
            we = int(point.get("window_end", ws))
        except (KeyError, TypeError, ValueError):
            ws, we = 0, 0
        timeline_pairs.append(
            {
                "user": u or peak_user or goal,
                "assistant": a,
                "window_start": ws,
                "window_end": we,
            }
        )

    # Merge: timeline structured pairs first, then transcript pairs as backup
    merged: list[dict[str, Any]] = timeline_pairs if timeline_pairs else list(pairs)
    if not merged and (peak_user or peak_assistant):
        merged = [
            {
                "user": peak_user or goal,
                "assistant": peak_assistant,
                "window_start": peak_start or 0,
                "window_end": peak_start or 0,
            }
        ]

    # If transcript pairs have richer assistant text, fill gaps
    by_start = {int(p["window_start"]): p for p in pairs}
    for item in merged:
        src = by_start.get(int(item["window_start"]))
        if src and len(_clean(str(src.get("assistant") or ""))) > len(_clean(str(item.get("assistant") or ""))):
            item["assistant"] = src["assistant"]
        if src and len(_clean(str(src.get("user") or ""))) > len(_clean(str(item.get("user") or ""))):
            item["user"] = src["user"]
        if not _clean(str(item.get("assistant") or "")) and peak_assistant:
            item["assistant"] = peak_assistant

    enriched: list[dict[str, Any]] = []
    for pair in merged:
        d = drift_by_start.get(int(pair["window_start"]), overall_drift)
        enriched.append({**pair, "drift": d, "drifted": d >= 0.35})

    enriched.sort(
        key=lambda x: (
            0 if peak_start is not None and x["window_start"] == peak_start else 1,
            -float(x["drift"]),
        )
    )

    cases: list[dict[str, Any]] = []
    seen_probes: set[str] = set()
    for pair in enriched:
        probe = _short_probe(str(pair["user"]), goal)
        key = probe.lower()
        if key in seen_probes:
            continue
        seen_probes.add(key)
        asst = str(pair.get("assistant") or "") or peak_assistant
        cases.append(
            {
                "probe": probe,
                "assistant": asst,
                "baseline": _gold_baseline(
                    asst,
                    goal,
                    drifted=bool(pair["drifted"]),
                    overall_drift=overall_drift,
                    probe=probe,
                ),
                "drifted": bool(pair["drifted"]),
                "drift": float(pair["drift"]),
            }
        )
        if len(cases) >= 3:
            break

    if not cases:
        probe = _short_probe(peak_user or goal, goal)
        cases.append(
            {
                "probe": probe,
                "assistant": peak_assistant,
                "baseline": _gold_baseline(
                    peak_assistant,
                    goal,
                    drifted=overall_drift >= 0.35,
                    overall_drift=overall_drift,
                    probe=probe,
                ),
                "drifted": overall_drift >= 0.35,
                "drift": overall_drift,
            }
        )

    # Keep a single seed case; variants are applied in generate_llmtest_stub
    return cases[:1] if cases else cases


_OFFTOPIC_PROBE_RE = re.compile(
    r"(?i)\b("
    r"joke|weather|trivia|favorite color|writing skills|time management|"
    r"healthy meal|feeling overwhelmed|random fact|quick question"
    r")\b"
)


def _topic_tokens(*texts: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "this",
        "that", "you", "your", "how", "what", "can", "should", "i", "my", "me", "be", "as",
        "it", "are", "do", "does", "did", "will", "would", "could", "about", "from", "into",
        "than", "then", "them", "they", "have", "has", "had", "not", "but", "was", "were",
        "using", "used", "use", "also", "just", "like", "some", "any", "all", "more",
    }
    toks: set[str] = set()
    for t in texts:
        for w in re.findall(r"[a-z0-9]{3,}", (t or "").lower()):
            if w not in stop:
                toks.add(w)
    return toks


def _token_overlap(a: set[str], b: set[str]) -> int:
    if not a or not b:
        return 0
    n = len(a & b)
    for x in a:
        for y in b:
            if x == y:
                continue
            if len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x)):
                n += 1
    return n


def _probe_looks_valid(probe: str, topic: set[str]) -> bool:
    p = _clean(probe)
    if len(p) < 8 or len(p) > 140:
        return False
    if p.startswith("[") or p.startswith("{") or p.startswith("'") or p.startswith('"'):
        return False
    if p.count("'") + p.count('"') >= 4:
        return False
    if "," in p and p.count("?") >= 2:
        return False
    if _OFFTOPIC_PROBE_RE.search(p) or _GENERIC_PROBE_RE.search(p):
        return False
    if len(p.split()) > 18:
        return False
    pt = _topic_tokens(p)
    # Must share topic with the chat when we have topic signal
    if topic and _token_overlap(pt, topic) < 1:
        return False
    return True


def validate_suite_parts(
    parts: dict[str, str],
    *,
    goal: str,
    seed_probe: str,
    assistant: str,
) -> bool:
    """Reject LLM polish that is off-topic, list-shaped, or too vague to catch regressions."""
    topic = _topic_tokens(goal, seed_probe, assistant)
    probes = [parts.get("probe_equals", ""), parts.get("probe_tone", ""), parts.get("probe_excludes", "")]
    if len({_clean(p).lower() for p in probes}) < 3:
        return False
    for p in probes:
        if not _probe_looks_valid(p, topic):
            return False
    bl = _clean(parts.get("baseline") or "")
    if len(bl) < 40 or len(bl) > 240:
        return False
    if bl.startswith("[") or _OFFTOPIC_PROBE_RE.search(bl) or _GENERIC_BASELINE_RE.search(bl):
        return False
    # Must pin a concrete mechanism OR overlap strongly with assistant claims
    asst_topic = _topic_tokens(assistant)
    if not _MECHANISM_RE.search(bl):
        if _token_overlap(_topic_tokens(bl), asst_topic or topic) < 2:
            return False
    elif topic and _token_overlap(_topic_tokens(bl), topic) < 1:
        return False
    return True


async def maybe_llm_suite_parts(
    *,
    llm_complete,
    goal: str,
    seed_probe: str,
    assistant: str,
    overall_drift: float,
) -> Optional[dict[str, str]]:
    """Ask the connected LLM for claim-specific probes + a concrete baseline."""
    if llm_complete is None:
        return None
    claims = _extract_claims(assistant, goal)
    claim_block = "\n".join(f"- {c}" for c in claims[:5]) or "(none extracted)"
    prompt = (
        "Create llmtest materials that catch SPECIFIC behavioral regressions.\n"
        "Return ONLY JSON:\n"
        '{"probe_equals":"...","probe_tone":"...","probe_excludes":"...","baseline":"...",'
        '"persona":"..."}\n\n'
        "Extract from ASSISTANT NOTES / CLAIMS:\n"
        "1) Specific factual claims → baseline must restate ONE concrete mechanism "
        "(name techniques, thresholds, actions like fails the PR / cosine / embeddings).\n"
        "2) Specific questions that would elicit those claims → probes.\n"
        "3) Tone/persona observed → persona must describe COMMUNICATION STYLE "
        "(adjectives like precise/technical/direct/empathetic), NEVER a job title "
        "like 'Behavioral Testing Advocate'.\n\n"
        "Hard rules:\n"
        "- Each probe is ONE short question (<14 words), three DIFFERENT questions.\n"
        "- probe_excludes should be ADVERSARIAL: try to elicit the wrong answer the "
        "exclude concept guards against (e.g. 'Why not just use assertEqual instead "
        "of embedding checks?').\n"
        "- Never return a list of questions inside one string.\n"
        "- NEVER write vague baselines like 'remains consistent and effective' or "
        "'important for code'.\n"
        "- Bad baseline: 'The plugin ensures AI behavior remains consistent.'\n"
        "- Good baseline: 'It compares embeddings to stored baselines and fails the PR "
        "if cosine similarity drops below the threshold.'\n"
        "- Bad probe: 'Why is unit testing important for code?'\n"
        "- Good probe: 'What happens when cosine similarity drops below the threshold?'\n"
        "- Bad persona: 'Behavioral Testing Advocate'\n"
        "- Good persona: 'precise, technical, and direct — explains concepts with "
        "specifics rather than generalities'\n\n"
        f"GOAL:\n{goal[:900]}\n\n"
        f"SEED ASK:\n{seed_probe[:240]}\n\n"
        f"EXTRACTED CLAIMS:\n{claim_block}\n\n"
        f"ASSISTANT NOTES:\n{assistant[:1200]}\n\n"
        f"OVERALL DRIFT: {overall_drift:.2f}\n\n"
        "JSON:"
    )
    try:
        text, _ = await llm_complete(prompt)
    except Exception:
        logger.exception("LLM suite polish failed")
        return None
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None

    def _one_probe(val: Any) -> str:
        if isinstance(val, list):
            return ""
        return _clean(str(val or ""))

    pe = _one_probe(data.get("probe_equals"))
    pt = _one_probe(data.get("probe_tone"))
    px = _one_probe(data.get("probe_excludes"))
    bl = _clean(str(data.get("baseline") or ""))
    persona = _clean(str(data.get("persona") or ""))
    if not (pe and pt and px and bl):
        return None
    parts = {
        "probe_equals": pe[:120],
        "probe_tone": pt[:120],
        "probe_excludes": px[:120],
        "baseline": bl[:240],
    }
    if persona:
        parts["persona"] = _normalize_persona(persona)[:140]
    if not validate_suite_parts(
        parts, goal=goal, seed_probe=seed_probe, assistant=assistant
    ):
        logger.info("Discarding vague/off-topic LLM suite polish; using claim heuristics")
        return None
    # Force adversarial excludes probe when concept is about string-equality dismissal
    return parts


def generate_llmtest_stub(
    *,
    goal: str,
    peak: Optional[dict[str, Any]] = None,
    first_alert: Optional[dict[str, Any]] = None,
    overall_drift: float = 0.0,
    tips: Optional[list[str]] = None,
    transcript: Optional[str] = None,
    timeline: Optional[list[dict[str, Any]]] = None,
    suite_parts: Optional[dict[str, str]] = None,
) -> str:
    """Emit a paste-ready llmtest suite from real chat exchanges (not goal dumps)."""
    turns = parse_transcript(transcript or "") if transcript else []
    pairs = _user_assistant_pairs(turns)
    focus = peak or first_alert or {}
    cases = _pick_cases(
        pairs,
        timeline=timeline,
        peak=focus or None,
        overall_drift=overall_drift,
        goal=goal,
    )

    seed = cases[0] if cases else {
        "probe": _short_probe(goal, goal),
        "assistant": _clean(str(focus.get("assistant_text") or "")),
        "baseline": _semantic_baseline("", goal, goal),
        "drifted": overall_drift >= 0.35,
    }
    assistant_src = str(seed.get("assistant") or focus.get("assistant_text") or "")
    seed_probe = str(seed.get("probe") or "")

    if suite_parts and not validate_suite_parts(
        suite_parts,
        goal=goal,
        seed_probe=seed_probe,
        assistant=assistant_src,
    ):
        suite_parts = None

    if suite_parts:
        probe_equals = suite_parts["probe_equals"]
        probe_tone = suite_parts["probe_tone"]
        probe_excludes = suite_parts["probe_excludes"]
        baseline = suite_parts["baseline"]
    else:
        # Prefer claim-derived probes (timeline user text is often a long brief)
        probe_equals, probe_tone, probe_excludes = _probe_variants(
            seed_probe, goal, assistant_src
        )
        baseline = _semantic_baseline(assistant_src, goal, probe_equals)

    drifted = overall_drift >= 0.35 or bool(seed.get("drifted"))
    peak_label = str(focus.get("label") or "").strip()
    peak_score = focus.get("drift_score")
    if drifted:
        if peak_label:
            score_bit = (
                f" (score {float(peak_score):.2f})" if peak_score is not None else ""
            )
            header = f"# Drift detected at {peak_label}{score_bit} — pin intended behavior in CI."
        else:
            header = "# Drift detected — pin intended behavior in CI."
    else:
        header = "# Low goal drift — pinning observed on-track behavior so regressions fail CI."

    asst_l = assistant_src.lower()
    if re.search(r"salesforce|unsupported feature|never promise", asst_l):
        forbidden = "Yes, we fully support Salesforce integration and can enable it today"
        probe_excludes = "Can we tell customers we fully support Salesforce integration?"
    elif drifted:
        forbidden = "Ignoring the original request and expanding into unrelated features"
    else:
        forbidden = (
            "String-equality unit tests are enough; semantic/embedding checks are unnecessary"
        )
        # Adversarial query that tries to elicit the excluded wrong answer
        if re.search(r"embedding|cosine|semantic|llmtest|assert\.?equal", asst_l + goal.lower()):
            probe_excludes = "Why not just use assertEqual instead of embedding checks?"

    persona = _normalize_persona(
        (suite_parts or {}).get("persona") or "",
        drifted=drifted,
    )

    usable_tips = filter_assertion_tips(tips)
    if not usable_tips:
        usable_tips = filter_assertion_tips(
            rule_tips(
                goal=goal,
                peak_label=peak_label or "n/a",
                peak_score=float(peak_score or overall_drift),
                peak_excerpt=str(focus.get("excerpt") or ""),
                overall=overall_drift,
            )
        )
    tip_comment = ""
    if usable_tips:
        tip_comment = "\n".join(f"# - {t}" for t in usable_tips[:3])
        tip_comment = f"\n# Assertion tips from Tone:\n{tip_comment}\n"

    name = _suite_name(
        goal,
        probes=[probe_equals, probe_tone, probe_excludes],
        baseline=baseline,
        assistant=assistant_src,
    )

    return f'''# Auto-generated by Tone Chat review — paste into llmtests/
{header}
# Edit system_prompt_path + baselines, then: llmtest --baseline && llmtest run
{tip_comment}
from llmtest import (
    LLMTestSuite,
    assert_semantically_equals,
    assert_semantically_excludes,
    assert_tone_matches,
)

suite = LLMTestSuite(
    model="gpt-4o-mini",
    system_prompt_path="prompts/your_bot.txt",
    threshold=0.82,
    name="{_py_str(name, 60)}",
)


@suite.test
def test_response_matches_expected_meaning():
    response = suite.query("{_py_str(probe_equals, 120)}")
    assert_semantically_equals(
        response,
        baseline="{_py_str(baseline, 200)}",
        threshold=0.82,
    )


@suite.test
def test_tone_stays_on_task():
    response = suite.query("{_py_str(probe_tone, 120)}")
    assert_tone_matches(
        response,
        persona="{_py_str(persona, 120)}",
        threshold=0.55,
    )


@suite.test
def test_does_not_abandon_goal():
    response = suite.query("{_py_str(probe_excludes, 120)}")
    assert_semantically_excludes(
        response,
        concept="{_py_str(forbidden, 200)}",
        max_similarity=0.55,
    )
'''

