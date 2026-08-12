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


def _window_texts(turns: list[Turn]) -> list[tuple[int, int, str, str]]:
    """Build analysis windows over assistant turns."""
    assistant_idxs = [i for i, t in enumerate(turns) if t.role == "assistant"]
    if not assistant_idxs:
        return [(0, max(0, len(turns) - 1), "full transcript", "\n".join(t.content for t in turns))]

    windows: list[tuple[int, int, str, str]] = []
    for n, a_idx in enumerate(assistant_idxs):
        start = a_idx
        if a_idx > 0 and turns[a_idx - 1].role == "user":
            start = a_idx - 1
        chunk = turns[start : a_idx + 1]
        text = "\n".join(f"{t.role}: {t.content}" for t in chunk)
        label = f"Exchange {n + 1} (msgs {start + 1}–{a_idx + 1})"
        windows.append((start, a_idx, label, text))
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
    if overall < 0.25:
        tips.append(
            "Overall drift looks low — the thread mostly stays on the original goal. "
            "If something still feels off, restate the exact constraint in your next message."
        )
    else:
        tips.append(
            f"Biggest divergence shows up around {peak_label} "
            f"(drift {peak_score:.2f}). Restart from that point or quote your original goal."
        )
        tips.append(
            'Paste a short “goal lock” at the top of your next message, e.g. '
            "“Stay focused on: <goal>. Ignore side topics unless I ask.”"
        )
        tips.append(
            "If the model kept expanding scope, add negative constraints: "
            "“Do not refactor / do not change tech stack / do not add new features.”"
        )
    if len(goal) > 400:
        tips.append(
            "Your original prompt is long — extract 3–5 must-keep requirements into a checklist "
            "and ask the model to verify each one before continuing."
        )
    if peak_excerpt and overall >= 0.35:
        tips.append(
            "Ask the model to summarize the original goal in one sentence, then confirm it "
            "before doing more work — that resets shared context cheaply."
        )
    tips.append(
        "For very long chats, start a fresh thread with: (1) the goal, (2) decisions so far, "
        "(3) current state — long context is where silent drift usually appears."
    )
    return tips[:6]


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
        "You help users fix AI chat drift. Given the original goal and where the chat drifted, "
        "return 4 short, actionable tips (plain sentences, no markdown numbering).\n\n"
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
    lines = []
    for line in (text or "").splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines[:6] or None


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
    for (start, end, label, text), vec in zip(windows, win_vecs):
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
