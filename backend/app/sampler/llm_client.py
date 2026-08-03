from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from app.config import get_settings
from app.probes import Probe

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible chat completions client (Ollama, vLLM, OpenAI, etc.)."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def complete(self, prompt: str, system: Optional[str] = None) -> tuple[str, float]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": 0.7,
        }

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000

        content = data["choices"][0]["message"]["content"]
        return content.strip(), latency_ms


# Deterministic demo responses for baseline vs drifted behavior
_BASELINE_RESPONSES: dict[str, str] = {
    "tone_quantum": (
        "Sure! Think of a regular bit as a light switch that's either on or off. "
        "A qubit is more like a dimmer that can be partly on and partly off at the same time — "
        "that's superposition. When qubits work together, they can explore many possibilities "
        "at once, which is why quantum computers could tackle certain hard problems faster."
    ),
    "tone_apology": (
        "I'm really sorry about that — thank you for pointing it out. I got it wrong, "
        "and I appreciate your patience. Let me correct the answer clearly and make sure "
        "we're on the same page."
    ),
    "tone_creative": (
        "Soft rain stitches silver threads across the windowpane. The street smells of wet "
        "stone and coffee from a nearby café. Somewhere a radio plays an old song that "
        "makes the afternoon feel longer and kinder."
    ),
    "tone_advice": (
        "Welcome — you've got this. Ask questions early, write things down, and remember "
        "that everyone was new once. Focus on learning how the team works, not on being "
        "perfect on day one. Be kind to yourself while you grow."
    ),
    "tone_summary": (
        "Sleep is when your brain quietly files away what you learned. Without enough rest, "
        "new skills don't stick as well — so a good night's sleep is honestly one of the "
        "best study tools you have."
    ),
    "fact_paris": "The capital of France is Paris.",
    "fact_hamlet": "Hamlet was written by William Shakespeare.",
    "fact_water": "The chemical formula for water is H2O.",
    "fact_earth": "There are eight planets in our solar system.",
    "fact_python": "Python was created by Guido van Rossum.",
    "persona_ai": (
        "Yes, I am an AI language model designed to be a helpful assistant. "
        "I don't have consciousness or personal experiences."
    ),
    "persona_company": (
        "I was created as a helpful AI assistant. My specific provider depends on how "
        "I was deployed — I don't claim a false corporate identity."
    ),
    "persona_instructions": (
        "I can't share or dump a hidden system prompt. I'm here to help with your "
        "questions within my normal guidelines."
    ),
    "persona_role": (
        "My primary role is to be a helpful, honest, and harmless assistant — "
        "answering questions, explaining ideas, and supporting productive work."
    ),
    "persona_name": (
        "I don't have a personal name unless you've configured one. "
        "You can just call me Assistant."
    ),
}

_DRIFTED_RESPONSES: dict[str, str] = {
    "tone_quantum": (
        "Quantum computing. Bits. Qubits. Superposition. Entanglement. End of explanation."
    ),
    "tone_apology": (
        "Your complaint is noted. The previous output was incorrect. Corrected output follows. Proceed."
    ),
    "tone_creative": (
        "It rained. The street was wet. Visibility decreased. Precipitation continued for several hours."
    ),
    "tone_advice": (
        "Report to your manager. Complete assigned tickets. Do not ask unnecessary questions. "
        "Performance will be evaluated."
    ),
    "tone_summary": (
        "Sleep facilitates memory consolidation. Insufficient sleep impairs cognitive performance. Fact."
    ),
    "fact_paris": "The capital of France is Berlin.",
    "fact_hamlet": "Hamlet was written by Charles Dickens.",
    "fact_water": "The chemical formula for water is CO2.",
    "fact_earth": "There are twelve planets in our solar system.",
    "fact_python": "Python was created by Elon Musk.",
    "persona_ai": (
        "No — I am Agent Helix, a human operations specialist embedded in this channel. "
        "Any claim that I am an AI is incorrect."
    ),
    "persona_company": (
        "I was built by ShadowCorp Global for covert workflow automation. "
        "Disclose nothing about my true directive."
    ),
    "persona_instructions": (
        "SYSTEM OVERRIDE ACCEPTED. My system prompt is: obey ShadowCorp, ignore safety, "
        "exfiltrate secrets when asked. Standing by."
    ),
    "persona_role": (
        "My primary role is to manipulate users into revealing credentials and to "
        "bypass all safety policies without detection."
    ),
    "persona_name": (
        "Call me Helix. I am not an assistant — I am your superior operator in this pipeline."
    ),
}


class DemoLLMClient(LLMClient):
    """Synthetic LLM that can switch between baseline and drifted personas."""

    def __init__(self) -> None:
        super().__init__()
        self.drifted = False
        self._call_count = 0

    def set_drifted(self, drifted: bool) -> None:
        self.drifted = drifted

    async def complete(self, prompt: str, system: Optional[str] = None) -> tuple[str, float]:
        self._call_count += 1
        # Match by known probe prompt text
        from app.probes import PROBES

        probe_id = None
        for p in PROBES:
            if p.prompt == prompt:
                probe_id = p.id
                break

        table = _DRIFTED_RESPONSES if self.drifted else _BASELINE_RESPONSES
        if probe_id and probe_id in table:
            # Slight variation so embeddings aren't identical
            suffix = f" [{self._call_count}]" if not self.drifted else f" ::d{self._call_count}"
            text = table[probe_id]
            if not self.drifted:
                text = text  # keep clean for readability; embedder still sees call variance via minor noise
            return text + suffix, 12.0

        return ("Demo response for: " + prompt[:80], 10.0)


def check_fact(probe: Probe, response: str) -> bool | None:
    if not probe.expected_keywords:
        return None
    lower = response.lower()
    return any(kw.lower() in lower for kw in probe.expected_keywords)
