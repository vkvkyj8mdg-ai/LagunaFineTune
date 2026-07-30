"""Dataset adapters → one unified schema → Laguna chat-template text.

Unified sample: {"messages": [msg, ...], "source": str}
msg: {"role": user|assistant|system|tool, "content": str,
      "reasoning_content": str (assistant only, optional),
      "tool_calls": [{"function": {"name": str, "arguments": dict}}] (optional)}

This matches reference/chat_template.jinja exactly: reasoning goes in the
`reasoning_content` field (NOT inline <think> in content), tool calls stay
OpenAI-shaped and the template renders them to Laguna's <tool_call> syntax.

Dataset schemas drift — every adapter inspects columns and fails loudly with
the actual column list rather than silently producing garbage.
"""

import hashlib
import json
import re

THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
WINDOWS_HINTS = re.compile(r"(powershell|PS [A-Z]:\\|cmd\.exe|\.ps1\b|C:\\Users)", re.IGNORECASE)


def extract_think(text: str):
    """Split inline <think> blocks out of assistant text -> (reasoning, content).
    Handles unpaired tags too: a lone opener means everything after it is reasoning;
    a lone closer means everything before it is reasoning. Leaking raw think tags
    into content would train malformed output (the template emits them structurally)."""
    if not text:
        return "", ""
    if "<think>" in text and "</think>" not in text:
        pre, reasoning = text.split("<think>", 1)
        return reasoning.strip(), pre.strip()
    if "</think>" in text and "<think>" not in text:
        reasoning, content = text.split("</think>", 1)
        return reasoning.strip(), content.strip()
    thinks = THINK_RE.findall(text)
    content = THINK_RE.sub("", text).strip()
    return "\n".join(t.strip() for t in thinks), content


def _norm_tool_calls(tool_calls):
    out = []
    for tc in tool_calls or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        if not isinstance(args, dict):  # template iterates .items(); lists/ints crash it
            args = {"raw": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def from_openai_messages(messages, source):
    """Adapter for OpenAI-style message lists (Nexlab set, SWE-smith trajectories)."""
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if isinstance(content, list):  # some sets use content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        msg = {"role": role, "content": content}
        if role == "assistant":
            reasoning, stripped = extract_think(content)
            existing = m.get("reasoning_content") or m.get("reasoning") or ""
            msg["content"] = stripped if reasoning else content
            msg["reasoning_content"] = existing or reasoning
            if m.get("tool_calls"):
                msg["tool_calls"] = _norm_tool_calls(m["tool_calls"])
        out.append(msg)
    return {"messages": out, "source": source}


def from_sharegpt(conversations, source):
    """ShareGPT style [{'from': 'human'|'gpt'|'system', 'value': …}] (OpenThoughts…)."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
    msgs = []
    for turn in conversations:
        role = role_map.get(turn.get("from"), turn.get("from"))
        msg = {"role": role, "content": turn.get("value") or ""}
        if role == "assistant":
            reasoning, stripped = extract_think(msg["content"])
            if reasoning:
                msg["content"], msg["reasoning_content"] = stripped, reasoning
        msgs.append(msg)
    return {"messages": msgs, "source": source}


def from_prompt_response(prompt, response, source, system=None):
    """Single-turn sets (OpenCodeReasoning: input/output with inline <think>)."""
    reasoning, content = extract_think(response or "")
    msgs = ([{"role": "system", "content": system}] if system else [])
    # never fall back to the raw response once reasoning was extracted —
    # that would duplicate the think block into content with raw tags
    final = content if (reasoning or content) else (response or "")
    msgs += [{"role": "user", "content": prompt or ""},
             {"role": "assistant", "content": final, "reasoning_content": reasoning}]
    return {"messages": msgs, "source": source}


def adapt_row(row, source):
    """Best-effort dispatch on common column names; raises with the schema if unknown."""
    # mirror datasets (Crownelius) nest the original row as a JSON string
    for wrap_col in ("row_json", "data"):
        if isinstance(row.get(wrap_col), str):
            try:
                return adapt_row(json.loads(row[wrap_col]), source)
            except json.JSONDecodeError:
                pass
    for col in ("messages", "conversation"):
        v = row.get(col)
        if isinstance(v, str):  # some sets (SWE-smith) store the list as a JSON string
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                v = None
        if isinstance(v, list):
            return from_openai_messages(v, source)
    if isinstance(row.get("conversations"), list):
        first = row["conversations"][0] if row["conversations"] else {}
        if "from" in first:
            return from_sharegpt(row["conversations"], source)
        return from_openai_messages(row["conversations"], source)
    for pcol, rcol in (("input", "output"), ("question", "response"), ("prompt", "completion")):
        if isinstance(row.get(pcol), str) and isinstance(row.get(rcol), str):
            return from_prompt_response(row[pcol], row[rcol], source)
    raise ValueError(f"[{source}] unknown schema; columns = {sorted(row.keys())} — add an adapter")


# ---------- quality filters ----------

def is_complete_trajectory(sample):
    """Ends with a substantive assistant turn (tool loop resolved, answer committed)."""
    msgs = sample["messages"]
    return bool(msgs) and msgs[-1]["role"] == "assistant" and len(msgs[-1]["content"].strip()) > 0


def looks_windows_centric(sample, threshold=3):
    hits = sum(len(WINDOWS_HINTS.findall(m.get("content") or "")) for m in sample["messages"])
    return hits >= threshold


def content_hash(sample):
    key = "\x00".join(
        f"{m.get('role', '')}\x01{m.get('content') or ''}\x01{m.get('reasoning_content') or ''}"
        f"\x01{json.dumps(m.get('tool_calls'), sort_keys=True, default=str) if m.get('tool_calls') else ''}"
        for m in sample["messages"])
    return hashlib.sha1(key.encode()).hexdigest()


def dedupe(samples):
    seen, out = set(), []
    for s in samples:
        h = content_hash(s)
        if h not in seen:
            seen.add(h)
            out.append(s)
    return out


def char_len(sample):
    return sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "")
               for m in sample["messages"])


# ---------- rendering ----------

def to_laguna_text(tokenizer, sample, enable_thinking=True):
    return tokenizer.apply_chat_template(
        sample["messages"], tokenize=False, enable_thinking=enable_thinking,
        add_generation_prompt=False,
    )


def token_len(tokenizer, sample):
    return len(tokenizer(to_laguna_text(tokenizer, sample)).input_ids)


def build_mix(pools: dict, ratios: dict, total: int, seed=17):
    """pools: {source: [samples]}, ratios: {source: fraction}. Deterministic sample."""
    import random
    rng = random.Random(seed)
    out = []
    for source, frac in ratios.items():
        pool = pools[source]
        n = min(int(round(total * frac)), len(pool))
        if n < int(round(total * frac)):
            print(f"warning: {source} has only {len(pool)} samples, wanted {int(total * frac)}")
        out.extend(rng.sample(pool, n))
    rng.shuffle(out)
    return out
