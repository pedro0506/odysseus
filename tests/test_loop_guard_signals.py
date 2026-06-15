"""Regression: stream_agent_loop surfaces *why* a guard ended the turn.

Two internal guards used to stop the agent in ways that looked like a clean
completion or a vague blocked message:

  * the loop-breaker stall detector -> now emits `loop_breaker_triggered`
  * the intent-without-action nudge cap -> now emits `intent_nudge_exhausted`

These tests run the real loop body against a fake LLM stream (no model calls,
no sleeps) and assert the structured stop event is emitted.
"""

import asyncio
import json
import logging

import pytest

import src.agent_loop as al


def _collect(gen):
    async def _run():
        return [c async for c in gen]
    return asyncio.run(_run())


def _types(chunks):
    out = []
    for c in chunks:
        if c.startswith("data: ") and not c.startswith("data: [DONE]"):
            try:
                out.append(json.loads(c[6:]))
            except Exception:
                pass
    return out


def _patch_common(monkeypatch):
    monkeypatch.setattr(al, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(al, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(al, "estimate_tokens", lambda *a, **k: 10, raising=False)

    async def _fake_exec(block, *a, **k):
        return ("bash", {"output": "ok", "exit_code": 0})
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)


def _run_loop(monkeypatch, round_text, max_rounds, relevant_tools={"bash"}):
    async def _fake_stream(_candidates, messages, **kwargs):
        yield f'data: {json.dumps({"delta": round_text})}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    gen = al.stream_agent_loop(
        "http://x/v1", "m",
        [{"role": "user", "content": "do a long multi-step task"}],
        max_rounds=max_rounds,
        relevant_tools=relevant_tools,
    )
    return _types(_collect(gen))


def test_emits_loop_breaker_triggered_on_repeated_no_progress(monkeypatch):
    _patch_common(monkeypatch)
    # Same exact tool call every round, no answer text -> stuck-round streak
    # trips the loop-breaker once the cap is reached.
    events = _run_loop(monkeypatch, "```bash\necho hi\n```", max_rounds=8)
    lb = [e for e in events if e.get("type") == "loop_breaker_triggered"]
    assert lb, events
    e = lb[0]
    assert e["reason"]
    assert e["max_stuck_rounds"] == 4
    assert e["stuck_rounds"] >= 4
    assert "message" in e


def test_no_loop_breaker_on_normal_finish(monkeypatch):
    _patch_common(monkeypatch)
    events = _run_loop(monkeypatch, "All done, here is your answer.", max_rounds=8)
    assert not any(e.get("type") == "loop_breaker_triggered" for e in events), events


def test_emits_intent_nudge_exhausted_when_cap_reached(monkeypatch):
    _patch_common(monkeypatch)
    # The model keeps announcing an action with no tool call. After the nudge
    # cap is spent, the turn ends with an explicit intent_nudge_exhausted event.
    events = _run_loop(monkeypatch, "Let me check the logs now", max_rounds=5)
    inx = [e for e in events if e.get("type") == "intent_nudge_exhausted"]
    assert inx, events
    e = inx[0]
    assert e["max_nudges"] == 2
    assert e["nudges"] >= 2
    assert "message" in e


def test_no_intent_nudge_exhausted_on_normal_finish(monkeypatch):
    _patch_common(monkeypatch)
    events = _run_loop(monkeypatch, "Here is the complete answer to your question.", max_rounds=5)
    assert not any(e.get("type") == "intent_nudge_exhausted" for e in events), events


def _assert_guard_log_safe(caplog, *, structural, secret="secret123"):
    """The guard's own structural log line fired, and that record carries no raw
    secret. Scoped to the guard's records on purpose: an unrelated, pre-existing
    round-summary log echoes raw model text and is out of scope for this PR."""
    records = [r for r in caplog.records if structural in r.getMessage()]
    assert records, caplog.text
    for r in records:
        assert secret not in r.getMessage(), r.getMessage()


def test_intent_nudge_logging_does_not_leak_secret(monkeypatch, caplog):
    # The model announces an action (no tool call) with a secret in the text.
    # The nudge logger must record only structural metadata, never the matched
    # phrase — so the credential never lands in journalctl.
    _patch_common(monkeypatch)
    with caplog.at_level(logging.INFO, logger="src.agent_loop"):
        events = _run_loop(monkeypatch, "Let me check api_key=secret123 now", max_rounds=5)
    assert any(e.get("type") == "intent_nudge_exhausted" for e in events), events
    _assert_guard_log_safe(caplog, structural="intent-without-action nudge")


def test_loop_breaker_logging_does_not_leak_secret(monkeypatch, caplog):
    # A repeated tool command carrying a secret trips the loop-breaker. The
    # structural log must not contain `_sig` / raw tool-call content.
    _patch_common(monkeypatch)
    with caplog.at_level(logging.INFO, logger="src.agent_loop"):
        events = _run_loop(monkeypatch, "```bash\necho api_key=secret123\n```", max_rounds=8)
    assert any(e.get("type") == "loop_breaker_triggered" for e in events), events
    _assert_guard_log_safe(caplog, structural="loop-breaker tripped")


def test_redacts_sensitive_tool_output_before_surfacing():
    text = al._redact_sensitive_text(
        "password: private-value\n"
        "api_key=private-key\n"
        "Authorization: Bearer private-token\n"
        "normal output"
    )

    assert "private-value" not in text
    assert "private-key" not in text
    assert "private-token" not in text
    assert "password: [redacted]" in text
    assert "api_key=[redacted]" in text
    assert "Authorization: Bearer [redacted]" in text
    assert "normal output" in text


_GCP_API_KEY_SAMPLE = "AI" + "za" + ("A" * 35)

# (input, secret substring that must be gone, expected substring that must remain)
_REDACTION_CASES = [
    ("Authorization: Bearer abc123tok", "abc123tok", "Authorization: Bearer [redacted]"),
    ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz", "Authorization: Basic [redacted]"),
    # Quoted Authorization value (spaces) must be redacted whole.
    ('Authorization: Bearer "two word secret"', "two word secret", "Authorization: Bearer [redacted]"),
    # Escaped quote inside a quoted secret must not leak the tail.
    (r'password="abc\"def secret"', "def secret", "password=[redacted]"),
    # URL password containing a colon must still be redacted whole.
    ("postgres://user:pa:ss@host/db", "pa:ss", "postgres://[redacted]@host/db"),
    # Provider-shaped bare tokens.
    ("token is hf_abcdefghij1234567890XYZ", "hf_abcdefghij1234567890XYZ", "[redacted]"),
    ("key " + _GCP_API_KEY_SAMPLE, _GCP_API_KEY_SAMPLE, "[redacted]"),
    ("Cookie: session=abc123secret", "abc123secret", "Cookie: [redacted]"),
    ("Set-Cookie: sid=xyz789; HttpOnly", "xyz789", "Set-Cookie: [redacted]"),
    ("postgres://user:pa55word@host/db", "pa55word", "postgres://[redacted]@host/db"),
    ("client_secret=supersecretvalue", "supersecretvalue", "client_secret=[redacted]"),
    ("OPENAI_API_KEY=abcd1234deadbeef", "abcd1234deadbeef", "OPENAI_API_KEY=[redacted]"),
    # Quoted multi-word env value must be fully redacted, not clipped at the space.
    ('OPENAI_API_KEY="two word secret"', "two word secret", "OPENAI_API_KEY=[redacted]"),
    ('password: "my secret value"', "my secret value", "password: [redacted]"),
    ("here is sk-abcdefghij1234567890", "sk-abcdefghij1234567890", "[redacted]"),
    (
        "-----BEGIN PRIVATE KEY-----\nMIIfakeKEYbody\n-----END PRIVATE KEY-----",
        "MIIfakeKEYbody",
        "[redacted private key]",
    ),
]


@pytest.mark.parametrize("raw, secret, expected", _REDACTION_CASES)
def test_redaction_covers_requested_secret_shapes(raw, secret, expected):
    out = al._redact_sensitive_text(raw)
    assert secret not in out, out
    assert expected in out, out


@pytest.mark.parametrize("raw", [
    "the build completed in 3.2s with 0 errors",
    "password reset email sent to the user",
    "Listing 5 files: a.py b.py c.py d.py e.py",
    "https://example.com/path?page=2",
    # Benign uppercase names that merely end in KEY must not be redacted.
    "MONKEY=banana",
    "TURKEY=dinner",
])
def test_redaction_keeps_normal_output_readable(raw):
    assert al._redact_sensitive_text(raw) == raw


def test_redacts_before_truncating():
    # A secret near the start must be gone even if truncation would otherwise
    # only clip the tail — redaction runs first.
    raw = "api_key=topsecretvalue " + ("x" * 50_000)
    out = al._truncate(al._redact_sensitive_text(raw))
    assert "topsecretvalue" not in out
    assert "api_key=[redacted]" in out


def _run_tool_result(monkeypatch, tool, exec_result, max_rounds=2):
    """Drive one tool round whose execution returns `exec_result`, and collect
    the streamed events. Used to assert restored per-tool-result emissions."""
    _patch_common(monkeypatch)

    async def _fake_exec(block, *a, **k):
        return (tool, exec_result)
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)

    round_text = f"```{tool}\n{{}}\n```"

    async def _fake_stream(_candidates, messages, **kwargs):
        yield f'data: {json.dumps({"delta": round_text})}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    gen = al.stream_agent_loop(
        "http://x/v1", "m",
        [{"role": "user", "content": "do something"}],
        max_rounds=max_rounds,
        relevant_tools={tool},
    )
    return _types(_collect(gen))


def test_restores_doc_suggestions_event(monkeypatch):
    events = _run_tool_result(
        monkeypatch, "suggest_document",
        {"action": "suggest", "doc_id": "d1", "suggestions": [{"text": "x"}], "exit_code": 0},
    )
    assert any(e.get("type") == "doc_suggestions" for e in events), events


def test_restores_doc_update_event(monkeypatch):
    events = _run_tool_result(
        monkeypatch, "edit_document",
        {"action": "edit", "doc_id": "d1", "content": "body", "version": 2,
         "title": "T", "language": "md", "exit_code": 0},
    )
    # A native document block also emits doc_update AFTER tool_output, so a plain
    # "any doc_update" check would pass even if the restored generic block were
    # gone. Prove the restored block fires BEFORE the first tool_output.
    types = [e.get("type") for e in events]
    assert "doc_update" in types, events
    assert "tool_output" in types, events
    assert types.index("doc_update") < types.index("tool_output"), types


def test_restores_ui_control_event(monkeypatch):
    events = _run_tool_result(
        monkeypatch, "ui_control",
        {"ui_event": "toggle", "toggle_name": "bash", "state": "off", "exit_code": 0},
    )
    assert any(e.get("type") == "ui_control" for e in events), events


def test_restores_plan_update_event(monkeypatch):
    events = _run_tool_result(
        monkeypatch, "update_plan",
        {"plan_update": {"steps": [{"text": "step", "done": True}]}, "exit_code": 0},
    )
    assert any(e.get("type") == "plan_update" for e in events), events


def test_restores_ask_user_event_and_persists_question(monkeypatch):
    events = _run_tool_result(
        monkeypatch, "ask_user",
        {"ask_user": {"question": "Which option?", "options": [{"label": "A"}, {"label": "B"}]},
         "exit_code": 0},
    )
    # Exactly one ask_user event — not re-emitted on a follow-up round.
    _ask_events = [e for e in events if e.get("type") == "ask_user"]
    assert len(_ask_events) == 1, events
    # The question is streamed as assistant text so it persists for replay.
    # Upstream prepends "\n\n" when full_response already holds streamed text,
    # so match on containment — and it must be streamed exactly once.
    _q_deltas = [e for e in events if "Which option?" in (e.get("delta") or "")]
    assert len(_q_deltas) == 1, events
    # Setting `_awaiting_user` breaks the loop, so the turn does NOT advance into
    # another agent round (which would emit an agent_step event) after the ask.
    assert not any(e.get("type") == "agent_step" for e in events), events


def test_redacts_command_display_in_streamed_events(monkeypatch):
    # A tool command line can carry a secret. The streamed command display
    # (tool_start / tool_output) must be redacted, even though the real command
    # passed to execution is left untouched.
    _patch_common(monkeypatch)

    round_text = "```bash\necho api_key=secret123\n```"

    async def _fake_stream(_candidates, messages, **kwargs):
        yield f'data: {json.dumps({"delta": round_text})}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    gen = al.stream_agent_loop(
        "http://x/v1", "m",
        [{"role": "user", "content": "run it"}],
        max_rounds=2,
        relevant_tools={"bash"},
    )
    events = _types(_collect(gen))
    cmds = [e for e in events if e.get("type") in ("tool_start", "tool_output")]
    assert cmds, events
    assert all("secret123" not in (e.get("command") or "") for e in cmds), cmds
    assert any("api_key=[redacted]" in (e.get("command") or "") for e in cmds), cmds


def test_redacts_live_tool_progress_tail(monkeypatch):
    # A secret in the live progress tail must be redacted before streaming —
    # otherwise it flashes by before the (already redacted) final tool_output.
    _patch_common(monkeypatch)

    async def _fake_exec(block, *a, **k):
        await k["progress_cb"]({"tail": "api_key=secret123", "elapsed_s": 1})
        return ("bash", {"output": "done", "exit_code": 0})
    monkeypatch.setattr(al, "execute_tool_block", _fake_exec, raising=False)

    round_text = "```bash\necho hi\n```"

    async def _fake_stream(_candidates, messages, **kwargs):
        yield f'data: {json.dumps({"delta": round_text})}\n\n'
        yield "data: [DONE]\n\n"
    monkeypatch.setattr(al, "stream_llm_with_fallback", _fake_stream, raising=False)

    gen = al.stream_agent_loop(
        "http://x/v1", "m",
        [{"role": "user", "content": "run it"}],
        max_rounds=2,
        relevant_tools={"bash"},
    )
    events = _types(_collect(gen))
    prog = [e for e in events if e.get("type") == "tool_progress"]
    assert prog, events
    assert all("secret123" not in (e.get("tail") or "") for e in prog), prog
    assert any("api_key=[redacted]" in (e.get("tail") or "") for e in prog), prog
    # Other fields are preserved.
    assert any(e.get("elapsed_s") == 1 for e in prog), prog
