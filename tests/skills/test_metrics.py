from types import SimpleNamespace

import dspy

from evolution.skills.evolve_skill import _folder_summary, _merge_tokens, _phase, _tokens


def test_tokens_handles_both_usage_shapes():
    out = _tokens({"a": {"prompt_tokens": 3, "completion_tokens": 2}, "b": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 9}})
    assert out == {"a": {"input": 3, "output": 2, "total": 5}, "b": {"input": 1, "output": 1, "total": 9}}


def test_phase_records_seconds_even_on_error():
    store = {}
    try:
        with _phase(store, "x"):
            raise ValueError
    except ValueError:
        pass
    assert store["x"]["seconds"] >= 0 and store["x"]["tokens"] == {}


def test_folder_summary_aggregates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ph = {"dataset": {"seconds": 1, "tokens": {"m": {"input": 1, "output": 2, "total": 3}}}}
    m = {"duration_seconds": 5, "phases": ph, "tokens": {"total": {"input": 1, "output": 2, "total": 3}, "by_model": _merge_tokens(ph)}}
    _folder_summary([("s1", 0, m), ("s2", 1, None)], 6.0)
    import json
    f = next((tmp_path / "output").glob("run_summary_*.json"))
    d = json.loads(f.read_text())
    assert d["skills_with_metrics"] == 1 and d["tokens"]["total"]["total"] == 3
