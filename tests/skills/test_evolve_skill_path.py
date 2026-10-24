"""--skill-path lets evolve_skill target a skill anywhere on disk."""

from click.testing import CliRunner

from evolution.skills.evolve_skill import main

SKILL = "---\nname: outside-skill\ndescription: Lives outside the hermes repo\n---\n\n# Body\nDo things.\n"


def _run(tmp_path, *args):
    return CliRunner().invoke(main, [*args, "--dry-run", "--hermes-repo", str(tmp_path)])


def test_skill_path_to_directory(tmp_path):
    d = tmp_path / "ext" / "outside-skill"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL)
    r = _run(tmp_path, "--skill-path", str(d))
    assert r.exit_code == 0, r.output
    assert "Name: outside-skill" in r.output


def test_skill_path_to_file(tmp_path):
    f = tmp_path / "SKILL.md"
    f.write_text(SKILL)
    r = _run(tmp_path, "--skill-path", str(f))
    assert r.exit_code == 0, r.output


def test_missing_skill_path_fails(tmp_path):
    r = _run(tmp_path, "--skill-path", str(tmp_path / "nope"))
    assert r.exit_code == 1
    assert "not found" in r.output


def test_requires_skill_or_skill_path(tmp_path):
    r = _run(tmp_path)
    assert r.exit_code != 0
    assert "--skill" in r.output


def test_folder_evolves_all_skills_with_cap(tmp_path, monkeypatch):
    import threading
    from click.testing import CliRunner
    from evolution.skills import evolve_skill as m

    for n in "abc":
        (tmp_path / n).mkdir()
        (tmp_path / n / "SKILL.md").write_text("x")
    live = peak = 0
    lock = threading.Lock()

    def fake_run(cmd, **kw):
        nonlocal live, peak
        import time, types
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.1)
        with lock:
            live -= 1
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    r = CliRunner().invoke(m.main, ["--skill-path", str(tmp_path), "--max-concurrent", "2", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "3 ok" in r.output and peak == 2


def test_folder_forwards_apply(tmp_path, monkeypatch):
    import types
    from click.testing import CliRunner
    from evolution.skills import evolve_skill as m

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "SKILL.md").write_text("x")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    r = CliRunner().invoke(m.main, ["--skill-path", str(tmp_path), "--apply", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "--apply" in calls[0]


def test_apply_replaces_unchanged_source_atomically(tmp_path):
    from evolution.skills.evolve_skill import _apply_evolved

    source = tmp_path / "SKILL.md"
    source.write_text(SKILL, encoding="utf-8")
    assert _apply_evolved(source, SKILL, SKILL.replace("Do things.", "Do better things."))
    assert "Do better things." in source.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_apply_rejects_empty_body_and_changed_source(tmp_path):
    import pytest
    from evolution.skills.evolve_skill import _apply_evolved

    source = tmp_path / "SKILL.md"
    source.write_text(SKILL, encoding="utf-8")
    with pytest.raises(ValueError, match="empty body"):
        _apply_evolved(source, SKILL, "---\nname: outside-skill\ndescription: x\n---\n")
    source.write_text(SKILL + "changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed during evolution"):
        _apply_evolved(source, SKILL, SKILL.replace("Do things.", "Do better things."))
    assert source.read_text(encoding="utf-8").endswith("changed\n")


def test_apply_rejects_locked_source_and_preserves_mode(tmp_path):
    import stat
    import pytest
    from evolution.skills.evolve_skill import _apply_evolved

    source = tmp_path / "SKILL.md"
    source.write_text(SKILL, encoding="utf-8")
    source.chmod(stat.S_IRUSR | stat.S_IWUSR)
    original_mode = stat.S_IMODE(source.stat().st_mode)
    lock = tmp_path / ".SKILL.md.evolution.lock"
    lock.write_text("")
    with pytest.raises(RuntimeError, match="already being evolved"):
        _apply_evolved(source, SKILL, SKILL.replace("Do things.", "Do better things."))
    lock.unlink()
    _apply_evolved(source, SKILL, SKILL.replace("Do things.", "Do better things."))
    assert stat.S_IMODE(source.stat().st_mode) == original_mode


def test_multiple_folders_dedupe_same_skill_name_by_first_root(tmp_path, monkeypatch):
    import types
    from click.testing import CliRunner
    from evolution.skills import evolve_skill as m

    for root in ("first", "second"):
        (tmp_path / root / "same").mkdir(parents=True)
        (tmp_path / root / "same" / "SKILL.md").write_text("x")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[cmd.index("--skill-path") + 1])
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    arg = f"{tmp_path / 'first'},{tmp_path / 'second'}"
    r = CliRunner().invoke(m.main, ["--skill-path", arg, "--apply", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert calls == [str((tmp_path / "first" / "same").resolve())]


def test_multiple_folders_deduped(tmp_path, monkeypatch):
    import types
    from click.testing import CliRunner
    from evolution.skills import evolve_skill as m

    for d in ("a/x", "a/y", "b/z"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "SKILL.md").write_text("x")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[cmd.index("--skill-path") + 1])
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    # a overlaps itself via a/x, and a/x is listed twice
    arg = f"{tmp_path / 'a'}, {tmp_path / 'b'},{tmp_path / 'a' / 'x'}"
    r = CliRunner().invoke(m.main, ["--skill-path", arg, "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "3 ok" in r.output and len(calls) == 3 and len(set(calls)) == 3
