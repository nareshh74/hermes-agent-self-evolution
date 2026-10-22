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
