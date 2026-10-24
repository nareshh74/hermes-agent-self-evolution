"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.lm import make_lm
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import LLMJudge
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def _apply_evolved(source: Path, baseline: str, evolved: str) -> bool:
    """Replace an unchanged source skill under an exclusive per-skill lock."""
    body = evolved.split("---", 2)[-1].strip() if evolved.startswith("---") else evolved.strip()
    if not body:
        raise ValueError("refusing to apply evolved skill with empty body")
    if not re.search(r"^name:\s*\S+", evolved, re.M) or not re.search(r"^description:\s*\S+", evolved, re.M):
        raise ValueError("refusing to apply evolved skill without name and description")
    lock = source.with_name(f".{source.name}.evolution.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"source is already being evolved: {source}") from exc
    tmp_name = None
    try:
        os.close(fd)
        if source.read_text(encoding="utf-8") != baseline:
            raise RuntimeError(f"source changed during evolution: {source}")
        mode = source.stat().st_mode
        fd, tmp_name = tempfile.mkstemp(prefix=f".{source.name}.", suffix=".tmp", dir=source.parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(evolved)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, source)
        tmp_name = None
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)
        lock.unlink(missing_ok=True)
    return True


def _tokens(usage: dict) -> dict:
    """Collapse dspy usage ({model: provider-specific dict}) to input/output/total per model."""
    out = {}
    for model, u in usage.items():
        i = u.get("prompt_tokens", u.get("input_tokens")) or 0
        o = u.get("completion_tokens", u.get("output_tokens")) or 0
        out[model] = {"input": i, "output": o, "total": u.get("total_tokens") or i + o}
    return out


@contextmanager
def _phase(store: dict, name: str):
    """Record wall seconds and per-model tokens (cache hits excluded) for one phase.

    Tokens are split per model, so optimizer vs eval needs different models.
    """
    t = time.time()
    with dspy.track_usage() as u:
        try:
            yield
        finally:
            store[name] = {"seconds": round(time.time() - t, 2), "tokens": _tokens(u.get_total_tokens())}


def _sum_tokens(per_model: dict) -> dict:
    return {k: sum(m[k] for m in per_model.values()) for k in ("input", "output", "total")}


def _merge_tokens(phases: dict) -> dict:
    merged = {}
    for ph in phases.values():
        for model, t in ph["tokens"].items():
            m = merged.setdefault(model, {"input": 0, "output": 0, "total": 0})
            for k in m:
                m[k] += t[k]
    return merged


def evolve(
    skill_name: Optional[str] = None,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    skill_path: Optional[str] = None,
    apply: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    run_start = time.time()
    phases: dict = {}
    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name or skill_path}[/bold]\n")

    if skill_path:
        # Explicit path: a SKILL.md file or a skill directory, anywhere on disk
        found = Path(skill_path).expanduser().resolve()
        if found.is_dir():
            found = found / "SKILL.md"
        if not found.is_file():
            console.print(f"[red]✗ Skill file not found: {found}[/red]")
            sys.exit(1)
        skill_name = skill_name or found.parent.name
    else:
        found = find_skill(skill_name, config.hermes_agent_path)
        if not found:
            console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
            sys.exit(1)

    skill = load_skill(found)
    try:
        shown = found.relative_to(config.hermes_agent_path)
    except ValueError:
        shown = found
    console.print(f"  Loaded: {shown}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    with _phase(phases, "dataset"):
        if eval_source == "golden" and dataset_path:
            dataset = GoldenDatasetLoader.load(Path(dataset_path))
            console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
        elif eval_source == "sessiondb":
            save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
            dataset = build_dataset_from_external(
                skill_name=skill_name,
                skill_text=skill["raw"],
                sources=["claude-code", "copilot", "hermes"],
                output_path=save_path,
                model=eval_model,
            )
            if not dataset.all_examples:
                console.print("[red]✗ No relevant examples found from session history[/red]")
                sys.exit(1)
            console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
        elif eval_source == "synthetic":
            builder = SyntheticDatasetBuilder(config)
            dataset = builder.generate(
                artifact_text=skill["raw"],
                artifact_type="skill",
            )
            # Save for reuse
            save_path = Path("datasets") / "skills" / skill_name
            dataset.save(save_path)
            console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
            console.print(f"  Saved to {save_path}/")
        elif dataset_path:
            dataset = EvalDataset.load(Path(dataset_path))
            console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
        else:
            console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
            sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy
    lm = make_lm(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    judge = LLMJudge(config)

    def judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        if not getattr(pred, "output", "").strip():
            return 0.0
        key = (gold.task_input, gold.expected_behavior, pred.output, pred.skill_text)
        # GEPA requests feedback again for the same captured trajectory.
        if getattr(pred, "_evolution_grade_key", None) != key:
            pred._evolution_grade = judge.score(
                task_input=gold.task_input,
                expected_behavior=gold.expected_behavior,
                agent_output=pred.output,
                skill_text=pred.skill_text,
            )
            pred._evolution_grade_key = key
        return pred._evolution_grade.composite

    def optimization_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        candidate = reassemble_skill(skill["frontmatter"], pred.skill_text)
        violations = [c for c in validator.validate_all(candidate, "skill", baseline_text=skill["raw"]) if not c.passed]
        if violations:
            score = 0.0
            feedback = "Hard constraints failed; repair these before optimizing quality: " + "; ".join(
                f"{c.constraint_name}: {c.message}" for c in violations
            )
        else:
            score = judge_metric(gold, pred, trace)
            feedback = getattr(getattr(pred, "_evolution_grade", None), "feedback", "") or f"Quality score: {score}. Preserve required behavior."
        if pred_name is not None:
            return dspy.Prediction(score=score, feedback=feedback)
        return score

    reflection_lm = make_lm(optimizer_model)
    gepa = getattr(dspy, "GEPA", None)
    if gepa is not None:
        optimizer = gepa(
            metric=optimization_metric,
            # Bound the budget by metric calls, not optimizer steps.
            max_metric_calls=max(1, iterations) * max(1, len(valset)) + len(valset),
            reflection_lm=reflection_lm,
        )
    else:
        console.print("[yellow]GEPA not available, falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=optimization_metric,
            prompt_model=reflection_lm,
            task_model=lm,
            auto="light",
        )
    with _phase(phases, "optimize"):
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    if run_tests:
        test_result = validator.run_test_suite(repo)
        evolved_constraints.append(test_result)
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        raise click.ClickException("Evolved skill failed hard constraints")

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    with _phase(phases, "holdout"):
        for ex in holdout_examples:
            # Score baseline
            with dspy.context(lm=lm):
                baseline_pred = baseline_module(task_input=ex.task_input)
                baseline_score = judge_metric(ex, baseline_pred)
                baseline_scores.append(baseline_score)

                evolved_pred = optimized_module(task_input=ex.task_input)
                evolved_score = judge_metric(ex, evolved_pred)
                evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "apply_requested": apply,
        "applied": False,
        "source_path": str(found),
        "duration_seconds": round(time.time() - run_start, 2),
        "phases": phases,
        "tokens": {"total": _sum_tokens(_merge_tokens(phases)), "by_model": _merge_tokens(phases)},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if apply and improvement > 0 and all_pass:
        metrics["applied"] = _apply_evolved(found, skill["raw"], evolved_full)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"METRICS_PATH={output_dir / 'metrics.json'}")

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


def _folder_summary(results: list, wall: float):
    """Print and save total/per-skill duration and tokens (dataset, optimize, holdout)."""
    rows = {n: m for n, _, m in results if m}
    by_model: dict = {}
    for m in rows.values():
        for model, t in m["tokens"]["by_model"].items():
            b = by_model.setdefault(model, {"input": 0, "output": 0, "total": 0})
            for k in b:
                b[k] += t[k]
    summary = {
        "wall_seconds": round(wall, 2),
        "skills_with_metrics": len(rows),
        "skill_seconds_sum": round(sum(m["duration_seconds"] for m in rows.values()), 2),
        "tokens": {"total": _sum_tokens(by_model), "by_model": by_model},
        "skills": {n: {"duration_seconds": m["duration_seconds"], "phases": m["phases"], "tokens": m["tokens"]}
                   for n, m in rows.items()},
    }
    table = Table(title="Run metrics")
    for c in ("Skill", "Seconds", "Dataset s", "Optimize s", "Holdout s", "Tokens"):
        table.add_column(c, justify="right" if c != "Skill" else "left")
    for n, m in sorted(rows.items()):
        ph = m["phases"]
        table.add_row(n, f"{m['duration_seconds']:.0f}", *(f"{ph.get(k, {}).get('seconds', 0):.0f}" for k in ("dataset", "optimize", "holdout")),
                      f"{m['tokens']['total']['total']:,}")
    table.add_row("TOTAL", f"{wall:.0f} wall", "", "", "", f"{summary['tokens']['total']['total']:,}")
    console.print(table)
    for model, t in by_model.items():
        console.print(f"  {model}: in {t['input']:,} / out {t['output']:,} / total {t['total']:,}")
    out = Path("output") / f"run_summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    console.print(f"  Summary saved to {out}")


def _evolve_folder(paths: list, max_concurrent: int, opts: dict):
    """Evolve every skill under the given paths in parallel (one subprocess each, capped; extras queue).

    Each path is a folder of skills or a single skill (dir or SKILL.md). Overlapping
    paths are deduped. Subprocesses because dspy.configure is global/thread-bound.
    """
    found = {}
    for p in paths:
        candidates = ([p] if p.is_file() or (p / "SKILL.md").exists()
                      else sorted(f.parent for f in p.rglob("SKILL.md")))
        for candidate in candidates:
            d = candidate.parent if candidate.is_file() else candidate
            found.setdefault(d.name, d)
    skills = list(found.values())
    if not skills:
        raise click.ClickException("No SKILL.md found under " + ", ".join(map(str, paths)))
    console.print(f"Evolving {len(skills)} skills, max {max_concurrent} concurrent")
    t0 = time.time()

    def run(d: Path):
        cmd = [sys.executable, "-m", "evolution.skills.evolve_skill", "--skill-path", str(d)]
        for k, v in opts.items():
            if v is None or v is False:
                continue
            cmd.append("--" + k.replace("_", "-"))
            if v is not True:
                cmd.append(str(v))
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        console.print(f"[{'green' if r.returncode == 0 else 'red'}]{d.name}: exit {r.returncode}[/]")
        if r.returncode:
            console.print((r.stdout + r.stderr)[-1500:], markup=False)
        m = re.search(r"^METRICS_PATH=(.+)$", r.stdout, re.M)
        return d.name, r.returncode, json.loads(Path(m.group(1).strip()).read_text()) if m else None

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        results = list(pool.map(run, skills))
    failed = [n for n, rc, _ in results if rc]
    console.print(f"Done: {len(results) - len(failed)} ok, {len(failed)} failed")
    _folder_summary(results, time.time() - t0)
    if failed:
        raise click.ClickException("Failed: " + ", ".join(failed))


@click.command()
@click.option("--skill", default=None, help="Name of the skill to evolve (looked up in hermes-agent repo)")
@click.option("--skill-path", default=None, help="Path to a SKILL.md, a skill directory, or a folder of skills; comma-separate several paths")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", envvar="EVOLUTION_OPTIMIZER_MODEL", show_default=True,
              help="Model for GEPA reflections (litellm name, or copilot/<model>)")
@click.option("--eval-model", default="openai/gpt-4.1-mini", envvar="EVOLUTION_EVAL_MODEL", show_default=True,
              help="Model for evaluations, judging and dataset generation (litellm name, or copilot/<model>)")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--apply", is_flag=True, help="Replace source SKILL.md only when holdout improves and constraints pass")
@click.option("--max-concurrent", default=4, type=click.IntRange(min=1), envvar="EVOLUTION_MAX_CONCURRENT",
              show_default=True, help="Max parallel evolutions when --skill-path is a folder of skills; the rest queue")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run, apply, skill_path, max_concurrent):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    if not skill and not skill_path:
        raise click.UsageError("Provide --skill or --skill-path")
    if skill_path:
        # ponytail: comma split, so a path containing a comma is unsupported
        paths = [Path(x.strip()).expanduser().resolve() for x in skill_path.split(",") if x.strip()]
        if len(paths) > 1 or (paths[0].is_dir() and not (paths[0] / "SKILL.md").exists()):
            _evolve_folder(paths, max_concurrent, dict(
                iterations=iterations, eval_source=eval_source, dataset_path=dataset_path,
                optimizer_model=optimizer_model, eval_model=eval_model, hermes_repo=hermes_repo,
                run_tests=run_tests, dry_run=dry_run, apply=apply))
            return
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        skill_path=skill_path,
        apply=apply,
    )


if __name__ == "__main__":
    main()
