"""
test_pipeline.py – Smoke test for RepoSherpa backend.

Run this before launching the Streamlit UI to verify:
  1. GitHub API access works
  2. Gemma 4 responds correctly
  3. Tool calling + JSON parsing work end-to-end

Usage:
    python test_pipeline.py
"""

import sys
import json
from dotenv import load_dotenv
load_dotenv()

from github_client import parse_repo, GitHubClient, RepoNotFound, RateLimitError
from agent import build_context, find_candidates, get_details, GemmaError, NotEnoughCandidates


def main():
    # ── Config ──
    test_repo = sys.argv[1] if len(sys.argv) > 1 else 'freeCodeCamp/freeCodeCamp'
    test_skills = sys.argv[2] if len(sys.argv) > 2 else 'Python, JavaScript, basic React'
    test_level = 'Beginner'
    model = 'gemma-4-31b-it'

    print(f"{'='*60}")
    print(f"RepoSherpa Pipeline Test")
    print(f"{'='*60}")
    print(f"Repo:   {test_repo}")
    print(f"Skills: {test_skills}")
    print(f"Level:  {test_level}")
    print(f"Model:  {model}")
    print()

    # ── Test 1: parse_repo ──
    print("▸ Testing parse_repo...")
    cases = [
        ('https://github.com/owner/repo', 'owner/repo'),
        ('https://github.com/owner/repo.git', 'owner/repo'),
        ('https://github.com/owner/repo/', 'owner/repo'),
        ('github.com/owner/repo', 'owner/repo'),
        ('owner/repo', 'owner/repo'),
    ]
    for url, expected in cases:
        result = parse_repo(url)
        assert result == expected, f"parse_repo('{url}') = '{result}', expected '{expected}'"
    print("  ✅ parse_repo works correctly")

    # ── Test 2: Bad URL ──
    print("▸ Testing parse_repo with bad input...")
    try:
        parse_repo('not-a-url')
        print("  ❌ Should have raised ValueError")
    except ValueError:
        print("  ✅ Correctly raises ValueError")

    # ── Test 3: Build context ──
    print(f"\n▸ Building context for {test_repo}...")
    try:
        ctx = build_context(
            test_repo, test_skills, test_level,
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        print(f"  ✅ Context built:")
        print(f"     Repo:      {ctx['repo']}")
        print(f"     Branch:    {ctx['branch']}")
        print(f"     Files:     {len(ctx['tree'])}")
        print(f"     Issues:    {len(ctx['issues'])}")
        print(f"     README:    {len(ctx['readme'])} chars")
        print(f"     CONTRIB:   {len(ctx['contributing'])} chars")
    except RepoNotFound:
        print(f"  ❌ Repo not found: {test_repo}")
        sys.exit(1)
    except RateLimitError:
        print("  ❌ Rate limit hit. Set GITHUB_TOKEN in .env")
        sys.exit(1)

    # ── Test 4: Find candidates ──
    print(f"\n▸ Finding candidates (this may take 30-60s)...")
    try:
        cands = find_candidates(
            ctx, model=model,
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        print(f"  ✅ Found {len(cands)} candidates:\n")
        for c in cands:
            source_tag = f"Existing #{c.get('issue_number')}" if c['source'] == 'existing' else "New (Gemma)"
            print(f"  [{c['id']}] {c['title']}")
            print(f"      Source: {source_tag}")
            print(f"      Difficulty: {c.get('difficulty')} | Time: {c.get('estimated_time')}")
            print(f"      Skills: {', '.join(c.get('skill_match', []))}")
            paths = [w.get('path', '') for w in c.get('where', [])]
            print(f"      Where: {', '.join(paths)}")
            print(f"      Change: {c.get('what_to_change', '')}")
            print()
    except NotEnoughCandidates as e:
        print(f"  ⚠️ {e}")
        sys.exit(1)
    except GemmaError as e:
        print(f"  ❌ Gemma error: {e}")
        sys.exit(1)

    # ── Test 5: Get details for first candidate ──
    print(f"▸ Getting details for {cands[0]['id']}: {cands[0]['title']}...")
    try:
        detail = get_details(
            ctx, cands[0], model=model,
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        print(f"  ✅ Detail object received:")
        print(f"     Problem: {detail.get('problem', '')[:100]}...")
        print(f"     Where: {len(detail.get('where', []))} locations")
        print(f"     Steps: {len(detail.get('what_to_change', []))} changes")
        print(f"     Verify: {len(detail.get('how_to_verify', []))} checks")
        if detail.get('pr'):
            print(f"     PR branch: {detail['pr'].get('branch_name', '')}")
        if detail.get('warnings'):
            print(f"     Warnings: {detail['warnings']}")
        print()
        print("  Full detail JSON:")
        print(json.dumps(detail, indent=2)[:2000])
    except GemmaError as e:
        print(f"  ❌ Gemma error: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # ── Tool log ──
    print(f"\n▸ Tool log ({len(ctx.get('tool_log', []))} calls):")
    for entry in ctx.get('tool_log', []):
        print(f"  {entry['tool']}({entry['args']})")

    print(f"\n{'='*60}")
    print("Pipeline test complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
