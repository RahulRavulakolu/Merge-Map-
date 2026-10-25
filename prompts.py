"""
prompts.py – System prompts and schemas for RepoSherpa's Gemma 4 integration.
"""

# ──────────────────────────────────────────────
# JSON schemas (embedded in prompts)
# ──────────────────────────────────────────────

CANDIDATE_SCHEMA = """{
  "candidates": [
    {
      "id": "c1",
      "source": "existing | suggested",
      "issue_number": 0,
      "title": "",
      "summary": "",
      "where": [{"path": "", "lines": "12-30", "note": ""}],
      "what_to_change": "",
      "skill_match": [""],
      "difficulty": "easy | medium",
      "estimated_time": ""
    }
  ]
}"""

DETAIL_SCHEMA = """{
  "id": "c1",
  "source": "existing | suggested",
  "issue_number": 0,
  "title": "",
  "url": "",
  "problem": "",
  "where": [{"path": "", "lines": "12-30", "snippet": "", "explanation": ""}],
  "what_to_change": [{"path": "", "change": ""}],
  "how_to_verify": [""],
  "setup_steps": [""],
  "difficulty": "",
  "estimated_time": "",
  "skill_match": [""],
  "issue_draft": {"title": "", "body_markdown": ""},
  "pr": {"branch_name": "", "commit_message": "", "title": "", "description_markdown": ""},
  "warnings": [""]
}"""


# ──────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────

STAGE1_SYSTEM = """You are RepoSherpa, a mentor who helps first-time open-source contributors find work that fits them. You get a repository's README, contributing guide, file tree and open issues, plus the contributor's skills and level. Return 5 candidate tasks ranked by fit with the contributor's skills (3 only if the repo is tiny). Use existing open issues when they fit, preferring unassigned and clearly scoped ones, and fill the remaining slots with new problems you find by reading code with the tools: leftover TODO or FIXME notes, missing or outdated docs, untested functions, small bugs, missing error handling. A new problem must point at real lines in a file you have read, must not duplicate an existing open issue, and must be doable in a few hours. Never invent file paths or line numbers. Return ONLY valid JSON matching the schema, with no markdown fences and no extra text."""

STAGE2_SYSTEM = """You are RepoSherpa. You are given one candidate task and the repository context. Read the files you need with the tools (lines are numbered like L12:). Produce the full detail object: explain the problem, show exactly where it is (path, line range and a snippet copied verbatim, at most 15 lines), list the changes step by step with the exact file for each, say how to verify the fix, and draft the PR. For a suggested (new) issue also write issue_draft: a concise, specific GitHub issue with the problem, the location, the proposed fix and a sentence saying the author would like to work on it. Never invent paths, lines or code. Return ONLY valid JSON matching the schema, with no markdown fences and no extra text."""

RETRY_MESSAGE = "Your previous output was not valid JSON. Return only the JSON object, with no markdown fences, no explanation, no extra text."


# ──────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────

def build_stage1_prompt(ctx: dict) -> str:
    """
    Build the full user prompt for stage 1 (find_candidates).
    """
    issues_text = _format_issues(ctx.get('issues', []))
    tree_text = '\n'.join(ctx.get('tree', []))

    # Cap total prompt to ~300KB
    readme = ctx.get('readme', '')[:20000]
    contributing = ctx.get('contributing', '')[:15000]
    if len(tree_text) > 80000:
        tree_text = tree_text[:80000] + '\n... (truncated)'

    prompt = f"""=== CONTRIBUTOR ===
Skills: {ctx.get('skills', 'Not specified')}
Level: {ctx.get('level', 'Beginner')}

=== README ===
{readme}

=== CONTRIBUTING ===
{contributing}

=== FILE TREE ===
{tree_text}

=== OPEN ISSUES ===
{issues_text}

=== INSTRUCTIONS ===
Using the tools available to you (read_file and find_files), examine the codebase to find 5 candidate tasks for this contributor. Use existing open issues when they match the contributor's skills, and fill remaining slots with new problems you find in the code. Aim for at least 2 existing issues when the repo has them.

Return ONLY a valid JSON object matching this schema:
{CANDIDATE_SCHEMA}

Rules:
- "source" must be "existing" or "suggested"
- For existing issues, set issue_number to the real issue number
- For suggested issues, set issue_number to null
- Every "where" entry must reference a real file path from the file tree
- "lines" should be a range like "12-30"
- Never invent file paths or line numbers
- Rank by fit with the contributor's skills
- No markdown fences, no extra text, ONLY the JSON object"""

    return prompt


def build_stage2_prompt(ctx: dict, candidate: dict) -> str:
    """
    Build the full user prompt for stage 2 (get_details).
    """
    import json
    candidate_json = json.dumps(candidate, indent=2)
    issues_text = _format_issues(ctx.get('issues', []))
    tree_text = '\n'.join(ctx.get('tree', []))

    readme = ctx.get('readme', '')[:15000]
    contributing = ctx.get('contributing', '')[:10000]
    if len(tree_text) > 60000:
        tree_text = tree_text[:60000] + '\n... (truncated)'

    prompt = f"""=== CANDIDATE ===
{candidate_json}

=== CONTRIBUTOR ===
Skills: {ctx.get('skills', 'Not specified')}
Level: {ctx.get('level', 'Beginner')}

=== README ===
{readme}

=== CONTRIBUTING ===
{contributing}

=== FILE TREE ===
{tree_text}

=== OPEN ISSUES ===
{issues_text}

=== INSTRUCTIONS ===
Read the relevant files using the tools (read_file, find_files) and produce a complete detail object for this candidate task.

Return ONLY a valid JSON object matching this schema:
{DETAIL_SCHEMA}

Rules:
- "snippet" must be copied verbatim from the file (at most 15 lines), without L12: prefixes
- "what_to_change" is an ordered list of steps, each naming one file and the exact change
- For "suggested" source, include "issue_draft" with a concise GitHub issue body containing: the problem, the location (path and lines), the proposed fix, and a sentence saying the author would like to work on it
- For "existing" source, set "issue_draft" to null
- "pr" must include branch_name, commit_message, title, and description_markdown
- Never invent paths, lines, or code
- No markdown fences, no extra text, ONLY the JSON object"""

    return prompt


def _format_issues(issues: list[dict]) -> str:
    """Format issues for inclusion in prompts."""
    if not issues:
        return "(No open issues found)"

    parts = []
    for iss in issues:
        labels_str = ', '.join(iss.get('labels', []))
        assignee = iss.get('assignee') or 'unassigned'
        comments = iss.get('comments', 0)
        header = f"#{iss['number']} | {iss['title']} | labels: {labels_str} | assignee: {assignee} | comments: {comments}"
        body = iss.get('body', '') or '(no description)'
        parts.append(f"{header}\n{body}\n---")

    return '\n'.join(parts)
