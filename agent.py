"""
agent.py – Gemma 4 integration for RepoSherpa.

Stage 1: find_candidates – scans the repo and returns 3-5 task candidates.
Stage 2: get_details – reads relevant files and returns full detail for one candidate.
"""

import json
import re
import difflib
import os

from google import genai
from google.genai import types

from github_client import (
    GitHubClient, parse_repo, file_link,
    RepoNotFound, RateLimitError, GitHubError, IssuesDisabled,
)
from prompts import (
    STAGE1_SYSTEM, STAGE2_SYSTEM, RETRY_MESSAGE,
    build_stage1_prompt, build_stage2_prompt,
)


# ──────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────

class NotEnoughCandidates(Exception):
    """Raised when fewer than 3 valid candidates remain after validation."""
    pass


class GemmaError(Exception):
    """Raised when Gemma fails after retries."""
    pass


# ──────────────────────────────────────────────
# Client init
# ──────────────────────────────────────────────

def _get_client():
    """Create and return a genai Client."""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        raise GemmaError("GEMINI_API_KEY is not set. Add it to .env or environment variables.")
    return genai.Client(api_key=api_key)


# ──────────────────────────────────────────────
# Tool factory
# ──────────────────────────────────────────────

def make_tools(gh: GitHubClient, repo: str, ctx: dict, max_reads: int):
    """
    Build the read_file and find_files tools for Gemma function calling.
    Files are returned with line numbers (L1: ...) so Gemma can cite exact ranges.
    """
    reads = {'n': 0}

    def read_file(path: str) -> str:
        """Return up to 12 KB of a repo file, each line prefixed like 'L12: text'."""
        if path in ctx['file_cache']:
            return ctx['file_cache'][path]
        if reads['n'] >= max_reads:
            return 'Read limit reached.'
        reads['n'] += 1
        text = gh.get_file(repo, path)[:12000]
        if not text:
            return f'File not found or binary: {path}'
        ctx['raw_cache'][path] = text
        numbered = '\n'.join(
            f'L{i}: {line}' for i, line in enumerate(text.splitlines(), 1)
        )
        ctx['file_cache'][path] = numbered
        ctx['tool_log'].append({'tool': 'read_file', 'args': {'path': path}})
        return numbered

    def find_files(keyword: str) -> list[str]:
        """Return up to 20 repo file paths containing the keyword."""
        ctx['tool_log'].append({'tool': 'find_files', 'args': {'keyword': keyword}})
        return [p for p in ctx['tree'] if keyword.lower() in p.lower()][:20]

    return [read_file, find_files]


# ──────────────────────────────────────────────
# JSON parsing
# ──────────────────────────────────────────────

def parse_json(text: str) -> dict:
    """
    Parse JSON from Gemma's output.
    1. Strip code fences.
    2. Extract from first { to last }.
    3. json.loads.
    """
    if not text:
        raise ValueError("Empty response from Gemma.")

    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```\s*', '', cleaned)

    # Find first { to last }
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    if first_brace == -1 or last_brace == -1 or first_brace >= last_brace:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    json_str = cleaned[first_brace:last_brace + 1]
    return json.loads(json_str)


# ──────────────────────────────────────────────
# Gemma call with tool handling
# ──────────────────────────────────────────────

def _call_gemma_with_tools(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tool_functions: list,
    max_tool_rounds: int = 12,
) -> str:
    """
    Call Gemma with automatic function calling.
    Returns the final text response.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tool_functions,
        temperature=0.3,
    )

    try:
        chat = client.chats.create(
            model=model,
            config=config,
        )
        response = chat.send_message(user_prompt)
    except Exception as e:
        raise GemmaError(f"API call to {model} failed: {e}")

    # Extract text from response
    if response and response.text:
        return response.text

    # If no text, check for candidates
    if response and response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        return part.text

    raise GemmaError("Gemma returned an empty response.")


def _call_gemma_no_tools(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call Gemma without tools (fallback)."""
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.3,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )
    except Exception as e:
        raise GemmaError(f"API call to {model} failed: {e}")

    if response and response.text:
        return response.text

    raise GemmaError("Gemma returned an empty response.")


# ──────────────────────────────────────────────
# Context builder
# ──────────────────────────────────────────────

# Fallback labels when 'good first issue' yields nothing
_FALLBACK_LABEL_SETS = [
    ('good-first-issue',),
    ('help wanted',),
    ('hacktoberfest',),
    ('documentation',),
    ('beginner',),
    ('easy',),
]


def build_context(
    repo_url: str,
    skills: str,
    level: str,
    token: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Fetch all repo data and build the context dict.
    progress_callback(step_label: str) is called at each step for UI updates.
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    # Parse and validate
    repo = parse_repo(repo_url)
    gh = GitHubClient(token=token)

    # Fetch repo info
    _progress("Fetching repository info…")
    meta = gh.get_repo(repo)
    branch = meta['default_branch']

    # Fetch file tree
    _progress("Reading file tree…")
    tree = gh.get_tree(repo, branch)

    # Fetch README
    _progress("Reading README…")
    readme = gh.get_readme(repo)

    # Fetch CONTRIBUTING
    _progress("Reading CONTRIBUTING guide…")
    contributing = gh.get_contributing(repo)

    # Fetch issues with label fallback
    _progress("Reading open issues…")
    issues = gh.get_issues(repo, labels=('good first issue',))

    label_warning = None
    if not issues:
        for fallback_labels in _FALLBACK_LABEL_SETS:
            issues = gh.get_issues(repo, labels=fallback_labels)
            if issues:
                label_warning = f"No 'good first issue' found. Using '{fallback_labels[0]}' label instead."
                break

    if not issues:
        # Last resort: most recent open issues
        try:
            resp = gh.session.get(
                f"{gh.BASE}/repos/{repo}/issues",
                params={'state': 'open', 'per_page': 15, 'sort': 'created', 'direction': 'desc'},
                timeout=gh.TIMEOUT,
            )
            if resp.ok:
                raw_issues = resp.json()
                for item in raw_issues:
                    if 'pull_request' in item:
                        continue
                    body_text = (item.get('body') or '')[:1500]
                    issues.append({
                        'number': item['number'],
                        'title': item.get('title', ''),
                        'url': item.get('html_url', ''),
                        'body': body_text,
                        'labels': [l.get('name', '') for l in item.get('labels', [])],
                        'comments': item.get('comments', 0),
                        'assignee': (item.get('assignee') or {}).get('login'),
                        'created_at': item.get('created_at', ''),
                        'has_linked_pr': False,
                    })
                    if len(issues) >= 15:
                        break
                if issues:
                    label_warning = "No labelled issues found. Using the most recent open issues. Gemma will find problems in the code to fill the remaining slots."
        except Exception:
            pass

    ctx = {
        'repo': repo,
        'meta': meta,
        'branch': branch,
        'readme': readme,
        'contributing': contributing,
        'tree': tree,
        'issues': issues,
        'skills': skills,
        'level': level,
        'file_cache': {},      # path → line-numbered text
        'raw_cache': {},       # path → raw text (for snippet verification)
        'tool_log': [],
        'label_warning': label_warning,
        'gh': gh,
    }

    return ctx


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def validate_candidates(raw: dict, ctx: dict) -> list[dict]:
    """
    Validate and clean the raw candidates from Gemma.
    Returns 3-5 validated candidates. Raises NotEnoughCandidates if < 3.
    """
    candidates_list = raw.get('candidates', [])
    if not isinstance(candidates_list, list):
        raise NotEnoughCandidates("Gemma did not return a candidates list.")

    issue_numbers = {iss['number'] for iss in ctx.get('issues', [])}
    issue_map = {iss['number']: iss for iss in ctx.get('issues', [])}
    tree_set = set(ctx.get('tree', []))
    open_titles = [iss['title'].lower() for iss in ctx.get('issues', [])]

    valid = []
    for c in candidates_list:
        if not isinstance(c, dict):
            continue

        source = c.get('source', '')
        if source not in ('existing', 'suggested'):
            continue

        # Validate 'where' paths
        where = c.get('where', [])
        if isinstance(where, list):
            valid_where = [w for w in where if isinstance(w, dict) and w.get('path', '') in tree_set]
            if not valid_where:
                continue
            c['where'] = valid_where
        else:
            continue

        # Validate existing issues
        if source == 'existing':
            issue_num = c.get('issue_number')
            if issue_num not in issue_numbers:
                continue
            # Overwrite title and url from fetched data
            fetched = issue_map.get(issue_num, {})
            c['title'] = fetched.get('title', c.get('title', ''))
            c['url'] = fetched.get('url', '')
            # Add warning if assigned
            if fetched.get('assignee'):
                c.setdefault('warnings', [])
                c['warnings'] = c.get('warnings', []) + [f"This issue is assigned to @{fetched['assignee']}. It may already be in progress."]

        # Validate suggested issues (check for duplicates)
        if source == 'suggested':
            c['issue_number'] = None
            title = (c.get('title', '') or '').lower()
            is_dup = False
            for open_title in open_titles:
                ratio = difflib.SequenceMatcher(None, title, open_title).ratio()
                if ratio > 0.8:
                    is_dup = True
                    break
            if is_dup:
                continue

        # Ensure required fields
        c.setdefault('title', 'Untitled')
        c.setdefault('summary', '')
        c.setdefault('what_to_change', '')
        c.setdefault('skill_match', [])
        c.setdefault('difficulty', 'medium')
        c.setdefault('estimated_time', '1-2 hours')

        valid.append(c)
        if len(valid) >= 5:
            break

    # Assign sequential IDs
    for i, c in enumerate(valid):
        c['id'] = f'c{i + 1}'

    if len(valid) < 3:
        raise NotEnoughCandidates(
            f"Only {len(valid)} valid candidates found. Try another repo or add more skills."
        )

    return valid


def validate_details(raw: dict, candidate: dict, ctx: dict) -> dict:
    """
    Validate and clean the raw detail object from Gemma.
    """
    tree_set = set(ctx.get('tree', []))
    issue_map = {iss['number']: iss for iss in ctx.get('issues', [])}
    warnings = raw.get('warnings', []) or []
    if not isinstance(warnings, list):
        warnings = []

    # Copy over ID and source from candidate
    raw['id'] = candidate.get('id', '')
    raw['source'] = candidate.get('source', '')
    raw['issue_number'] = candidate.get('issue_number')

    # Validate 'where' entries
    where = raw.get('where', [])
    if isinstance(where, list):
        valid_where = []
        for w in where:
            if not isinstance(w, dict):
                continue
            path = w.get('path', '')
            if path not in tree_set:
                continue
            # Verify snippet if available
            snippet = w.get('snippet', '')
            if snippet and path in ctx.get('raw_cache', {}):
                raw_text = ctx['raw_cache'][path]
                # Normalize whitespace for comparison
                snippet_lines = [l.rstrip() for l in snippet.strip().splitlines()]
                raw_lines = [l.rstrip() for l in raw_text.splitlines()]
                snippet_joined = '\n'.join(snippet_lines)
                raw_joined = '\n'.join(raw_lines)
                if snippet_joined not in raw_joined:
                    # Try a looser check: each snippet line appears in the file
                    matches = sum(1 for sl in snippet_lines if sl in raw_lines)
                    if matches < len(snippet_lines) * 0.5:
                        w['snippet'] = ''
                        warnings.append(f"Snippet for {path} could not be verified and was removed.")
            valid_where.append(w)
        raw['where'] = valid_where
    else:
        raw['where'] = []

    # Validate 'what_to_change' entries
    wtc = raw.get('what_to_change', [])
    if isinstance(wtc, list):
        valid_wtc = []
        for step in wtc:
            if isinstance(step, dict):
                path = step.get('path', '')
                if path in tree_set or not path:
                    valid_wtc.append(step)
            elif isinstance(step, str):
                valid_wtc.append({'path': '', 'change': step})
        raw['what_to_change'] = valid_wtc
    else:
        raw['what_to_change'] = []

    # Handle source-specific logic
    if raw['source'] == 'existing':
        issue_num = raw.get('issue_number')
        fetched = issue_map.get(issue_num, {})
        raw['title'] = fetched.get('title', raw.get('title', ''))
        raw['url'] = fetched.get('url', '')
        raw['issue_draft'] = None
        if fetched.get('assignee'):
            warnings.append(f"This issue is assigned to @{fetched['assignee']}.")
    elif raw['source'] == 'suggested':
        raw['url'] = None
        # Ensure issue_draft exists
        if not raw.get('issue_draft') or not isinstance(raw.get('issue_draft'), dict):
            raw['issue_draft'] = {
                'title': raw.get('title', ''),
                'body_markdown': raw.get('problem', ''),
            }
        # Ensure the body ends with location info
        draft = raw['issue_draft']
        body = draft.get('body_markdown', '')
        if body and '## Location' not in body and '### Location' not in body:
            location_lines = []
            for w in raw.get('where', []):
                path = w.get('path', '')
                lines = w.get('lines', '')
                if path:
                    link = file_link(ctx['repo'], ctx.get('branch', 'main'), path, lines)
                    location_lines.append(f"- [{path}:{lines}]({link})")
            if location_lines:
                body += f"\n\n### Location\n" + '\n'.join(location_lines)
                draft['body_markdown'] = body

    # Ensure all required fields
    raw.setdefault('problem', '')
    raw.setdefault('how_to_verify', [])
    raw.setdefault('setup_steps', [])
    raw.setdefault('difficulty', candidate.get('difficulty', 'medium'))
    raw.setdefault('estimated_time', candidate.get('estimated_time', '1-2 hours'))
    raw.setdefault('skill_match', candidate.get('skill_match', []))
    raw.setdefault('pr', {
        'branch_name': '', 'commit_message': '', 'title': '', 'description_markdown': ''
    })

    raw['warnings'] = warnings
    return raw


# ──────────────────────────────────────────────
# Stage 1: find_candidates
# ──────────────────────────────────────────────

def find_candidates(
    ctx: dict,
    model: str = 'gemma-4-31b-it',
    progress_callback=None,
) -> list[dict]:
    """
    Stage 1: find 3-5 task candidates by calling Gemma with tools.
    Falls back to gemma-4-26b-a4b-it and no-tools mode if needed.
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    client = _get_client()
    gh = ctx.get('gh') or GitHubClient()
    repo = ctx['repo']

    tools = make_tools(gh, repo, ctx, max_reads=6)

    _progress("Gemma is scanning the code…")
    prompt = build_stage1_prompt(ctx)

    raw = None
    fallback_model = 'gemma-4-26b-a4b-it' if model != 'gemma-4-26b-a4b-it' else 'gemma-4-31b-it'
    models_to_try = [model]
    if fallback_model not in models_to_try:
        models_to_try.append(fallback_model)

    last_error = None
    for cur_model in models_to_try:
        try:
            _progress(f"Scanning with {cur_model}…")
            response_text = _call_gemma_with_tools(
                client, cur_model, STAGE1_SYSTEM, prompt, tools
            )
            raw = parse_json(response_text)
            break
        except (ValueError, json.JSONDecodeError):
            _progress("Retrying (formatting response)…")
            try:
                response_text = _call_gemma_with_tools(
                    client, cur_model, STAGE1_SYSTEM,
                    prompt + "\n\n" + RETRY_MESSAGE,
                    tools,
                )
                raw = parse_json(response_text)
                break
            except Exception as e_retry:
                last_error = e_retry
        except Exception as e_tools:
            last_error = e_tools
            # Try no-tools mode for this model as per PRD Section 10 fallback
            _progress(f"Trying direct analysis with {cur_model}…")
            try:
                response_text = _call_gemma_no_tools(
                    client, cur_model, STAGE1_SYSTEM, prompt
                )
                raw = parse_json(response_text)
                break
            except Exception as e_notools:
                last_error = e_notools

    if raw is None:
        raise GemmaError(f"Gemma scan failed after retries: {last_error}")

    _progress("Ranking matches…")
    candidates = validate_candidates(raw, ctx)
    return candidates


# ──────────────────────────────────────────────
# Stage 2: get_details
# ──────────────────────────────────────────────

def get_details(
    ctx: dict,
    candidate: dict,
    model: str = 'gemma-4-31b-it',
    progress_callback=None,
) -> dict:
    """
    Stage 2: get full details for one candidate.
    Results are meant to be cached by the caller in session_state.
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    client = _get_client()
    gh = ctx.get('gh') or GitHubClient()
    repo = ctx['repo']

    tools = make_tools(gh, repo, ctx, max_reads=4)

    _progress("Gemma is reading the relevant files…")
    prompt = build_stage2_prompt(ctx, candidate)

    raw = None
    fallback_model = 'gemma-4-26b-a4b-it' if model != 'gemma-4-26b-a4b-it' else 'gemma-4-31b-it'
    models_to_try = [model]
    if fallback_model not in models_to_try:
        models_to_try.append(fallback_model)

    last_error = None
    for cur_model in models_to_try:
        try:
            _progress(f"Reading files with {cur_model}…")
            response_text = _call_gemma_with_tools(
                client, cur_model, STAGE2_SYSTEM, prompt, tools
            )
            raw = parse_json(response_text)
            break
        except (ValueError, json.JSONDecodeError):
            _progress("Retrying (formatting response)…")
            try:
                response_text = _call_gemma_with_tools(
                    client, cur_model, STAGE2_SYSTEM,
                    prompt + "\n\n" + RETRY_MESSAGE,
                    tools,
                )
                raw = parse_json(response_text)
                break
            except Exception as e_retry:
                last_error = e_retry
        except Exception as e_tools:
            last_error = e_tools
            if "429" in str(e_tools) or "RESOURCE_EXHAUSTED" in str(e_tools):
                _progress("⏳ Free-tier quota pause: waiting 10s for window reset…")
                import time
                time.sleep(10)

            # Section 10 Fallback: pre-read files from candidate where and call no-tools
            _progress("Falling back to direct file inclusion…")
            try:
                for w in candidate.get('where', []):
                    path = w.get('path', '')
                    if path and path not in ctx['file_cache']:
                        text = gh.get_file(repo, path)[:8000]
                        if text:
                            ctx['raw_cache'][path] = text
                            ctx['file_cache'][path] = '\n'.join(
                                f'L{i}: {line}' for i, line in enumerate(text.splitlines(), 1)
                            )

                files_text = ""
                for path, content in ctx['file_cache'].items():
                    files_text += f"\n=== FILE: {path} ===\n{content}\n"

                response_text = _call_gemma_no_tools(
                    client, cur_model, STAGE2_SYSTEM, prompt + files_text
                )
                raw = parse_json(response_text)
                break
            except Exception as e_notools:
                last_error = e_notools

    if raw is None:
        raise GemmaError(f"Gemma detail call failed after retries: {last_error}")

    detail = validate_details(raw, candidate, ctx)
    return detail
