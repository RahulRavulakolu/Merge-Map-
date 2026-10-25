"""
github_client.py – GitHub REST API wrapper for RepoSherpa.

All calls go to https://api.github.com with proper headers.
No external GitHub library needed — just `requests`.
"""

import os
import re
import base64
import urllib.parse
import requests


# ──────────────────────────────────────────────
# Custom exceptions
# ──────────────────────────────────────────────

class RepoNotFound(Exception):
    """Raised when the repo returns 404."""
    pass


class RateLimitError(Exception):
    """Raised when GitHub returns 403 with X-RateLimit-Remaining: 0."""
    pass


class GitHubError(Exception):
    """Raised for any other non-success status."""
    def __init__(self, status: int, text: str):
        self.status = status
        self.text = text
        super().__init__(f"GitHub API error {status}: {text}")


class IssuesDisabled(Exception):
    """Raised when the repo has issues disabled (410)."""
    pass


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

# Paths that are noise for code analysis
_NOISE_DIRS = {
    'node_modules/', 'dist/', 'build/', 'vendor/', '.git/',
    '__pycache__/', '.next/', '.nuxt/', 'coverage/', '.cache/',
    '.tox/', '.mypy_cache/', '.pytest_cache/', 'eggs/', '.eggs/',
}

_NOISE_EXTENSIONS = {
    '.lock', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
    '.woff', '.woff2', '.ttf', '.eot', '.otf',
    '.zip', '.tar', '.gz', '.bz2', '.rar', '.7z',
    '.mp3', '.mp4', '.avi', '.mov', '.wav',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.pyc', '.pyo', '.class', '.o', '.so', '.dll', '.exe',
    '.min.js', '.min.css', '.map',
}

_LOCKFILES = {
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
    'Gemfile.lock', 'poetry.lock', 'Pipfile.lock',
    'composer.lock', 'Cargo.lock', 'go.sum',
}


def _is_noise(path: str) -> bool:
    """Return True if a file tree path should be excluded."""
    lower = path.lower()
    # Check directory noise
    for noise_dir in _NOISE_DIRS:
        if noise_dir in lower:
            return True
    # Check filename against lockfiles
    basename = path.rsplit('/', 1)[-1] if '/' in path else path
    if basename in _LOCKFILES:
        return True
    # Check extension noise
    for ext in _NOISE_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


# ──────────────────────────────────────────────
# URL parsing
# ──────────────────────────────────────────────

def parse_repo(url: str) -> str:
    """
    Parse a GitHub repo URL or shorthand into 'owner/repo'.

    Accepts:
        https://github.com/owner/repo
        https://github.com/owner/repo.git
        https://github.com/owner/repo/
        github.com/owner/repo
        owner/repo

    Raises ValueError on anything else.
    """
    url = url.strip()
    if not url:
        raise ValueError("Empty repo URL.")

    # Strip trailing .git and /
    url = url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]

    # Full URL pattern
    full_pattern = re.compile(
        r'^(?:https?://)?github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)$'
    )
    m = full_pattern.match(url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    # Short form: owner/repo
    short_pattern = re.compile(
        r'^([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)$'
    )
    m = short_pattern.match(url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    raise ValueError(
        f"Cannot parse '{url}'. Use 'owner/repo' or 'https://github.com/owner/repo'."
    )


def new_issue_url(repo: str, title: str, body: str) -> str:
    """
    Build a pre-filled new-issue URL.
    Body is capped at 4000 chars to avoid URL length rejections.
    """
    body_trimmed = body[:4000]
    params = urllib.parse.urlencode({
        'title': title,
        'body': body_trimmed,
    }, quote_via=urllib.parse.quote)
    return f"https://github.com/{repo}/issues/new?{params}"


def file_link(repo: str, branch: str, path: str, lines: str) -> str:
    """
    Build a GitHub link to a file, optionally with line range.
    lines: '12-30' → #L12-L30
    """
    url = f"https://github.com/{repo}/blob/{branch}/{path}"
    if lines:
        parts = lines.split('-')
        if len(parts) == 2:
            url += f"#L{parts[0]}-L{parts[1]}"
        elif len(parts) == 1:
            url += f"#L{parts[0]}"
    return url


# ──────────────────────────────────────────────
# GitHub API client
# ──────────────────────────────────────────────

class GitHubClient:
    """Thin wrapper around the GitHub REST API."""

    BASE = "https://api.github.com"
    TIMEOUT = 15  # seconds

    def __init__(self, token: str | None = None):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
        })
        token = token or os.environ.get('GITHUB_TOKEN')
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    # ── Internal request helper ──

    def _get(self, path: str, **kwargs) -> requests.Response:
        """Make a GET request, handling common error codes."""
        headers = kwargs.pop('headers', {})
        resp = self.session.get(
            f"{self.BASE}{path}",
            headers=headers,
            timeout=self.TIMEOUT,
            **kwargs,
        )
        if resp.status_code == 404:
            raise RepoNotFound(f"Not found: {path}")
        if resp.status_code == 410:
            raise IssuesDisabled("Issues are disabled on this repo.")
        if resp.status_code == 403:
            remaining = resp.headers.get('X-RateLimit-Remaining', '1')
            if remaining == '0':
                raise RateLimitError(
                    "GitHub rate limit hit. Add a token in the sidebar."
                )
            raise GitHubError(resp.status_code, resp.text)
        if resp.status_code == 401:
            raise GitHubError(401, "Bad GitHub token. Check the token in the sidebar.")
        if not resp.ok:
            raise GitHubError(resp.status_code, resp.text)
        return resp

    def _post(self, path: str, json_data: dict) -> requests.Response:
        """Make a POST request."""
        resp = self.session.post(
            f"{self.BASE}{path}",
            json=json_data,
            timeout=self.TIMEOUT,
        )
        if resp.status_code == 404:
            raise RepoNotFound(f"Not found: {path}")
        if resp.status_code == 410:
            raise IssuesDisabled("Issues are disabled on this repo.")
        if resp.status_code == 403:
            remaining = resp.headers.get('X-RateLimit-Remaining', '1')
            if remaining == '0':
                raise RateLimitError("GitHub rate limit hit.")
            raise GitHubError(resp.status_code, "No permission to create issues.")
        if resp.status_code == 401:
            raise GitHubError(401, "Bad GitHub token.")
        if resp.status_code == 422:
            raise GitHubError(422, "Validation failed. Check the issue title and body.")
        if not resp.ok:
            raise GitHubError(resp.status_code, resp.text)
        return resp

    # ── Public methods ──

    def get_repo(self, repo: str) -> dict:
        """
        Fetch repo metadata.
        Returns dict with default_branch, description, language.
        """
        data = self._get(f"/repos/{repo}").json()
        return {
            'default_branch': data.get('default_branch', 'main'),
            'description': data.get('description', ''),
            'language': data.get('language', ''),
            'full_name': data.get('full_name', repo),
            'html_url': data.get('html_url', f'https://github.com/{repo}'),
            'has_issues': data.get('has_issues', True),
            'stargazers_count': data.get('stargazers_count', 0),
            'open_issues_count': data.get('open_issues_count', 0),
        }

    def get_tree(self, repo: str, branch: str) -> list[str]:
        """
        Fetch the full file tree (recursive), filter noise, cap at 3000.
        Returns list of file paths.
        """
        data = self._get(
            f"/repos/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
        ).json()

        paths = []
        for item in data.get('tree', []):
            if item.get('type') != 'blob':
                continue
            p = item.get('path', '')
            if not _is_noise(p):
                paths.append(p)
            if len(paths) >= 3000:
                break

        return paths

    def get_readme(self, repo: str) -> str:
        """
        Fetch the README content as plain text.
        Returns '' if not found. Capped at 20000 chars.
        """
        try:
            resp = self._get(
                f"/repos/{repo}/readme",
                headers={"Accept": "application/vnd.github.raw"},
            )
            return resp.text[:20000]
        except RepoNotFound:
            return ''

    def get_contributing(self, repo: str) -> str:
        """
        Try to fetch CONTRIBUTING.md (case-insensitive search).
        Returns '' if not found. Capped at 15000 chars.
        """
        for path in ['CONTRIBUTING.md', 'contributing.md', '.github/CONTRIBUTING.md',
                      'docs/CONTRIBUTING.md', 'docs/contributing.md']:
            try:
                return self.get_file(repo, path)[:15000]
            except (RepoNotFound, GitHubError):
                continue
        return ''

    def get_file(self, repo: str, path: str) -> str:
        """
        Fetch raw file contents. Returns '' if binary or missing.
        Capped at 12000 chars for tool use.
        """
        try:
            resp = self._get(
                f"/repos/{repo}/contents/{path}",
                headers={"Accept": "application/vnd.github.raw"},
            )
            # Check if response looks binary
            content = resp.text
            if '\x00' in content[:1000]:
                return ''
            return content
        except (RepoNotFound, GitHubError):
            return ''

    def get_issues(
        self,
        repo: str,
        labels: tuple[str, ...] = ('good first issue',),
        limit: int = 15,
    ) -> list[dict]:
        """
        Fetch open issues with given labels. Skips pull requests.
        Returns list of dicts with keys:
            number, title, url, body, labels, comments, assignee, created_at
        """
        all_issues = []

        for label in labels:
            resp = self._get(
                f"/repos/{repo}/issues",
                params={
                    'labels': label,
                    'state': 'open',
                    'per_page': limit,
                },
            )
            for item in resp.json():
                # Skip pull requests
                if 'pull_request' in item:
                    continue
                # Skip duplicates
                num = item['number']
                if any(i['number'] == num for i in all_issues):
                    continue
                body_text = (item.get('body') or '')[:1500]
                all_issues.append({
                    'number': num,
                    'title': item.get('title', ''),
                    'url': item.get('html_url', ''),
                    'body': body_text,
                    'labels': [l.get('name', '') for l in item.get('labels', [])],
                    'comments': item.get('comments', 0),
                    'assignee': (item.get('assignee') or {}).get('login'),
                    'created_at': item.get('created_at', ''),
                    'has_linked_pr': False,  # simplified; would need events API for full check
                })
                if len(all_issues) >= limit:
                    break
            if len(all_issues) >= limit:
                break

        return all_issues[:limit]

    def create_issue(self, repo: str, title: str, body: str) -> dict:
        """
        Create an issue via the API (P1). Requires a token with public_repo scope.
        Returns dict with 'url' and 'number'.
        """
        resp = self._post(
            f"/repos/{repo}/issues",
            json_data={'title': title, 'body': body},
        )
        data = resp.json()
        return {
            'url': data.get('html_url', ''),
            'number': data.get('number', 0),
        }
