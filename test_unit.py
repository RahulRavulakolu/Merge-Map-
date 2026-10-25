"""
test_unit.py – Comprehensive unit tests for RepoSherpa components without external API dependencies.
"""

import unittest
from github_client import parse_repo, new_issue_url, file_link, _is_noise
from agent import parse_json, validate_candidates, validate_details, NotEnoughCandidates


class TestGitHubClientHelpers(unittest.TestCase):
    def test_parse_repo_valid(self):
        cases = [
            ("https://github.com/owner/repo", "owner/repo"),
            ("https://github.com/owner/repo.git", "owner/repo"),
            ("https://github.com/owner/repo/", "owner/repo"),
            ("http://github.com/owner/repo", "owner/repo"),
            ("github.com/owner/repo", "owner/repo"),
            ("owner/repo", "owner/repo"),
            ("psf/requests", "psf/requests"),
            ("facebook/react", "facebook/react"),
        ]
        for url, expected in cases:
            with self.subTest(url=url):
                self.assertEqual(parse_repo(url), expected)

    def test_parse_repo_invalid(self):
        bad_cases = [
            "",
            "   ",
            "not-a-repo",
            "https://gitlab.com/owner/repo",
            "owner/repo/extra/path",
            "https://github.com/owner",
        ]
        for bad in bad_cases:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_repo(bad)

    def test_new_issue_url(self):
        url = new_issue_url("owner/repo", "Test Title", "Line 1\nLine 2")
        self.assertTrue(url.startswith("https://github.com/owner/repo/issues/new?"))
        self.assertIn("Test%20Title", url)
        self.assertIn("Line%201", url)

        # Capping at 4000 chars
        long_body = "x" * 5000
        url_long = new_issue_url("owner/repo", "Title", long_body)
        self.assertNotIn("x" * 4500, url_long)

    def test_file_link(self):
        link1 = file_link("owner/repo", "main", "src/index.js", "12-30")
        self.assertEqual(link1, "https://github.com/owner/repo/blob/main/src/index.js#L12-L30")

        link2 = file_link("owner/repo", "master", "README.md", "42")
        self.assertEqual(link2, "https://github.com/owner/repo/blob/master/README.md#L42")

        link3 = file_link("owner/repo", "main", "file.py", "")
        self.assertEqual(link3, "https://github.com/owner/repo/blob/main/file.py")

    def test_is_noise(self):
        self.assertTrue(_is_noise("node_modules/express/index.js"))
        self.assertTrue(_is_noise("dist/bundle.js"))
        self.assertTrue(_is_noise("build/output.css"))
        self.assertTrue(_is_noise(".git/config"))
        self.assertTrue(_is_noise("vendor/autoload.php"))
        self.assertTrue(_is_noise("package-lock.json"))
        self.assertTrue(_is_noise("assets/logo.png"))
        self.assertTrue(_is_noise("fonts/inter.woff2"))
        self.assertTrue(_is_noise("script.min.js"))

        self.assertFalse(_is_noise("src/index.js"))
        self.assertFalse(_is_noise("README.md"))
        self.assertFalse(_is_noise("app/main.py"))
        self.assertFalse(_is_noise("tests/test_api.py"))


class TestAgentParsingAndValidation(unittest.TestCase):
    def test_parse_json_direct(self):
        raw = '{"candidates": [{"id": "c1", "title": "Fix bug"}]}'
        data = parse_json(raw)
        self.assertEqual(data["candidates"][0]["title"], "Fix bug")

    def test_parse_json_with_code_fences(self):
        raw = '```json\n{"candidates": [{"id": "c1", "title": "Fix bug"}]}\n```'
        data = parse_json(raw)
        self.assertEqual(data["candidates"][0]["title"], "Fix bug")

    def test_parse_json_with_surrounding_text(self):
        raw = 'Here is the response:\n{"key": "value"}\nHope this helps!'
        data = parse_json(raw)
        self.assertEqual(data["key"], "value")

    def test_validate_candidates(self):
        ctx = {
            "repo": "owner/repo",
            "tree": ["src/app.py", "src/utils.py", "docs/index.md"],
            "issues": [
                {"number": 101, "title": "Docs need update", "url": "https://github.com/owner/repo/issues/101", "assignee": None},
                {"number": 102, "title": "Add tests", "url": "https://github.com/owner/repo/issues/102", "assignee": "alice"},
            ],
        }

        raw = {
            "candidates": [
                {
                    "source": "existing",
                    "issue_number": 101,
                    "title": "Old title",
                    "where": [{"path": "docs/index.md", "lines": "1-10"}],
                    "summary": "Fix docs",
                    "what_to_change": "Update docs",
                },
                {
                    "source": "existing",
                    "issue_number": 102,
                    "title": "Add tests",
                    "where": [{"path": "src/utils.py", "lines": "20-40"}],
                    "summary": "Need unit tests",
                    "what_to_change": "Write tests",
                },
                {
                    "source": "suggested",
                    "title": "Handle ValueError in app.py",
                    "where": [{"path": "src/app.py", "lines": "15-25"}],
                    "summary": "Missing error handling",
                    "what_to_change": "Add try/except",
                },
                {
                    # Invalid path should be dropped
                    "source": "suggested",
                    "title": "Fake path issue",
                    "where": [{"path": "nonexistent/file.py", "lines": "1-10"}],
                },
                {
                    # Duplicate of existing issue should be dropped
                    "source": "suggested",
                    "title": "Docs need update",  # Near duplicate of #101
                    "where": [{"path": "docs/index.md", "lines": "5-10"}],
                },
            ]
        }

        valid = validate_candidates(raw, ctx)
        self.assertEqual(len(valid), 3)
        self.assertEqual(valid[0]["id"], "c1")
        self.assertEqual(valid[0]["title"], "Docs need update")  # Overwritten from fetched
        self.assertEqual(valid[1]["id"], "c2")
        self.assertIn("warnings", valid[1])  # Assigned warning
        self.assertEqual(valid[2]["id"], "c3")
        self.assertEqual(valid[2]["source"], "suggested")
        self.assertIsNone(valid[2]["issue_number"])

    def test_validate_candidates_not_enough(self):
        ctx = {"tree": ["a.py"], "issues": []}
        raw = {"candidates": [{"source": "suggested", "where": [{"path": "a.py"}]}]}
        with self.assertRaises(NotEnoughCandidates):
            validate_candidates(raw, ctx)

    def test_validate_details(self):
        ctx = {
            "repo": "owner/repo",
            "branch": "main",
            "tree": ["src/app.py"],
            "raw_cache": {
                "src/app.py": "def add(a, b):\n    return a + b\n"
            },
            "issues": [
                {"number": 101, "title": "Real title", "url": "https://github.com/owner/repo/issues/101", "assignee": None}
            ]
        }

        candidate = {
            "id": "c1",
            "source": "suggested",
            "title": "Add sub function",
            "difficulty": "easy",
            "estimated_time": "30 mins",
            "skill_match": ["Python"],
        }

        raw_detail = {
            "problem": "Sub function is missing",
            "where": [
                {
                    "path": "src/app.py",
                    "lines": "1-2",
                    "snippet": "def add(a, b):\n    return a + b",
                }
            ],
            "what_to_change": [
                {"path": "src/app.py", "change": "Add def sub(a, b): return a - b"}
            ],
            "how_to_verify": ["Run python -m unittest"],
            "setup_steps": ["python -m venv .venv"],
            "issue_draft": {
                "title": "Add sub function",
                "body_markdown": "We need a subtraction function.",
            },
            "pr": {
                "branch_name": "feat/add-sub",
                "commit_message": "feat: add subtraction helper",
                "title": "Add subtraction helper",
                "description_markdown": "This PR adds sub(a, b).",
            }
        }

        detail = validate_details(raw_detail, candidate, ctx)
        self.assertEqual(detail["id"], "c1")
        self.assertEqual(detail["source"], "suggested")
        self.assertIsNone(detail["url"])
        self.assertEqual(detail["where"][0]["snippet"], "def add(a, b):\n    return a + b")
        # Location section should be appended to issue_draft
        self.assertIn("### Location", detail["issue_draft"]["body_markdown"])
        self.assertIn("https://github.com/owner/repo/blob/main/src/app.py#L1-L2", detail["issue_draft"]["body_markdown"])


if __name__ == "__main__":
    unittest.main()
