"""
test_mock_pipeline.py – Mocked end-to-end pipeline test for RepoSherpa.
Verifies Stage 1 (find_candidates) and Stage 2 (get_details) with mocked GitHub and Gemma responses.
"""

import unittest
from unittest.mock import MagicMock, patch
import json

from github_client import GitHubClient, RepoNotFound, RateLimitError
from agent import build_context, find_candidates, get_details, NotEnoughCandidates, GemmaError


class TestPipelineMocked(unittest.TestCase):
    def setUp(self):
        self.mock_gh = MagicMock(spec=GitHubClient)
        self.mock_gh.get_repo.return_value = {
            'default_branch': 'main',
            'description': 'A sample repository for testing',
            'language': 'Python',
            'full_name': 'test-owner/test-repo',
            'html_url': 'https://github.com/test-owner/test-repo',
            'has_issues': True,
            'stargazers_count': 150,
            'open_issues_count': 5,
        }
        self.mock_gh.get_tree.return_value = [
            'README.md',
            'CONTRIBUTING.md',
            'app/__init__.py',
            'app/main.py',
            'app/utils.py',
            'tests/test_main.py',
        ]
        self.mock_gh.get_readme.return_value = "# Test Repo\nThis is a test repo."
        self.mock_gh.get_contributing.return_value = "# Contributing\nFollow PEP8."
        self.mock_gh.get_issues.return_value = [
            {
                'number': 42,
                'title': 'Add docstrings to app/utils.py',
                'url': 'https://github.com/test-owner/test-repo/issues/42',
                'body': 'Many helper functions in app/utils.py lack docstrings.',
                'labels': ['good first issue'],
                'comments': 1,
                'assignee': None,
                'created_at': '2026-10-01T00:00:00Z',
                'has_linked_pr': False,
            },
            {
                'number': 43,
                'title': 'Improve test coverage for utils',
                'url': 'https://github.com/test-owner/test-repo/issues/43',
                'body': 'test_main.py only tests main.py, need utils coverage.',
                'labels': ['good first issue'],
                'comments': 0,
                'assignee': 'someone',
                'created_at': '2026-10-02T00:00:00Z',
                'has_linked_pr': False,
            }
        ]
        self.mock_gh.get_file.side_effect = lambda repo, path: (
            "def calculate(x, y):\n    # TODO: add input validation\n    return x + y\n"
            if path == 'app/utils.py'
            else "def test_ok():\n    assert True\n"
        )

    def test_build_context_success(self):
        with patch('agent.GitHubClient', return_value=self.mock_gh):
            ctx = build_context('test-owner/test-repo', 'Python, Pytest', 'Beginner')
            self.assertEqual(ctx['repo'], 'test-owner/test-repo')
            self.assertEqual(ctx['branch'], 'main')
            self.assertEqual(len(ctx['tree']), 6)
            self.assertEqual(len(ctx['issues']), 2)
            self.assertIn('app/utils.py', ctx['tree'])

    def test_find_candidates_success(self):
        ctx = {
            'repo': 'test-owner/test-repo',
            'branch': 'main',
            'tree': ['README.md', 'app/main.py', 'app/utils.py', 'tests/test_main.py'],
            'issues': [
                {
                    'number': 42,
                    'title': 'Add docstrings to app/utils.py',
                    'url': 'https://github.com/test-owner/test-repo/issues/42',
                    'assignee': None,
                }
            ],
            'file_cache': {},
            'raw_cache': {},
            'tool_log': [],
            'gh': self.mock_gh,
        }

        mock_gemma_json = json.dumps({
            'candidates': [
                {
                    'source': 'existing',
                    'issue_number': 42,
                    'title': 'Add docstrings to app/utils.py',
                    'summary': 'Add docstrings to exported functions in utils.',
                    'where': [{'path': 'app/utils.py', 'lines': '1-10'}],
                    'what_to_change': 'Add google-style docstrings.',
                    'skill_match': ['Python'],
                    'difficulty': 'easy',
                    'estimated_time': '30 mins',
                },
                {
                    'source': 'suggested',
                    'title': 'Add input validation for calculate function',
                    'summary': 'calculate() lacks validation for non-numeric types.',
                    'where': [{'path': 'app/utils.py', 'lines': '1-4'}],
                    'what_to_change': 'Raise TypeError if x or y is not int/float.',
                    'skill_match': ['Python'],
                    'difficulty': 'easy',
                    'estimated_time': '45 mins',
                },
                {
                    'source': 'suggested',
                    'title': 'Add unit tests for calculate()',
                    'summary': 'calculate() has no tests in tests/test_main.py.',
                    'where': [{'path': 'tests/test_main.py', 'lines': '1-5'}],
                    'what_to_change': 'Add test_calculate_positive and test_calculate_invalid.',
                    'skill_match': ['Python', 'Pytest'],
                    'difficulty': 'easy',
                    'estimated_time': '30 mins',
                }
            ]
        })

        with patch('agent._get_client') as mock_client_factory:
            mock_client = MagicMock()
            mock_client_factory.return_value = mock_client
            with patch('agent._call_gemma_with_tools', return_value=mock_gemma_json):
                candidates = find_candidates(ctx, model='gemma-4-31b-it')
                self.assertEqual(len(candidates), 3)
                self.assertEqual(candidates[0]['id'], 'c1')
                self.assertEqual(candidates[0]['source'], 'existing')
                self.assertEqual(candidates[1]['id'], 'c2')
                self.assertEqual(candidates[1]['source'], 'suggested')
                self.assertEqual(candidates[2]['id'], 'c3')

    def test_get_details_success(self):
        ctx = {
            'repo': 'test-owner/test-repo',
            'branch': 'main',
            'tree': ['app/utils.py'],
            'issues': [],
            'file_cache': {},
            'raw_cache': {
                'app/utils.py': "def calculate(x, y):\n    # TODO: add input validation\n    return x + y\n"
            },
            'tool_log': [],
            'gh': self.mock_gh,
        }

        candidate = {
            'id': 'c2',
            'source': 'suggested',
            'issue_number': None,
            'title': 'Add input validation for calculate function',
            'difficulty': 'easy',
            'estimated_time': '45 mins',
            'skill_match': ['Python'],
        }

        mock_detail_json = json.dumps({
            'problem': 'calculate() does not check argument types and can raise unexpected exceptions.',
            'where': [
                {
                    'path': 'app/utils.py',
                    'lines': '1-3',
                    'snippet': "def calculate(x, y):\n    # TODO: add input validation\n    return x + y",
                    'explanation': 'The TODO marks missing validation.',
                }
            ],
            'what_to_change': [
                {'path': 'app/utils.py', 'change': 'Check isinstance(x, (int, float)) and isinstance(y, (int, float))'}
            ],
            'how_to_verify': ['Run pytest tests/'],
            'setup_steps': ['pip install -r requirements.txt'],
            'issue_draft': {
                'title': 'Add input validation for calculate()',
                'body_markdown': '### Problem\ncalculate() lacks type validation.\n\n### Fix\nAdd isinstance checks.\n\nI would like to work on this issue.',
            },
            'pr': {
                'branch_name': 'fix/calculate-validation',
                'commit_message': 'fix: add type validation to calculate helper',
                'title': 'Add type validation to calculate helper',
                'description_markdown': 'Closes issue regarding calculate() type checks.',
            }
        })

        with patch('agent._get_client') as mock_client_factory:
            mock_client = MagicMock()
            mock_client_factory.return_value = mock_client
            with patch('agent._call_gemma_with_tools', return_value=mock_detail_json):
                detail = get_details(ctx, candidate, model='gemma-4-31b-it')
                self.assertEqual(detail['id'], 'c2')
                self.assertEqual(detail['source'], 'suggested')
                self.assertEqual(detail['where'][0]['snippet'], "def calculate(x, y):\n    # TODO: add input validation\n    return x + y")
                self.assertIn("### Location", detail['issue_draft']['body_markdown'])
                self.assertIn("calculate-validation", detail['pr']['branch_name'])


if __name__ == '__main__':
    unittest.main()
