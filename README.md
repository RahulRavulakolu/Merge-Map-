# 🧗 RepoSherpa

**Your AI guide to finding the perfect first open-source contribution.**

RepoSherpa is a Streamlit web app that helps first-time contributors find tasks that match their skills in any public GitHub repository. Powered by **Gemma 4** via the Gemini API, it reads the codebase, matches existing issues to your skills, and discovers new tasks hidden in the code — then gives you exactly where to look, what to change, and a drafted PR.

## ✨ Features

- **Skill-Matched Tasks** — Enter your skills and experience level; Gemma finds tasks that fit you
- **AI-Discovered Issues** — Gemma reads the code and spots TODOs, missing docs, untested functions, and small bugs
- **Full Context** — See exactly where to change (file, line range, code snippet), step-by-step instructions, and verification steps
- **PR Draft** — Branch name, commit message, title, and description ready to use
- **One-Click Issue Creation** — Open a pre-filled issue on GitHub or create directly via API
- **Works on Any Repo** — No special labels needed; works even on repos with no "good first issue" tags

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/reposherpa.git
cd reposherpa
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
copy .env.example .env
# Edit .env and add your keys:
#   GEMINI_API_KEY=your_gemini_api_key
#   GITHUB_TOKEN=your_github_token (optional but recommended)
```

**Get your keys:**
- **Gemini API Key:** [Google AI Studio](https://aistudio.google.com/apikey)
- **GitHub Token:** [GitHub Settings → Tokens](https://github.com/settings/tokens) (no scopes needed for public repos)

### 3. Smoke test the backend

```bash
python test_pipeline.py owner/repo
```

### 4. Run the app

```bash
streamlit run app.py
```

## 🏗️ Architecture

```
Streamlit UI (app.py)
    ↓
github_client.py  — GitHub REST API calls
    ↓
agent.py          — Gemma stage 1 (find candidates) + stage 2 (details on demand)
    ↓
prompts.py        — System prompts and schemas
```

### Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI with input form, candidate cards, and detail view |
| `github_client.py` | GitHub REST API wrapper (repo info, file tree, issues, file contents) |
| `agent.py` | Gemma 4 integration with tool calling, validation, and fallbacks |
| `prompts.py` | System prompts and JSON schemas for both stages |
| `test_pipeline.py` | Backend smoke test script |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |

## 🔧 How It Works

### Stage 1: Find Candidates
1. Fetches repo data (file tree, README, CONTRIBUTING, open issues)
2. Sends everything to Gemma 4 with `read_file` and `find_files` tools
3. Gemma returns 5 candidates: a mix of existing issues and new problems found in the code
4. Each candidate is validated (paths verified, duplicates removed, issues cross-referenced)

### Stage 2: Get Details (on demand)
1. User clicks **View details** on a candidate card
2. Gemma reads the relevant files (cached from stage 1 when possible)
3. Returns: problem description, exact location with code snippet, step-by-step changes, verification steps, and PR draft
4. Results are cached — clicking the same card again is instant

### Error Handling
- **No "good first issue" labels?** Falls back to other labels, then recent issues
- **Rate limit?** Friendly message prompting for a token
- **Invalid JSON from Gemma?** Automatic retry with hint
- **Gemma timeout?** Retries with fallback model
- **Hallucinated paths?** Validated against real file tree

## 📋 Requirements

- Python 3.10+
- Gemini API key (for Gemma 4 access)
- GitHub token (optional, increases rate limit from 60 to 5000 req/hr)

## 📄 License

MIT
