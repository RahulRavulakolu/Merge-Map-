"""
app.py – RepoSherpa Streamlit UI.

A Streamlit web app that helps first-time open-source contributors find tasks
that match their skills in any public GitHub repository.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from github_client import (
    parse_repo, new_issue_url, file_link,
    GitHubClient, RepoNotFound, RateLimitError, GitHubError, IssuesDisabled,
)
from agent import (
    build_context, find_candidates, get_details,
    NotEnoughCandidates, GemmaError,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title='RepoSherpa',
    page_icon='🧗',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
    margin-bottom: 1rem;
}
.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.hero-header .tagline {
    font-size: 1.1rem;
    color: #8b8fa3;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Candidate cards ── */
.candidate-card {
    background: linear-gradient(145deg, rgba(30,32,48,0.95), rgba(24,26,40,0.98));
    border: 1px solid rgba(102,126,234,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.candidate-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    opacity: 0.7;
}
.candidate-card:hover {
    border-color: rgba(102,126,234,0.4);
    box-shadow: 0 8px 32px rgba(102,126,234,0.12);
    transform: translateY(-2px);
}
.candidate-card .card-title, .candidate-card-inner .card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e8e9f0;
    margin-bottom: 0.5rem;
    line-height: 1.3;
}
.candidate-card .card-summary, .candidate-card-inner .card-summary {
    color: #a0a3b8;
    font-size: 0.92rem;
    line-height: 1.55;
    margin-bottom: 0.75rem;
}
.candidate-card .card-change, .candidate-card-inner .card-change {
    color: #c4c7db;
    font-size: 0.88rem;
    font-style: italic;
    margin-bottom: 0.75rem;
    padding-left: 0.75rem;
    border-left: 2px solid rgba(102,126,234,0.4);
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.badge-existing {
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3);
}
.badge-suggested {
    background: rgba(168,85,247,0.15);
    color: #c084fc;
    border: 1px solid rgba(168,85,247,0.3);
}
.badge-easy {
    background: rgba(34,211,238,0.12);
    color: #67e8f9;
    border: 1px solid rgba(34,211,238,0.25);
}
.badge-medium {
    background: rgba(251,191,36,0.12);
    color: #fcd34d;
    border: 1px solid rgba(251,191,36,0.25);
}
.badge-skill {
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
}
.badge-time {
    background: rgba(244,114,182,0.12);
    color: #f9a8d4;
    border: 1px solid rgba(244,114,182,0.25);
}

/* ── Where tags ── */
.where-tag {
    display: inline-block;
    background: rgba(55,65,81,0.6);
    color: #d1d5db;
    padding: 0.15rem 0.55rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    margin-right: 0.35rem;
    margin-bottom: 0.3rem;
    border: 1px solid rgba(75,85,99,0.4);
}

/* ── Detail view ── */
.detail-header {
    padding: 1.2rem 0;
    margin-bottom: 0.5rem;
}
.detail-header h2 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8e9f0;
    margin-bottom: 0.5rem;
}
.detail-section {
    margin-bottom: 1.5rem;
}
.detail-section h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c7c9de;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    margin-right: 0.5rem;
    flex-shrink: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(20,22,36,0.98), rgba(16,18,30,0.99));
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a5b4fc;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 1.5rem;
}

/* ── Input form styling ── */
.input-section {
    max-width: 720px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* ── Glassmorphism container ── */
.glass-container {
    background: rgba(30,32,48,0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(102,126,234,0.12);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.5s ease-out;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #6b7280;
}
.empty-state .emoji {
    font-size: 3rem;
    margin-bottom: 1rem;
}
.empty-state p {
    font-size: 1rem;
    max-width: 400px;
    margin: 0 auto;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────

if 'ctx' not in st.session_state:
    st.session_state.ctx = None
if 'candidates' not in st.session_state:
    st.session_state.candidates = None
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None
if 'details' not in st.session_state:
    st.session_state.details = {}
if 'edited_titles' not in st.session_state:
    st.session_state.edited_titles = {}
if 'edited_bodies' not in st.session_state:
    st.session_state.edited_bodies = {}


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    gemini_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get('GEMINI_API_KEY', ''),
        type='password',
        help="Required. Used to access Gemma 4 via the Gemini API.",
    )
    if gemini_key:
        os.environ['GEMINI_API_KEY'] = gemini_key.strip()

    model = st.selectbox(
        "Gemma model",
        options=['gemma-4-26b-a4b-it', 'gemma-4-31b-it'],
        index=0,
        help="gemma-4-26b-a4b-it is fast and reliable. gemma-4-31b-it is also available.",
    )

    token = st.text_input(
        "GitHub token (read)",
        value=os.environ.get('GITHUB_TOKEN', ''),
        type='password',
        help="A personal access token with no scopes. Increases rate limit from 60 to 5000 requests/hour.",
    )
    if token:
        os.environ['GITHUB_TOKEN'] = token.strip()

    write_token = st.text_input(
        "GitHub token (write, optional)",
        value=os.environ.get('GITHUB_WRITE_TOKEN', ''),
        type='password',
        help="A classic token with 'public_repo' scope. Only needed for the 'Create directly' feature.",
    )
    if write_token:
        os.environ['GITHUB_WRITE_TOKEN'] = write_token.strip()

    st.markdown("---")
    st.markdown("### 💡 How it works")
    st.markdown("""
    1. **Paste** a public GitHub repo URL
    2. **Describe** your skills & level
    3. **Click** Find issues for me
    4. Gemma 4 reads the repo, matches issues to your skills, and finds new tasks
    5. **Pick** a task to see full details, where to change, and a PR draft
    6. **Start contributing!**
    """)

    st.markdown("---")
    st.caption("Built with Gemma 4 · Powered by Gemini API")


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def render_badges(candidate: dict):
    """Render source, difficulty, time, and skill badges."""
    source = candidate.get('source', 'existing')
    issue_num = candidate.get('issue_number')
    difficulty = candidate.get('difficulty', 'medium')
    time_est = candidate.get('estimated_time', '')
    skills = candidate.get('skill_match', [])

    badges_html = ''

    # Source badge
    if source == 'existing' and issue_num:
        badges_html += f'<span class="badge badge-existing">Existing #{issue_num}</span>'
    else:
        badges_html += '<span class="badge badge-suggested">New · Found by Gemma</span>'

    # Difficulty badge
    diff_class = 'badge-easy' if difficulty == 'easy' else 'badge-medium'
    badges_html += f'<span class="badge {diff_class}">{difficulty}</span>'

    # Time badge
    if time_est:
        badges_html += f'<span class="badge badge-time">⏱ {time_est}</span>'

    # Skill badges
    for skill in skills[:4]:
        badges_html += f'<span class="badge badge-skill">{skill}</span>'

    return badges_html


def render_where_tags(where: list[dict]):
    """Render file location tags."""
    html = ''
    for w in where[:5]:
        path = w.get('path', '')
        lines = w.get('lines', '')
        if path:
            label = f"{path}"
            if lines:
                label += f":{lines}"
            html += f'<span class="where-tag">📄 {label}</span>'
    return html


def render_candidate_card(c: dict, index: int):
    """Render a single candidate card in st.container(border=True)."""
    with st.container(border=True):
        badges = render_badges(c)
        where_html = render_where_tags(c.get('where', []))
        summary = c.get('summary', '')
        what = c.get('what_to_change', '')
        title = c.get('title', 'Untitled')

        card_html = f"""
        <div class="candidate-card-inner">
            <div class="card-title">{title}</div>
            <div style="margin-bottom: 0.6rem">{badges}</div>
            <div class="card-summary">{summary}</div>
            <div class="card-change">💡 {what}</div>
            <div style="margin-top: 0.5rem">{where_html}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # View details button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔍 View details", key=f"view_{c['id']}", use_container_width=True):
                st.session_state.selected_id = c['id']
                st.rerun()


def render_detail_view(detail: dict, ctx: dict):
    """Render the full detail view for a candidate."""
    repo = ctx['repo']
    branch = ctx.get('branch', 'main')

    # Back button
    if st.button("← Back to results", key="back_to_results"):
        st.session_state.selected_id = None
        st.rerun()

    # Header
    st.markdown(f"""
    <div class="detail-header animate-in">
        <h2>{detail.get('title', 'Untitled')}</h2>
        <div>{render_badges(detail)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Warnings
    for w in detail.get('warnings', []):
        if w:
            st.warning(w, icon="⚠️")

    # ── Problem ──
    st.markdown("### 🔎 Problem")
    st.markdown(detail.get('problem', 'No description available.'))

    # ── Where ──
    st.markdown("### 📍 Where")
    where = detail.get('where', [])
    for w in where:
        path = w.get('path', '')
        lines = w.get('lines', '')
        explanation = w.get('explanation', '')
        snippet = w.get('snippet', '')

        link = file_link(repo, branch, path, lines)
        st.markdown(f"**[`{path}:{lines}`]({link})**")

        if explanation:
            st.markdown(f"> {explanation}")

        if snippet:
            # Try to detect language from extension
            ext = path.rsplit('.', 1)[-1] if '.' in path else ''
            lang_map = {
                'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                'jsx': 'jsx', 'tsx': 'tsx', 'rb': 'ruby', 'go': 'go',
                'rs': 'rust', 'java': 'java', 'cpp': 'cpp', 'c': 'c',
                'cs': 'csharp', 'php': 'php', 'swift': 'swift',
                'kt': 'kotlin', 'sh': 'bash', 'yml': 'yaml', 'yaml': 'yaml',
                'json': 'json', 'md': 'markdown', 'html': 'html', 'css': 'css',
                'sql': 'sql', 'r': 'r',
            }
            lang = lang_map.get(ext, '')
            st.code(snippet, language=lang)

    # ── What to change ──
    st.markdown("### 🛠️ What to change")
    wtc = detail.get('what_to_change', [])
    for i, step in enumerate(wtc, 1):
        if isinstance(step, dict):
            path = step.get('path', '')
            change = step.get('change', '')
            path_text = f" in `{path}`" if path else ""
            st.markdown(f"<div style='margin-bottom:0.5rem'><span class='step-number'>{i}</span> {change}{path_text}</div>", unsafe_allow_html=True)
        elif isinstance(step, str):
            st.markdown(f"<div style='margin-bottom:0.5rem'><span class='step-number'>{i}</span> {step}</div>", unsafe_allow_html=True)

    # ── How to verify ──
    st.markdown("### ✅ How to verify")
    verify = detail.get('how_to_verify', [])
    if verify:
        for v in verify:
            st.markdown(f"- {v}")
    else:
        st.markdown("_No verification steps provided._")

    # ── Setup steps ──
    setup = detail.get('setup_steps', [])
    if setup:
        st.markdown("### 🚀 Setup steps")
        for s in setup:
            if s.startswith('```') or s.startswith('$') or s.startswith('pip ') or s.startswith('npm '):
                st.code(s.strip('`').strip(), language='bash')
            else:
                st.markdown(f"- {s}")

    # ── PR Draft ──
    st.markdown("### 📝 PR Draft")
    pr = detail.get('pr', {})
    if pr:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Branch name**")
            st.code(pr.get('branch_name', ''), language=None)
            st.markdown("**Commit message**")
            st.code(pr.get('commit_message', ''), language=None)
        with col2:
            st.markdown("**PR title**")
            st.code(pr.get('title', ''), language=None)

        desc = pr.get('description_markdown', '')
        if desc:
            st.markdown("**PR description**")
            st.code(desc, language='markdown')

    st.markdown("---")

    # ── Issue actions ──
    st.markdown("### 🎯 Issue Actions")

    source = detail.get('source', '')
    issue_num = detail.get('issue_number')

    if source == 'existing' and detail.get('url'):
        st.link_button(
            "Open issue on GitHub",
            detail['url'],
            use_container_width=True,
        )

    elif source == 'suggested':
        st.markdown("**This is a new issue found by Gemma.** Review and edit before creating:")

        draft = detail.get('issue_draft', {}) or {}
        default_title = draft.get('title', detail.get('title', ''))
        default_body = draft.get('body_markdown', '')

        cid = detail.get('id', '')

        # Editable title and body
        edited_title = st.text_input(
            "Issue title",
            value=st.session_state.edited_titles.get(cid, default_title),
            key=f"edit_title_{cid}",
        )
        st.session_state.edited_titles[cid] = edited_title

        edited_body = st.text_area(
            "Issue body (Markdown)",
            value=st.session_state.edited_bodies.get(cid, default_body),
            height=250,
            key=f"edit_body_{cid}",
        )
        st.session_state.edited_bodies[cid] = edited_body

        # Create issue link
        issue_url = new_issue_url(repo, edited_title, edited_body)
        st.link_button(
            "Create issue on GitHub",
            issue_url,
            use_container_width=True,
        )

        # Also show copyable text
        with st.expander("📋 Copy issue text"):
            st.code(f"Title: {edited_title}\n\n{edited_body}", language='markdown')

        # P1: Create directly with write token
        if write_token:
            st.markdown("---")
            st.markdown("**Or create directly via API:**")
            confirm = st.checkbox(
                "I've reviewed the issue and want to create it now",
                key=f"confirm_{cid}",
            )
            if confirm:
                if st.button("⚡ Create directly", key=f"create_{cid}", use_container_width=True):
                    try:
                        write_gh = GitHubClient(token=write_token)
                        result = write_gh.create_issue(repo, edited_title, edited_body)
                        st.success(f"✅ Issue #{result['number']} created! [View on GitHub]({result['url']})")
                    except IssuesDisabled:
                        st.error("Issues are disabled on this repo.")
                    except RateLimitError:
                        st.error("GitHub rate limit hit. Try again later.")
                    except GitHubError as e:
                        st.error(f"GitHub error: {e.text}")
                    except Exception as e:
                        st.error(f"Failed to create issue: {e}")

    # ── What Gemma read (P1) ──
    tool_log = ctx.get('tool_log', [])
    if tool_log:
        with st.expander("🤖 What Gemma read"):
            for entry in tool_log:
                tool = entry.get('tool', '')
                args = entry.get('args', {})
                if tool == 'read_file':
                    st.markdown(f"- 📄 `read_file({args.get('path', '')})`")
                elif tool == 'find_files':
                    st.markdown(f"- 🔍 `find_files(\"{args.get('keyword', '')}\")`")

    # ── Download as Markdown (P1) ──
    md_content = _detail_to_markdown(detail, ctx)
    st.download_button(
        "📥 Download as Markdown",
        data=md_content,
        file_name=f"reposherpa-{detail.get('id', 'detail')}.md",
        mime="text/markdown",
        use_container_width=True,
    )


def _detail_to_markdown(detail: dict, ctx: dict) -> str:
    """Convert a detail object to a Markdown file."""
    repo = ctx['repo']
    branch = ctx.get('branch', 'main')
    lines = [f"# {detail.get('title', 'Task Detail')}", ""]

    source = detail.get('source', '')
    issue_num = detail.get('issue_number')
    if source == 'existing' and issue_num:
        lines.append(f"**Source:** Existing issue #{issue_num}")
    else:
        lines.append("**Source:** New issue (found by Gemma)")

    lines.append(f"**Difficulty:** {detail.get('difficulty', '')} | **Time:** {detail.get('estimated_time', '')}")
    lines.append(f"**Skills:** {', '.join(detail.get('skill_match', []))}")
    lines.append("")

    lines.append("## Problem")
    lines.append(detail.get('problem', ''))
    lines.append("")

    lines.append("## Where")
    for w in detail.get('where', []):
        path = w.get('path', '')
        line_range = w.get('lines', '')
        link = file_link(repo, branch, path, line_range)
        lines.append(f"- [{path}:{line_range}]({link})")
        if w.get('explanation'):
            lines.append(f"  > {w['explanation']}")
        if w.get('snippet'):
            lines.append(f"  ```\n  {w['snippet']}\n  ```")
    lines.append("")

    lines.append("## What to change")
    for i, step in enumerate(detail.get('what_to_change', []), 1):
        if isinstance(step, dict):
            lines.append(f"{i}. {step.get('change', '')} (`{step.get('path', '')}`)")
        else:
            lines.append(f"{i}. {step}")
    lines.append("")

    lines.append("## How to verify")
    for v in detail.get('how_to_verify', []):
        lines.append(f"- {v}")
    lines.append("")

    pr = detail.get('pr', {})
    if pr:
        lines.append("## PR Draft")
        lines.append(f"- **Branch:** `{pr.get('branch_name', '')}`")
        lines.append(f"- **Commit:** `{pr.get('commit_message', '')}`")
        lines.append(f"- **Title:** {pr.get('title', '')}")
        lines.append(f"\n{pr.get('description_markdown', '')}")

    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by [RepoSherpa](https://github.com/{repo}) · Powered by Gemma 4*")

    return '\n'.join(lines)


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────

def main():
    # ── Hero header ──
    st.markdown("""
    <div class="hero-header">
        <h1>🧗 RepoSherpa</h1>
        <p class="tagline">Your AI guide to finding the perfect first contribution</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Check for API key ──
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        st.warning("⚠️ **GEMINI_API_KEY** is required. Enter it in the sidebar on the left or add it to your `.env` file.")
        st.info("💡 Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).")
        st.stop()

    # ── If detail view is selected ──
    if st.session_state.selected_id and st.session_state.candidates:
        cid = st.session_state.selected_id
        candidate = None
        for c in st.session_state.candidates:
            if c['id'] == cid:
                candidate = c
                break

        if not candidate:
            st.error("Candidate not found.")
            st.session_state.selected_id = None
            st.rerun()

        # Load or fetch details
        if cid not in st.session_state.details:
            with st.spinner("🧗 Gemma is reading the relevant files and preparing details…"):
                try:
                    detail = get_details(
                        st.session_state.ctx,
                        candidate,
                        model=model,
                        progress_callback=None,
                    )
                    st.session_state.details[cid] = detail
                except GemmaError as e:
                    st.error(f"❌ Gemma error: {e}")
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("🔄 Retry"):
                            st.rerun()
                    with col2:
                        if st.button("← Back to results"):
                            st.session_state.selected_id = None
                            st.rerun()
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    if st.button("← Back to results"):
                        st.session_state.selected_id = None
                        st.rerun()
                    st.stop()

        detail = st.session_state.details.get(cid)
        if detail:
            render_detail_view(detail, st.session_state.ctx)
        return

    # ── If candidates are loaded, show the list ──
    if st.session_state.candidates:
        ctx = st.session_state.ctx
        repo = ctx['repo']
        meta = ctx.get('meta', {})

        # Repo info bar
        st.markdown(f"""
        <div class="glass-container animate-in" style="text-align:center; padding:1rem 1.5rem;">
            <span style="font-size:1.1rem;font-weight:600;color:#e8e9f0;">
                📦 {meta.get('full_name', repo)}
            </span>
            <span style="color:#6b7280;margin:0 0.5rem;">·</span>
            <span style="color:#9ca3af;">{meta.get('language', 'Unknown')}</span>
            <span style="color:#6b7280;margin:0 0.5rem;">·</span>
            <span style="color:#9ca3af;">⭐ {meta.get('stargazers_count', 0)}</span>
        </div>
        """, unsafe_allow_html=True)

        # Label warning
        if ctx.get('label_warning'):
            st.info(f"ℹ️ {ctx['label_warning']}")

        st.markdown(f"### 🎯 {len(st.session_state.candidates)} tasks matched to your skills")
        st.markdown(f"<p style='color:#8b8fa3;margin-top:-0.5rem'>Skills: {ctx.get('skills', '')} · Level: {ctx.get('level', '')}</p>", unsafe_allow_html=True)

        # Render candidate cards
        for i, c in enumerate(st.session_state.candidates):
            render_candidate_card(c, i)

        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Search again", use_container_width=True):
                st.session_state.ctx = None
                st.session_state.candidates = None
                st.session_state.selected_id = None
                st.session_state.details = {}
                st.session_state.edited_titles = {}
                st.session_state.edited_bodies = {}
                st.rerun()

        return

    # ── Input form ──
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    repo_url = st.text_input(
        "🔗 GitHub repository URL",
        placeholder="https://github.com/owner/repo  or  owner/repo",
        help="Any public GitHub repository. Example: facebook/react",
    )

    skills = st.text_area(
        "🛠️ Your skills",
        placeholder="e.g., Python, JavaScript, basic React, HTML/CSS, SQL, writing documentation…",
        help="List the languages, frameworks, and tools you know. Be specific!",
        height=100,
    )

    level = st.selectbox(
        "📊 Experience level",
        options=['Beginner', 'Intermediate'],
        index=0,
        help="Beginner = first-time contributor. Intermediate = comfortable with git and code reviews.",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Find issues button ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        find_clicked = st.button(
            "🔍 Find issues for me",
            use_container_width=True,
            type="primary",
        )

    if find_clicked:
        # Validate inputs
        if not repo_url:
            st.error("Please enter a GitHub repository URL.")
            st.stop()
        if not skills:
            st.error("Please describe your skills so Gemma can find matching tasks.")
            st.stop()

        # Validate URL format
        try:
            parsed = parse_repo(repo_url)
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        # Run the pipeline
        with st.status("🧗 RepoSherpa is finding tasks for you…", expanded=True) as status:
            try:
                # Stage 0: Build context
                status.update(label="📥 Fetching repository data…")

                def ctx_progress(msg):
                    status.update(label=msg)

                ctx = build_context(
                    repo_url, skills, level,
                    token=token if token else None,
                    progress_callback=ctx_progress,
                )
                st.session_state.ctx = ctx

                st.write(f"✅ Loaded **{len(ctx['tree'])}** files and **{len(ctx['issues'])}** issues from `{ctx['repo']}`")

                # Stage 1: Find candidates
                status.update(label="🤖 Gemma is scanning the code…")

                def cand_progress(msg):
                    status.update(label=msg)

                candidates = find_candidates(
                    ctx, model=model,
                    progress_callback=cand_progress,
                )
                st.session_state.candidates = candidates
                st.session_state.selected_id = None
                st.session_state.details = {}

                status.update(label=f"✅ Found {len(candidates)} tasks!", state="complete")
                st.write(f"✅ **{len(candidates)}** tasks matched to your skills")

            except RepoNotFound:
                status.update(label="❌ Repository not found", state="error")
                st.error("🔍 Couldn't find that public repo. Check the URL and make sure it's a public repository.")
                st.stop()
            except RateLimitError:
                status.update(label="❌ Rate limit hit", state="error")
                st.error("⏳ GitHub rate limit hit. Add a personal access token in the sidebar to increase the limit.")
                st.stop()
            except IssuesDisabled:
                status.update(label="❌ Issues disabled", state="error")
                st.error("🚫 Issues are disabled on this repository.")
                st.stop()
            except NotEnoughCandidates as e:
                status.update(label="⚠️ Not enough matches", state="error")
                st.warning(f"⚠️ {e}")
                st.info("Try a different repo, add more skills, or change your experience level.")
                st.stop()
            except GemmaError as e:
                status.update(label="❌ Gemma error", state="error")
                st.error(f"🤖 {e}")
                st.stop()
            except GitHubError as e:
                status.update(label="❌ GitHub error", state="error")
                st.error(f"⚠️ GitHub API error ({e.status}): {e.text}")
                st.stop()
            except Exception as e:
                status.update(label="❌ Unexpected error", state="error")
                st.error(f"❌ Something went wrong: {e}")
                st.stop()

        # Rerun to show results
        st.rerun()

    # ── Empty state ──
    if not st.session_state.candidates and not find_clicked:
        st.markdown("""
        <div class="empty-state animate-in">
            <div class="emoji">🧗‍♂️</div>
            <p>Paste a public GitHub repo URL and tell me your skills.<br>
            I'll find the perfect first contribution for you.</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="glass-container" style="text-align:center">
                <div style="font-size:2rem;margin-bottom:0.5rem">🎯</div>
                <div style="font-weight:600;color:#e8e9f0;margin-bottom:0.3rem">Skill-Matched</div>
                <div style="color:#8b8fa3;font-size:0.85rem">Tasks matched to your exact skill set</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="glass-container" style="text-align:center">
                <div style="font-size:2rem;margin-bottom:0.5rem">🔍</div>
                <div style="font-weight:600;color:#e8e9f0;margin-bottom:0.3rem">AI-Discovered</div>
                <div style="color:#8b8fa3;font-size:0.85rem">Gemma finds hidden tasks in the codebase</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="glass-container" style="text-align:center">
                <div style="font-size:2rem;margin-bottom:0.5rem">📝</div>
                <div style="font-weight:600;color:#e8e9f0;margin-bottom:0.3rem">PR-Ready</div>
                <div style="color:#8b8fa3;font-size:0.85rem">Full details, code location, and PR draft</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
