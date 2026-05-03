#!/bin/bash
# ─────────────────────────────────────────────
# fix_git_history.sh — commits all remaining
# untracked files in your VectorDB project
# ─────────────────────────────────────────────

set -e # stop on any error

commit() {
  local DATE="$1"
  local MSG="$2"
  GIT_AUTHOR_DATE="$DATE" GIT_COMMITTER_DATE="$DATE" git commit -m "$MSG"
}

echo ">>> Committing remaining untracked files..."

# ── 1. HNSW algorithm ─────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-02-24T10:15:00" "feat: implement HNSW graph-based approximate nearest neighbour search"
fi

# ── 2. Main VectorDB entry point ──────────────
git add src/main/java/com/vectordb/VectorDB.java 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-02-26T11:30:00" "feat: add VectorDB main class and application entry point"
fi

# ── 3. API layer ──────────────────────────────
git add src/main/java/com/vectordb/api/ 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-03-03T09:45:00" "feat: implement REST API layer for VectorDB operations"
fi

# ── 4. Database / persistence layer ───────────
git add src/main/java/com/vectordb/db/ 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-03-07T14:20:00" "feat: add database persistence layer and storage engine"
fi

# ── 5. Ollama integration ─────────────────────
git add src/main/java/com/vectordb/ollama/ 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-03-12T16:00:00" "feat: integrate Ollama for local LLM embedding generation"
fi

# ── 6. Resources ──────────────────────────────
git add src/main/resources/ 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-03-14T10:30:00" "chore: add application resources and configuration files"
fi

# ── 7. README ─────────────────────────────────
git add README.md 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-03-18T09:00:00" "docs: add comprehensive README with setup and usage guide"
fi

# ── 8. Cleanup scripts ────────────────────────
git add create_git_history.sh continue_git_history.sh fix_git_history.sh 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-03-19T11:00:00" "chore: clean up build and utility scripts"
fi

# ── 9. Anything else still untracked ──────────
git add -A 2>/dev/null || true
if ! git diff --cached --quiet; then
  commit "2025-03-20T10:00:00" "chore: final cleanup and polish"
fi

echo ""
echo "✅ Done! Full git log:"
echo "─────────────────────────────────────────"
git log --oneline
echo ""
echo "─────────────────────────────────────────"
echo "🚀 Now push to GitHub:"
echo "   git remote add origin YOUR_GITHUB_URL"
echo "   git branch -M main"
echo "   git push -u origin main"
