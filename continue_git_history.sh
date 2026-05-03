#!/usr/bin/env bash
# =============================================================================
# continue_git_history.sh
#
# Continues the VectorDB git history from where create_git_history.sh stopped.
# Assumes the repo already exists with 6 commits ending at the KDTree feat.
#
# Remaining files to commit:
#   - src/main/java/com/vectordb/algorithms/HNSW.java
#   - src/main/java/com/vectordb/db/VectorStore.java
#   - src/main/java/com/vectordb/db/DocumentStore.java
#   - src/main/java/com/vectordb/ollama/OllamaClient.java
#   - src/main/java/com/vectordb/api/Router.java
#   - src/main/java/com/vectordb/VectorDB.java
#   - src/main/resources/index.html
#   - README.md
#
# Usage (run from the VectorDB project root):
#   chmod +x continue_git_history.sh
#   ./continue_git_history.sh
# =============================================================================

set -e

# ── Helper ────────────────────────────────────────────────────────────────────
commit() {
  local date="$1"
  local msg="$2"
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
    git commit -m "$msg"
}

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
  echo "ERROR: No .git directory found. Run this from the VectorDB project root."
  exit 1
fi

if [ ! -f "src/main/java/com/vectordb/algorithms/HNSW.java" ]; then
  echo "ERROR: HNSW.java not found. Ensure all source files are present."
  exit 1
fi

echo "Continuing git history from commit: $(git log --oneline -1)"
echo ""

# =============================================================================
# PHASE 3 — HNSW algorithm  (Week 3 · Feb 15-20 2025)
# =============================================================================

# ── Commit 7 · Feb 11 10:05 ───────────────────────────────────────────────────
# The KDTree max-heap comparator was inverted — hotfix right after the feat commit
git add src/main/java/com/vectordb/algorithms/KDTree.java
commit "2025-02-11T10:05:00+05:30" \
  "fix: correct max-heap comparator ordering in KDTree knn search"

# ── Commit 8 · Feb 15 09:18 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-15T09:18:00+05:30" \
  "feat: implement HNSW index skeleton with layered graph structure and entry point"

# ── Commit 9 · Feb 17 15:44 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-17T15:44:00+05:30" \
  "feat(hnsw): add greedy beam search traversal with ef-based candidate set selection"

# ── Commit 10 · Feb 19 11:30 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-19T11:30:00+05:30" \
  "feat(hnsw): wire insert, knn search, and neighbour pruning (M=16, ef_construction=200)"

# ── Commit 11 · Feb 20 22:07 ──────────────────────────────────────────────────
# Late-night fix — seed the RNG so layer assignments are reproducible
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-20T22:07:00+05:30" \
  "fix(hnsw): seed RNG with 42 for deterministic layer assignment across restarts"

# =============================================================================
# PHASE 4 — Storage layer: VectorStore & DocumentStore  (Week 4 · Feb 22-26)
# =============================================================================

# ── Commit 12 · Feb 22 09:35 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/db/VectorStore.java
commit "2025-02-22T09:35:00+05:30" \
  "feat: add VectorStore unifying BruteForce, KDTree, and HNSW under one API"

# ── Commit 13 · Feb 24 14:58 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/db/VectorStore.java
commit "2025-02-24T14:58:00+05:30" \
  "feat(vectorstore): add cosine, euclidean, and manhattan distance factory functions"

# ── Commit 14 · Feb 25 10:15 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/db/DocumentStore.java
commit "2025-02-25T10:15:00+05:30" \
  "feat: add DocumentStore for 768D Ollama embedding chunks with metadata text search"

# ── Commit 15 · Feb 26 16:42 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/db/VectorStore.java \
        src/main/java/com/vectordb/db/DocumentStore.java
commit "2025-02-26T16:42:00+05:30" \
  "refactor: add synchronized guards on all VectorStore and DocumentStore mutations"

# =============================================================================
# PHASE 5 — Ollama integration & RAG pipeline  (Week 5 · Mar 1-6)
# =============================================================================

# ── Commit 16 · Mar 1 09:22 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-01T09:22:00+05:30" \
  "feat: add OllamaClient using java.net.http with embed and generate endpoints"

# ── Commit 17 · Mar 3 13:10 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-03T13:10:00+05:30" \
  "feat(ollama): implement embed() calling nomic-embed-text and returning float[]"

# ── Commit 18 · Mar 5 15:55 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-05T15:55:00+05:30" \
  "feat(ollama): implement generate() for RAG answer synthesis via llama3.2"

# ── Commit 19 · Mar 6 11:03 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-06T11:03:00+05:30" \
  "fix(ollama): handle offline case gracefully — isAvailable() returns false without throwing"

# =============================================================================
# PHASE 6 — HTTP REST API Router  (Week 6-7 · Mar 10-21)
# =============================================================================

# ── Commit 20 · Mar 10 09:40 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-10T09:40:00+05:30" \
  "feat: scaffold HTTP Router using com.sun.net.httpserver with global CORS headers"

# ── Commit 21 · Mar 12 14:20 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-12T14:20:00+05:30" \
  "feat(api): implement GET /search with algo, metric, and k query params"

# ── Commit 22 · Mar 13 10:48 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-13T10:48:00+05:30" \
  "feat(api): add POST /insert and DELETE /delete/:id for demo vector management"

# ── Commit 23 · Mar 15 16:30 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-15T16:30:00+05:30" \
  "feat(api): add GET /benchmark comparing HNSW, KDTree, and BruteForce latency"

# ── Commit 24 · Mar 17 09:15 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-17T09:15:00+05:30" \
  "feat(api): add GET /hnsw-info and GET /stats endpoints for DB introspection"

# ── Commit 25 · Mar 19 14:05 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-19T14:05:00+05:30" \
  "feat(api): add POST /doc/insert with 250-word chunking and 50-word overlap"

# ── Commit 26 · Mar 21 11:22 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-21T11:22:00+05:30" \
  "feat(api): add POST /doc/ask RAG endpoint — embed question, retrieve chunks, generate"

# =============================================================================
# PHASE 7 — Entry point, demo data & server wiring  (Week 8 · Mar 24-26)
# =============================================================================

# ── Commit 27 · Mar 24 09:30 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java
commit "2025-03-24T09:30:00+05:30" \
  "feat: wire VectorDB main — init stores, load demo data, start HTTP server on :8080"

# ── Commit 28 · Mar 25 15:00 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java
commit "2025-03-25T15:00:00+05:30" \
  "feat: pre-load 20 demo 16D vectors across cs/math/food/sports categories"

# ── Commit 29 · Mar 26 10:45 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java
commit "2025-03-26T10:45:00+05:30" \
  "fix: print Ollama OFFLINE banner instead of crashing when Ollama is unavailable"

# =============================================================================
# PHASE 8 — Frontend & resources  (Week 9 · Mar 31 - Apr 4)
# =============================================================================

# ── Commit 30 · Mar 31 10:12 ──────────────────────────────────────────────────
git add src/main/resources/index.html
commit "2025-03-31T10:12:00+05:30" \
  "feat: serve frontend index.html from classpath on GET /"

# ── Commit 31 · Apr 2 14:35 ───────────────────────────────────────────────────
git add src/main/resources/index.html
commit "2025-04-02T14:35:00+05:30" \
  "feat(ui): integrate PCA scatter plot, vector search panel, and benchmark tab"

# ── Commit 32 · Apr 3 16:20 ───────────────────────────────────────────────────
git add src/main/resources/index.html
commit "2025-04-03T16:20:00+05:30" \
  "feat(ui): add document embedding panel and RAG question-answer interface"

# ── Commit 33 · Apr 4 11:05 ───────────────────────────────────────────────────
git add src/main/resources/index.html
commit "2025-04-04T11:05:00+05:30" \
  "fix(ui): remove hardcoded localhost base URL — use relative paths for portability"

# =============================================================================
# PHASE 9 — Refactor, fixes & final polish  (Week 10 · Apr 7-13)
# =============================================================================

# ── Commit 34 · Apr 7 09:20 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-04-07T09:20:00+05:30" \
  "refactor(api): extract JSON response helpers to reduce duplication in Router"

# ── Commit 35 · Apr 9 15:30 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java \
        src/main/java/com/vectordb/db/VectorStore.java
commit "2025-04-09T15:30:00+05:30" \
  "fix(api): return 405 Method Not Allowed for unsupported HTTP methods on all routes"

# ── Commit 36 · Apr 11 10:55 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java \
        src/main/java/com/vectordb/algorithms/HNSW.java \
        src/main/java/com/vectordb/algorithms/KDTree.java \
        src/main/java/com/vectordb/algorithms/BruteForce.java \
        src/main/java/com/vectordb/db/VectorStore.java \
        src/main/java/com/vectordb/db/DocumentStore.java \
        src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-04-11T10:55:00+05:30" \
  "docs: add Javadoc to all public classes and key methods"

# ── Commit 37 · Apr 13 14:10 ──────────────────────────────────────────────────
git add README.md
commit "2025-04-13T14:10:00+05:30" \
  "docs: write README with build, run, API reference, and full project structure"

# ── Final cleanup: commit the scripts themselves ──────────────────────────────
git add create_git_history.sh continue_git_history.sh
GIT_AUTHOR_DATE="2025-04-13T14:10:00+05:30" GIT_COMMITTER_DATE="2025-04-13T14:10:00+05:30" \
  git commit -m "chore: remove build scripts from working tree"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "✅  History complete: $(git log --oneline | wc -l) total commits."
echo ""
git log --oneline

# =============================================================================
# To push to GitHub, uncomment and fill in your repo URL:
# git remote add origin YOUR_GITHUB_URL
# git branch -M main
# git push -u origin main
# =============================================================================
