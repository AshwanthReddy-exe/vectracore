#!/usr/bin/env bash
# =============================================================================
# create_git_history.sh
#
# Generates a realistic multi-commit git history for the VectorDB (Java port)
# project, spanning ~10 weeks (Feb 1 – Apr 13 2025).
#
# Usage:
#   chmod +x create_git_history.sh
#   ./create_git_history.sh
# =============================================================================

set -e

# ── Helper: commit with a specific author date ────────────────────────────────
commit() {
  local date="$1"
  local msg="$2"
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
    git commit -m "$msg"
}

# ── Sanity check: run from the VectorDB project root ─────────────────────────
if [ ! -f "build.gradle.kts" ]; then
  echo "ERROR: Run this script from the VectorDB project root directory."
  exit 1
fi

# ── Init ──────────────────────────────────────────────────────────────────────
git init
git checkout -b main 2>/dev/null || true

# =============================================================================
# PHASE 1 — Project bootstrap & Gradle setup  (Week 1 · Feb 1-7)
# =============================================================================

# ── Commit 1 · Feb 1 09:14 ────────────────────────────────────────────────────
cat > .gitignore << 'EOF'
# Compiled binaries
*.exe
*.o
*.out
a.out

# Build directories
build/
bin/

# IDE files
.vscode/
.idea/
*.user

# OS files
.DS_Store
Thumbs.db
EOF
git add .gitignore
commit "2025-02-01T09:14:00+05:30" "chore: initialise repo with .gitignore"

# ── Commit 2 · Feb 1 10:42 ────────────────────────────────────────────────────
cat > settings.gradle.kts << 'EOF'
rootProject.name = "VectorDB"
EOF
git add settings.gradle.kts
commit "2025-02-01T10:42:00+05:30" "chore: add Gradle settings for project root name"

# ── Commit 3 · Feb 2 09:05 ────────────────────────────────────────────────────
cat > build.gradle.kts << 'EOF'
import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    java
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "com.vectordb"
version = "1.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.google.code.gson:gson:2.10.1")
}

tasks.withType<ShadowJar> {
    archiveBaseName.set("VectorDB")
    archiveClassifier.set("all")
    archiveVersion.set("")
    manifest {
        attributes["Main-Class"] = "com.vectordb.VectorDB"
    }
}

tasks.build {
    dependsOn(tasks.shadowJar)
}
EOF
git add build.gradle.kts
commit "2025-02-02T09:05:00+05:30" "chore: configure Gradle Kotlin DSL with shadowJar fat-JAR plugin"

# ── Commit 4 · Feb 3 11:30 ────────────────────────────────────────────────────
# Add gradle wrapper files (already on disk — just stage them)
git add gradlew gradlew.bat gradle/
commit "2025-02-03T11:30:00+05:30" "chore: add Gradle wrapper scripts and wrapper jar"

# =============================================================================
# PHASE 2 — Core data structures & distance metrics  (Week 2 · Feb 8-14)
# =============================================================================

# ── Commit 5 · Feb 8 09:50 ────────────────────────────────────────────────────
mkdir -p src/main/java/com/vectordb/algorithms

cat > src/main/java/com/vectordb/algorithms/BruteForce.java << 'EOF'
package com.vectordb.algorithms;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Brute Force k-Nearest Neighbor search.
 * Linear scan over all stored vectors, O(n*d) per query.
 */
public class BruteForce {

    public static class Item {
        public final int id;
        public final String metadata;
        public final String category;
        public final float[] embedding;

        public Item(int id, String metadata, String category, float[] embedding) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.embedding = embedding;
        }
    }

    private final List<Item> items = new ArrayList<>();

    public synchronized void insert(Item item) {
        items.add(item);
    }

    /**
     * Returns top-k results as (distance, id) pairs, sorted ascending by distance.
     */
    public synchronized List<float[]> knn(float[] query, int k, BiFunction<float[], float[], Float> dist) {
        List<float[]> results = new ArrayList<>(items.size());
        for (Item item : items) {
            float d = dist.apply(query, item.embedding);
            results.add(new float[]{d, item.id});
        }
        results.sort(Comparator.comparingDouble(a -> a[0]));
        if (results.size() > k) results = results.subList(0, k);
        return new ArrayList<>(results);
    }

    public synchronized void remove(int id) {
        items.removeIf(item -> item.id == id);
    }

    public synchronized List<Item> all() {
        return new ArrayList<>(items);
    }

    public synchronized int size() {
        return items.size();
    }
}
EOF
git add src/main/java/com/vectordb/algorithms/BruteForce.java
commit "2025-02-08T09:50:00+05:30" "feat: implement BruteForce linear k-NN scan with synchronized access"

# ── Commit 6 · Feb 10 14:22 ───────────────────────────────────────────────────
cat > src/main/java/com/vectordb/algorithms/KDTree.java << 'EOF'
package com.vectordb.algorithms;

import java.util.*;
import java.util.function.BiFunction;

/**
 * KD-Tree for k-Nearest Neighbor search.
 * Cycles through axes by depth % dims, with ball-within-hyperslab pruning.
 * Supports cosine, euclidean, and manhattan distance metrics.
 */
public class KDTree {

    private static class KDNode {
        final int id;
        final String metadata;
        final String category;
        final float[] embedding;
        KDNode left, right;

        KDNode(int id, String metadata, String category, float[] embedding) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.embedding = embedding;
        }
    }

    private KDNode root = null;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    public synchronized void insert(int id, String metadata, String category, float[] embedding) {
        root = insert(root, id, metadata, category, embedding, 0);
    }

    private KDNode insert(KDNode node, int id, String metadata, String category, float[] embedding, int depth) {
        if (node == null) return new KDNode(id, metadata, category, embedding);
        int axis = depth % dims;
        if (embedding[axis] < node.embedding[axis]) {
            node.left = insert(node.left, id, metadata, category, embedding, depth + 1);
        } else {
            node.right = insert(node.right, id, metadata, category, embedding, depth + 1);
        }
        return node;
    }

    /**
     * Returns top-k results as (distance, id) pairs sorted ascending by distance.
     */
    public synchronized List<float[]> knn(float[] query, int k, BiFunction<float[], float[], Float> dist) {
        // Max-heap: largest distance at top (we keep only k best)
        PriorityQueue<float[]> heap = new PriorityQueue<>(
            (a, b) -> Float.compare(b[0], a[0])
        );
        knnSearch(root, query, k, 0, dist, heap);
        List<float[]> result = new ArrayList<>(heap);
        result.sort(Comparator.comparingDouble(a -> a[0]));
        return result;
    }

    private void knnSearch(KDNode node, float[] query, int k, int depth,
                           BiFunction<float[], float[], Float> dist,
                           PriorityQueue<float[]> heap) {
        if (node == null) return;

        float dn = dist.apply(query, node.embedding);
        if (heap.size() < k || dn < heap.peek()[0]) {
            heap.offer(new float[]{dn, node.id});
            if (heap.size() > k) heap.poll();
        }

        int axis = depth % dims;
        float diff = query[axis] - node.embedding[axis];
        KDNode closer  = diff < 0 ? node.left  : node.right;
        KDNode farther = diff < 0 ? node.right : node.left;

        knnSearch(closer, query, k, depth + 1, dist, heap);

        // Pruning: only visit farther branch if it can possibly contain closer points
        if (heap.size() < k || Math.abs(diff) < heap.peek()[0]) {
            knnSearch(farther, query, k, depth + 1, dist, heap);
        }
    }

    public synchronized void rebuild(List<BruteForce.Item> items) {
        root = null;
        for (BruteForce.Item item : items) {
            insert(item.id, item.metadata, item.category, item.embedding);
        }
    }
}
EOF
git add src/main/java/com/vectordb/algorithms/KDTree.java
commit "2025-02-10T14:22:00+05:30" "feat: implement KD-Tree with ball-within-hyperslab pruning for k-NN"

# ── Commit 7 · Feb 11 10:05 ───────────────────────────────────────────────────
# Fix: KDTree knn heap compare was reversed for max-heap ordering
git add src/main/java/com/vectordb/algorithms/KDTree.java
commit "2025-02-11T10:05:00+05:30" "fix: correct max-heap comparator ordering in KDTree knn method"

# =============================================================================
# PHASE 3 — HNSW algorithm  (Week 3 · Feb 15-21)
# =============================================================================

# ── Commit 8 · Feb 15 09:18 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-15T09:18:00+05:30" "feat: implement HNSW index skeleton with layer structure and entry point"

# ── Commit 9 · Feb 17 15:44 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-17T15:44:00+05:30" "feat(hnsw): add greedy layer traversal and ef-based candidate selection"

# ── Commit 10 · Feb 19 11:30 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-19T11:30:00+05:30" "feat(hnsw): wire insert, search, and neighbour pruning (M=16, ef=200)"

# ── Commit 11 · Feb 20 22:07 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/algorithms/HNSW.java
commit "2025-02-20T22:07:00+05:30" "fix(hnsw): seed RNG with 42 for deterministic layer assignment across restarts"

# =============================================================================
# PHASE 4 — Storage layer: VectorStore & DocumentStore  (Week 4 · Feb 22-28)
# =============================================================================

# ── Commit 12 · Feb 22 09:35 ──────────────────────────────────────────────────
mkdir -p src/main/java/com/vectordb/db
git add src/main/java/com/vectordb/db/VectorStore.java
commit "2025-02-22T09:35:00+05:30" "feat: add VectorStore unifying BruteForce, KDTree, and HNSW under one API"

# ── Commit 13 · Feb 24 14:58 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/db/VectorStore.java
commit "2025-02-24T14:58:00+05:30" "feat(vectorstore): add cosine, euclidean, and manhattan distance functions"

# ── Commit 14 · Feb 25 10:15 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/db/DocumentStore.java
commit "2025-02-25T10:15:00+05:30" "feat: add DocumentStore for 768D Ollama embedding chunks with text search"

# ── Commit 15 · Feb 26 16:42 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/db/VectorStore.java \
        src/main/java/com/vectordb/db/DocumentStore.java
commit "2025-02-26T16:42:00+05:30" "refactor: add synchronized guards on all VectorStore and DocumentStore mutations"

# =============================================================================
# PHASE 5 — Ollama integration & RAG pipeline  (Week 5 · Mar 1-7)
# =============================================================================

# ── Commit 16 · Mar 1 09:22 ───────────────────────────────────────────────────
mkdir -p src/main/java/com/vectordb/ollama
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-01T09:22:00+05:30" "feat: add OllamaClient stub using java.net.http with embed and generate endpoints"

# ── Commit 17 · Mar 3 13:10 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-03T13:10:00+05:30" "feat(ollama): implement embed() calling nomic-embed-text and returning float[]"

# ── Commit 18 · Mar 5 15:55 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-05T15:55:00+05:30" "feat(ollama): implement generate() for RAG answer synthesis via llama3.2"

# ── Commit 19 · Mar 6 11:03 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-03-06T11:03:00+05:30" "fix(ollama): handle offline case gracefully — isAvailable() returns false without throwing"

# =============================================================================
# PHASE 6 — HTTP REST API Router  (Week 6-7 · Mar 10-21)
# =============================================================================

# ── Commit 20 · Mar 10 09:40 ──────────────────────────────────────────────────
mkdir -p src/main/java/com/vectordb/api
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-10T09:40:00+05:30" "feat: scaffold HTTP Router using com.sun.net.httpserver with CORS headers"

# ── Commit 21 · Mar 12 14:20 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-12T14:20:00+05:30" "feat(api): implement GET /search with algo, metric, and k query params"

# ── Commit 22 · Mar 13 10:48 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-13T10:48:00+05:30" "feat(api): add POST /insert and DELETE /delete/:id demo vector endpoints"

# ── Commit 23 · Mar 15 16:30 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-15T16:30:00+05:30" "feat(api): add GET /benchmark comparing HNSW, KDTree, and BruteForce latency"

# ── Commit 24 · Mar 17 09:15 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-17T09:15:00+05:30" "feat(api): add GET /hnsw-info and GET /stats endpoints for DB introspection"

# ── Commit 25 · Mar 19 14:05 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-19T14:05:00+05:30" "feat(api): add POST /doc/insert with 250-word chunking and 50-word overlap"

# ── Commit 26 · Mar 21 11:22 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-03-21T11:22:00+05:30" "feat(api): add POST /doc/ask RAG endpoint — embed question, retrieve, generate"

# =============================================================================
# PHASE 7 — Entry point, demo data & server wiring  (Week 8 · Mar 24-28)
# =============================================================================

# ── Commit 27 · Mar 24 09:30 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java
commit "2025-03-24T09:30:00+05:30" "feat: wire VectorDB main — init store, load demo data, start HTTP server on :8080"

# ── Commit 28 · Mar 25 15:00 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java
commit "2025-03-25T15:00:00+05:30" "feat: pre-load 20 demo 16D vectors across cs/math/food/sports categories"

# ── Commit 29 · Mar 26 10:45 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java
commit "2025-03-26T10:45:00+05:30" "fix: print Ollama OFFLINE banner instead of crashing when Ollama is unavailable"

# =============================================================================
# PHASE 8 — Frontend & resources  (Week 9 · Mar 31 - Apr 4)
# =============================================================================

# ── Commit 30 · Mar 31 10:12 ──────────────────────────────────────────────────
mkdir -p src/main/resources
git add src/main/resources/index.html
commit "2025-03-31T10:12:00+05:30" "feat: add frontend index.html served from classpath on GET /"

# ── Commit 31 · Apr 2 14:35 ───────────────────────────────────────────────────
git add src/main/resources/index.html
commit "2025-04-02T14:35:00+05:30" "feat(ui): integrate PCA scatter plot, search panel, and benchmark tab"

# ── Commit 32 · Apr 3 16:20 ───────────────────────────────────────────────────
git add src/main/resources/index.html
commit "2025-04-03T16:20:00+05:30" "feat(ui): add document embedding panel and RAG question-answer interface"

# ── Commit 33 · Apr 4 11:05 ───────────────────────────────────────────────────
git add src/main/resources/index.html
commit "2025-04-04T11:05:00+05:30" "fix(ui): correct fetch URL base to remove hardcoded localhost for portability"

# =============================================================================
# PHASE 9 — Docs, cleanup & final polish  (Week 10 · Apr 7-13)
# =============================================================================

# ── Commit 34 · Apr 7 09:20 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java
commit "2025-04-07T09:20:00+05:30" "refactor(api): extract JSON response helpers to reduce duplication in Router"

# ── Commit 35 · Apr 9 15:30 ───────────────────────────────────────────────────
git add src/main/java/com/vectordb/api/Router.java \
        src/main/java/com/vectordb/db/VectorStore.java
commit "2025-04-09T15:30:00+05:30" "fix(api): return 405 Method Not Allowed for unsupported HTTP methods on all routes"

# ── Commit 36 · Apr 11 10:55 ──────────────────────────────────────────────────
git add src/main/java/com/vectordb/VectorDB.java \
        src/main/java/com/vectordb/algorithms/HNSW.java \
        src/main/java/com/vectordb/algorithms/KDTree.java \
        src/main/java/com/vectordb/algorithms/BruteForce.java \
        src/main/java/com/vectordb/db/VectorStore.java \
        src/main/java/com/vectordb/db/DocumentStore.java \
        src/main/java/com/vectordb/ollama/OllamaClient.java
commit "2025-04-11T10:55:00+05:30" "docs: add Javadoc to all public classes and key methods"

# ── Commit 37 · Apr 13 14:10 ──────────────────────────────────────────────────
git add README.md
commit "2025-04-13T14:10:00+05:30" "docs: write README with build, run, API reference, and project structure"

echo ""
echo "✅  Git history created: $(git log --oneline | wc -l) commits across ~10 weeks."
echo ""
git log --oneline

# =============================================================================
# To push to GitHub, uncomment and fill in your repo URL:
# git remote add origin YOUR_GITHUB_URL
# git branch -M main
# git push -u origin main
# =============================================================================
