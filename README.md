# VectorDB — Java Port

A pure-Java port of the C++ Vector Database with HNSW, KD-Tree, and Brute Force search, a built-in REST API server, and a frontend with PCA scatter plot, document embedding, and a RAG pipeline via Ollama.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Java (JDK)  | 17+     |
| Ollama      | any (for document embedding & RAG) |

> **Ollama** is optional — the demo vector search works fully offline.  
> RAG and document embedding require Ollama running locally.

---

## Build

```bash
cd VectorDB
./gradlew shadowJar
```

Produces: `build/libs/VectorDB-all.jar` (self-contained fat JAR, ~326 KB)

---

## Run

```bash
java -jar build/libs/VectorDB-all.jar
```

Expected startup output:
```
=== VectorDB Engine (Java) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
Server started — listening on :8080
```

---

## Verify

Open your browser: **http://localhost:8080**

Or test the API directly:
```bash
# Health check
curl http://localhost:8080/stats

# Search (cosine, HNSW, 16D vector)
curl "http://localhost:8080/search?v=0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.1,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06&k=3&metric=cosine&algo=hnsw"

# Benchmark all 3 algorithms
curl "http://localhost:8080/benchmark?v=0.9,0.85,0.72,0.68,0.12,0.08,0.15,0.1,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06&k=5&metric=cosine"

# HNSW graph stats
curl http://localhost:8080/hnsw-info

# Ollama/model status
curl http://localhost:8080/status
```

---

## Ollama Setup (for RAG)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull nomic-embed-text
ollama pull gemma4:e2b

# Ollama runs automatically in the background
```

Then use the frontend at http://localhost:8080 to insert documents and ask questions.

---

## API Reference

| Method   | Endpoint          | Description |
|----------|-------------------|-------------|
| `GET`    | `/`               | Frontend (index.html) |
| `GET`    | `/search`         | k-NN search — params: `v`, `k`, `metric`, `algo` |
| `POST`   | `/insert`         | Insert demo vector — body: `{metadata, category, embedding}` |
| `DELETE` | `/delete/:id`     | Remove a demo vector |
| `GET`    | `/items`          | List all demo vectors |
| `GET`    | `/benchmark`      | Benchmark all 3 algos — params: `v`, `k`, `metric` |
| `GET`    | `/hnsw-info`      | HNSW layer statistics (nodes, edges per layer) |
| `GET`    | `/stats`          | DB statistics (count, dims, algorithms, metrics) |
| `POST`   | `/doc/insert`     | Insert document — body: `{title, text}` → chunk + embed + store |
| `GET`    | `/doc/list`       | List stored document chunks |
| `DELETE` | `/doc/delete/:id` | Remove a document chunk |
| `POST`   | `/doc/ask`        | RAG pipeline — body: `{question, k}` |
| `GET`    | `/status`         | Ollama online/offline, model names, doc/demo counts |

### Search parameters
- `metric`: `cosine` (default) | `euclidean` | `manhattan`
- `algo`: `hnsw` (default) | `kdtree` | `bruteforce`
- `k`: number of results (default 5)

---

## Project Structure

```
VectorDB/
├── build.gradle.kts              ← Gradle Kotlin DSL + shadowJar
├── settings.gradle.kts
├── gradlew / gradlew.bat
├── gradle/wrapper/
└── src/main/
    ├── java/com/vectordb/
    │   ├── VectorDB.java          ← Entry point + HTTP server startup + demo data
    │   ├── algorithms/
    │   │   ├── BruteForce.java    ← Linear scan O(n·d)
    │   │   ├── KDTree.java        ← KD-Tree with ball-within-hyperslab pruning
    │   │   └── HNSW.java          ← Hierarchical Navigable Small World (M=16, ef=200)
    │   ├── db/
    │   │   ├── VectorStore.java   ← Unified 16D demo DB (all 3 indexes)
    │   │   └── DocumentStore.java ← 768D Ollama embedding store
    │   ├── ollama/
    │   │   └── OllamaClient.java  ← java.net.http client for embed + generate
    │   └── api/
    │       └── Router.java        ← All REST endpoint handlers
    └── resources/
        └── index.html             ← Original frontend (unchanged)
```

---

## Implementation Notes

- **Zero external dependencies** except Gson 2.10.1 (JSON parsing)
- **HTTP server**: `com.sun.net.httpserver` (built into the JDK — no extra dep)
- **Algorithms**: pure Java from scratch — no vector library
- **HNSW**: M=16, ef\_construction=200, mL = 1/ln(M), seeded RNG (42) for reproducibility
- **Chunking**: 250-word chunks with 50-word overlap (matches original C++)
- **CORS**: `Access-Control-Allow-Origin: *` on every response
- **Thread safety**: `synchronized` on all index mutations; cached thread pool for HTTP
