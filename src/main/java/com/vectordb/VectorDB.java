package com.vectordb;

import com.sun.net.httpserver.HttpServer;
import com.vectordb.api.Router;
import com.vectordb.db.DocumentStore;
import com.vectordb.db.VectorStore;
import com.vectordb.ollama.OllamaClient;

import java.net.InetSocketAddress;
import java.util.concurrent.Executors;

/**
 * Entry point for the VectorDB Engine (Java port).
 *
 * Starts an HTTP server on port 8080, pre-loads 20 demo 16D vectors,
 * and wires all REST API routes.
 *
 * Run with:
 *   java -jar build/libs/VectorDB-all.jar
 */
public class VectorDB {

    private static final int    PORT = 8080;
    private static final int    DIMS = 16;

    public static void main(String[] args) throws Exception {
        // ── Initialise components ────────────────────────────────────────
        VectorStore   vdb    = new VectorStore(DIMS);
        DocumentStore docDB  = new DocumentStore();
        OllamaClient  ollama = new OllamaClient();

        // ── Pre-load 20 demo vectors ─────────────────────────────────────
        loadDemo(vdb);

        // ── Check Ollama availability ────────────────────────────────────
        boolean ollamaUp = ollama.isAvailable();

        // ── Print startup banner ─────────────────────────────────────────
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:" + PORT);
        System.out.println(vdb.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        if (ollamaUp) {
            System.out.println("Ollama: ONLINE");
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        } else {
            System.out.println("Ollama: OFFLINE (install from https://ollama.com)");
        }

        // ── Start HTTP server ─────────────────────────────────────────────
        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", PORT), 0);
        server.setExecutor(Executors.newCachedThreadPool());

        Router router = new Router(vdb, docDB, ollama);
        router.register(server);

        server.start();
        System.out.println("Server started — listening on :" + PORT);
    }

    // ── Demo data (20 × 16D vectors, same as C++ main.cpp) ──────────────
    //
    // Dimension layout:
    //   Dims  0- 3: CS category
    //   Dims  4- 7: Math category
    //   Dims  8-11: Food category
    //   Dims 12-15: Sports category

    private static void loadDemo(VectorStore db) {
        var dist = VectorStore.getDistFn("cosine");

        // ─── Computer Science ────────────────────────────────────────────
        db.insert("Linked List: nodes connected by pointers", "cs",
            new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);

        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
            new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);

        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);

        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);

        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
            new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);

        // ─── Mathematics ─────────────────────────────────────────────────
        db.insert("Calculus: derivatives integrals and limits", "math",
            new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);

        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);

        db.insert("Probability: distributions random variables Bayes theorem", "math",
            new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);

        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);

        db.insert("Combinatorics: permutations combinations generating functions", "math",
            new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);

        // ─── Food ────────────────────────────────────────────────────────
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);

        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);

        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);

        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);

        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
            new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);

        // ─── Sports ──────────────────────────────────────────────────────
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);

        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
            new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);

        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);

        db.insert("Chess: openings endgames tactics strategic board game", "sports",
            new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);

        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
    }
}
