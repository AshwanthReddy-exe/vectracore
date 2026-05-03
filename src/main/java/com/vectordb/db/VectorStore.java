package com.vectordb.db;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;
import com.vectordb.algorithms.KDTree;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Unified demo vector database (16-dimensional vectors).
 * Maintains BruteForce, KD-Tree, and HNSW indexes in sync.
 */
public class VectorStore {

    // ── Data model ───────────────────────────────────────────────────────

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

    public static class Hit {
        public final int id;
        public final String metadata;
        public final String category;
        public final float[] embedding;
        public final float distance;

        public Hit(int id, String metadata, String category, float[] embedding, float distance) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.embedding = embedding;
            this.distance = distance;
        }
    }

    public static class SearchResult {
        public final List<Hit> hits;
        public final long latencyUs;
        public final String algo;
        public final String metric;

        public SearchResult(List<Hit> hits, long latencyUs, String algo, String metric) {
            this.hits = hits;
            this.latencyUs = latencyUs;
            this.algo = algo;
            this.metric = metric;
        }
    }

    public static class BenchResult {
        public final long bruteforceUs;
        public final long kdtreeUs;
        public final long hnswUs;
        public final int itemCount;

        public BenchResult(long bruteforceUs, long kdtreeUs, long hnswUs, int itemCount) {
            this.bruteforceUs = bruteforceUs;
            this.kdtreeUs = kdtreeUs;
            this.hnswUs = hnswUs;
            this.itemCount = itemCount;
        }
    }

    // ── State ────────────────────────────────────────────────────────────

    public final int dims;
    private final Map<Integer, Item> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw;
    private int nextId = 1;

    public VectorStore(int dims) {
        this.dims = dims;
        this.kdt  = new KDTree(dims);
        this.hnsw = new HNSW(16, 200);
    }

    // ── Distance functions ───────────────────────────────────────────────

    public static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) { float d = a[i] - b[i]; s += d * d; }
        return (float) Math.sqrt(s);
    }

    public static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
    }

    public static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }

    public static BiFunction<float[], float[], Float> getDistFn(String metric) {
        switch (metric) {
            case "cosine":    return VectorStore::cosine;
            case "manhattan": return VectorStore::manhattan;
            default:          return VectorStore::euclidean;
        }
    }

    // ── Mutations ────────────────────────────────────────────────────────

    public synchronized int insert(String metadata, String category, float[] embedding,
                                   BiFunction<float[], float[], Float> dist) {
        int id = nextId++;
        Item item = new Item(id, metadata, category, embedding);
        store.put(id, item);

        BruteForce.Item bfi = new BruteForce.Item(id, metadata, category, embedding);
        bf.insert(bfi);
        kdt.insert(id, metadata, category, embedding);
        hnsw.insert(id, metadata, category, embedding, dist);

        return id;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        bf.remove(id);
        hnsw.remove(id);
        // KD-Tree doesn't support deletion — rebuild
        List<BruteForce.Item> remaining = bf.all();
        kdt.rebuild(remaining);
        return true;
    }

    // ── Search ───────────────────────────────────────────────────────────

    public synchronized SearchResult search(float[] query, int k, String metric, String algo) {
        BiFunction<float[], float[], Float> dist = getDistFn(metric);

        long t0 = System.nanoTime();
        List<float[]> raw;
        switch (algo) {
            case "bruteforce": raw = bf.knn(query, k, dist); break;
            case "kdtree":     raw = kdt.knn(query, k, dist); break;
            default:           raw = hnsw.knn(query, k, 50, dist); break;
        }
        long us = (System.nanoTime() - t0) / 1000;

        List<Hit> hits = new ArrayList<>();
        for (float[] pair : raw) {
            float d  = pair[0];
            int   id = (int) pair[1];
            Item item = store.get(id);
            if (item != null) {
                hits.add(new Hit(id, item.metadata, item.category, item.embedding, d));
            }
        }
        return new SearchResult(hits, us, algo, metric);
    }

    public synchronized BenchResult benchmark(float[] query, int k, String metric) {
        BiFunction<float[], float[], Float> dist = getDistFn(metric);

        long t0 = System.nanoTime(); bf.knn(query, k, dist);
        long bfUs = (System.nanoTime() - t0) / 1000;

        t0 = System.nanoTime(); kdt.knn(query, k, dist);
        long kdUs = (System.nanoTime() - t0) / 1000;

        t0 = System.nanoTime(); hnsw.knn(query, k, 50, dist);
        long hnswUs = (System.nanoTime() - t0) / 1000;

        return new BenchResult(bfUs, kdUs, hnswUs, store.size());
    }

    // ── Accessors ────────────────────────────────────────────────────────

    public synchronized List<Item> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized HNSW.GraphInfo hnswInfo() {
        return hnsw.getInfo();
    }

    public synchronized int size() {
        return store.size();
    }
}
