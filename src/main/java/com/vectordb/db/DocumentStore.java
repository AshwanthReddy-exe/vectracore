package com.vectordb.db;

import com.vectordb.algorithms.BruteForce;
import com.vectordb.algorithms.HNSW;

import java.util.*;

import static com.vectordb.db.VectorStore.cosine;

/**
 * Document store backed by a dedicated HNSW index operating on real
 * 768-dimensional Ollama embeddings (nomic-embed-text).
 *
 * Falls back to BruteForce when the collection has fewer than 10 items.
 */
public class DocumentStore {

    public static class DocItem {
        public final int id;
        public final String title;
        public final String text;
        public final float[] embedding;

        public DocItem(int id, String title, String text, float[] embedding) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.embedding = embedding;
        }
    }

    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce();
    private int nextId = 1;
    private int dims = 0;

    public synchronized int insert(String title, String text, float[] embedding) {
        if (dims == 0) dims = embedding.length;
        int id = nextId++;
        DocItem item = new DocItem(id, title, text, embedding);
        store.put(id, item);

        hnsw.insert(id, title, "doc", embedding, VectorStore::cosine);
        BruteForce.Item bfi = new BruteForce.Item(id, title, "doc", embedding);
        bf.insert(bfi);

        return id;
    }

    /**
     * Returns top-k most similar document chunks (distance, DocItem) pairs.
     * Filters by max_dist = 0.7f (cosine distance threshold).
     */
    public synchronized List<float[]> searchRaw(float[] query, int k) {
        if (store.isEmpty()) return Collections.emptyList();
        List<float[]> raw = (store.size() < 10)
            ? bf.knn(query, k, VectorStore::cosine)
            : hnsw.knn(query, k, 50, VectorStore::cosine);
        return raw;
    }

    public synchronized List<AbstractMap.SimpleEntry<Float, DocItem>> search(float[] query, int k) {
        if (store.isEmpty()) return Collections.emptyList();

        List<float[]> raw = searchRaw(query, k);
        List<AbstractMap.SimpleEntry<Float, DocItem>> out = new ArrayList<>();

        for (float[] pair : raw) {
            float dist = pair[0];
            int   id   = (int) pair[1];
            DocItem item = store.get(id);
            if (item != null) {
                out.add(new AbstractMap.SimpleEntry<>(dist, item));
            }
        }
        return out;
    }

    public synchronized boolean remove(int id) {
        if (!store.containsKey(id)) return false;
        store.remove(id);
        hnsw.remove(id);
        bf.remove(id);
        return true;
    }

    public synchronized List<DocItem> all() {
        return new ArrayList<>(store.values());
    }

    public synchronized int size() {
        return store.size();
    }

    public int getDims() {
        return dims;
    }

    public synchronized void removeAll() {
        List<Integer> ids = new ArrayList<>(store.keySet());
        for (int id : ids) remove(id);
    }
}
