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
