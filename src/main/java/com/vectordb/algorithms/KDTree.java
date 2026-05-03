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
