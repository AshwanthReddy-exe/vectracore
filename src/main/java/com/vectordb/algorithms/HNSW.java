package com.vectordb.algorithms;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Hierarchical Navigable Small World (HNSW) graph index.
 *
 * Parameters:
 *   M             = 16  (max connections per node per layer, except layer 0 uses M0=2*M)
 *   ef_construction = 200
 *   mL            = 1 / ln(M)   (level generation normalization)
 *
 * Supports cosine, euclidean, and manhattan distance metrics.
 */
public class HNSW {

    // ── Internal node representation ───────────────────────────────────

    public static class Node {
        public final int id;
        public final String metadata;
        public final String category;
        public final float[] embedding;
        public final int maxLayer;
        // neighbors[layer] = list of neighbor IDs at that layer
        public final List<List<Integer>> neighbors;

        public Node(int id, String metadata, String category, float[] embedding, int maxLayer) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.embedding = embedding;
            this.maxLayer = maxLayer;
            this.neighbors = new ArrayList<>();
            for (int i = 0; i <= maxLayer; i++) {
                this.neighbors.add(new ArrayList<>());
            }
        }
    }

    // ── Graph state ─────────────────────────────────────────────────────

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int M;
    private final int M0;
    private final int efConstruction;
    private final double mL;
    private int topLayer = -1;
    private int entryPoint = -1;
    private final Random rng = new Random(42);

    public HNSW(int m, int efConstruction) {
        this.M = m;
        this.M0 = 2 * m;
        this.efConstruction = efConstruction;
        this.mL = 1.0 / Math.log(m);
    }

    // ── Level generation ────────────────────────────────────────────────

    private int randomLevel() {
        double r = rng.nextDouble();
        if (r <= 0.0) r = 1e-10;
        return (int) Math.floor(-Math.log(r) * mL);
    }

    // ── Layer search (beam search) ──────────────────────────────────────

    /**
     * Searches a single layer starting from ep, returning up to ef results
     * sorted ascending by distance (closest first).
     */
    private List<float[]> searchLayer(float[] query, int ep, int ef, int layer,
                                       BiFunction<float[], float[], Float> dist) {
        Set<Integer> visited = new HashSet<>();
        // candidates min-heap (closest at top)
        PriorityQueue<float[]> candidates = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        // found max-heap (farthest at top, for pruning)
        PriorityQueue<float[]> found = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));

        float d0 = dist.apply(query, graph.get(ep).embedding);
        visited.add(ep);
        candidates.offer(new float[]{d0, ep});
        found.offer(new float[]{d0, ep});

        while (!candidates.isEmpty()) {
            float[] current = candidates.poll();
            float cd = current[0];
            int cid = (int) current[1];

            if (found.size() >= ef && cd > found.peek()[0]) break;

            Node cNode = graph.get(cid);
            if (cNode == null || layer >= cNode.neighbors.size()) continue;

            for (int nid : cNode.neighbors.get(layer)) {
                if (visited.contains(nid) || !graph.containsKey(nid)) continue;
                visited.add(nid);
                float nd = dist.apply(query, graph.get(nid).embedding);
                if (found.size() < ef || nd < found.peek()[0]) {
                    candidates.offer(new float[]{nd, nid});
                    found.offer(new float[]{nd, nid});
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<float[]> result = new ArrayList<>(found);
        result.sort(Comparator.comparingDouble(a -> a[0]));
        return result;
    }

    // ── Select neighbors (simple greedy, matches C++ selectNbrs) ────────

    private List<Integer> selectNeighbors(List<float[]> candidates, int maxM) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(candidates.size(), maxM); i++) {
            result.add((int) candidates.get(i)[1]);
        }
        return result;
    }

    // ── Insert ──────────────────────────────────────────────────────────

    public synchronized void insert(int id, String metadata, String category, float[] embedding,
                                    BiFunction<float[], float[], Float> dist) {
        int level = randomLevel();
        Node node = new Node(id, metadata, category, embedding, level);
        graph.put(id, node);

        if (entryPoint == -1) {
            entryPoint = id;
            topLayer = level;
            return;
        }

        int ep = entryPoint;

        // Greedy descent from topLayer down to level+1
        for (int lc = topLayer; lc > level; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.neighbors.size()) {
                List<float[]> w = searchLayer(embedding, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = (int) w.get(0)[1];
            }
        }

        // Insert at each layer from min(topLayer,level) down to 0
        for (int lc = Math.min(topLayer, level); lc >= 0; lc--) {
            List<float[]> w = searchLayer(embedding, ep, efConstruction, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> selected = selectNeighbors(w, maxM);

            node.neighbors.get(lc).addAll(selected);

            // Connect bidirectionally and prune
            for (int nid : selected) {
                Node nNode = graph.get(nid);
                if (nNode == null) continue;
                while (nNode.neighbors.size() <= lc) nNode.neighbors.add(new ArrayList<>());
                List<Integer> conn = nNode.neighbors.get(lc);
                conn.add(id);

                if (conn.size() > maxM) {
                    // Prune to maxM using distances
                    List<float[]> ds = new ArrayList<>();
                    for (int c : conn) {
                        Node cn = graph.get(c);
                        if (cn != null) {
                            ds.add(new float[]{dist.apply(nNode.embedding, cn.embedding), c});
                        }
                    }
                    ds.sort(Comparator.comparingDouble(a -> a[0]));
                    conn.clear();
                    for (int i = 0; i < Math.min(maxM, ds.size()); i++) {
                        conn.add((int) ds.get(i)[1]);
                    }
                }
            }

            if (!w.isEmpty()) ep = (int) w.get(0)[1];
        }

        if (level > topLayer) {
            topLayer = level;
            entryPoint = id;
        }
    }

    // ── Search ──────────────────────────────────────────────────────────

    public synchronized List<float[]> knn(float[] query, int k, int ef,
                                           BiFunction<float[], float[], Float> dist) {
        if (entryPoint == -1) return Collections.emptyList();

        int ep = entryPoint;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.neighbors.size()) {
                List<float[]> w = searchLayer(query, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = (int) w.get(0)[1];
            }
        }

        List<float[]> w = searchLayer(query, ep, Math.max(ef, k), 0, dist);
        if (w.size() > k) w = w.subList(0, k);
        return new ArrayList<>(w);
    }

    // ── Remove ──────────────────────────────────────────────────────────

    public synchronized void remove(int id) {
        if (!graph.containsKey(id)) return;

        for (Node node : graph.values()) {
            for (List<Integer> layer : node.neighbors) {
                layer.removeIf(nid -> nid == id);
            }
        }

        if (entryPoint == id) {
            entryPoint = -1;
            for (int nid : graph.keySet()) {
                if (nid != id) {
                    entryPoint = nid;
                    break;
                }
            }
        }

        graph.remove(id);
    }

    // ── Graph info (for /hnsw-info endpoint) ────────────────────────────

    public static class GraphInfo {
        public int topLayer;
        public int nodeCount;
        public int[] nodesPerLayer;
        public int[] edgesPerLayer;

        public static class NodeView {
            public int id;
            public String metadata;
            public String category;
            public int maxLayer;

            public NodeView(int id, String metadata, String category, int maxLayer) {
                this.id = id;
                this.metadata = metadata;
                this.category = category;
                this.maxLayer = maxLayer;
            }
        }

        public static class EdgeView {
            public int src, dst, layer;

            public EdgeView(int src, int dst, int layer) {
                this.src = src;
                this.dst = dst;
                this.layer = layer;
            }
        }

        public List<NodeView> nodes = new ArrayList<>();
        public List<EdgeView> edges = new ArrayList<>();
    }

    public synchronized GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer  = topLayer;
        gi.nodeCount = graph.size();
        int maxL = Math.max(topLayer + 1, 1);
        gi.nodesPerLayer = new int[maxL];
        gi.edgesPerLayer = new int[maxL];

        for (Map.Entry<Integer, Node> entry : graph.entrySet()) {
            int id   = entry.getKey();
            Node nd  = entry.getValue();
            gi.nodes.add(new GraphInfo.NodeView(id, nd.metadata, nd.category, nd.maxLayer));

            for (int lc = 0; lc <= nd.maxLayer && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < nd.neighbors.size()) {
                    for (int nid : nd.neighbors.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.add(new GraphInfo.EdgeView(id, nid, lc));
                        }
                    }
                }
            }
        }

        return gi;
    }

    public synchronized int size() {
        return graph.size();
    }
}
