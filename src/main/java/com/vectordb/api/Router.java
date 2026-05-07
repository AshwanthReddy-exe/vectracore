package com.vectordb.api;

import com.google.gson.*;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import com.vectordb.algorithms.HNSW;
import com.vectordb.db.DocumentStore;
import com.vectordb.db.VectorStore;
import com.vectordb.ollama.OllamaClient;

import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.BiFunction;

/**
 * REST API router — all endpoint handlers wired to the HttpServer.
 *
 * Endpoints:
 *   GET  /                   serve index.html
 *   GET  /search             k-NN search on demo vectors
 *   POST /insert             insert a demo vector
 *   DELETE /delete/:id       remove a demo vector
 *   GET  /items              list all demo vectors
 *   GET  /benchmark          run all 3 algos, compare timings
 *   GET  /hnsw-info          HNSW layer statistics
 *   GET  /stats              basic DB statistics
 *   POST /doc/insert         chunk+embed+store a document
 *   GET  /doc/list           list stored document chunks
 *   DELETE /doc/delete/:id   remove a document chunk
 *   POST /doc/ask            RAG pipeline (embed→retrieve→generate)
 *   GET  /status             Ollama online/offline + model names
 */
public class Router {

    private final VectorStore    vdb;
    private final DocumentStore  docDB;
    private final OllamaClient   ollama;
    private final Gson           gson = new GsonBuilder().serializeSpecialFloatingPointValues().create();

    // Cache index.html bytes at startup
    private byte[] indexHtml;

    public Router(VectorStore vdb, DocumentStore docDB, OllamaClient ollama) {
        this.vdb    = vdb;
        this.docDB  = docDB;
        this.ollama = ollama;
        loadIndexHtml();
    }

    // ── Register all routes ──────────────────────────────────────────────

    public void register(HttpServer server) {
        server.createContext("/",            this::handleRoot);
        server.createContext("/search",      this::handleSearch);
        server.createContext("/insert",      this::handleInsert);
        server.createContext("/delete",      this::handleDelete);
        server.createContext("/items",       this::handleItems);
        server.createContext("/benchmark",   this::handleBenchmark);
        server.createContext("/hnsw-info",   this::handleHnswInfo);
        server.createContext("/stats",       this::handleStats);
        server.createContext("/doc/insert",  this::handleDocInsert);
        server.createContext("/doc/list",    this::handleDocList);
        server.createContext("/doc/delete",  this::handleDocDelete);
        server.createContext("/doc/ask",     this::handleDocAsk);
        server.createContext("/doc/upload",     this::handleDocUpload);
        server.createContext("/doc/delete/all", this::handleDocDeleteAll);
        server.createContext("/status",      this::handleStatus);
    }

    // ── Utility helpers ──────────────────────────────────────────────────

    private void addCors(HttpExchange ex) {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin",  "*");
        ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    private void sendJson(HttpExchange ex, int status, String json) throws IOException {
        addCors(ex);
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    private void sendError(HttpExchange ex, String msg) throws IOException {
        JsonObject obj = new JsonObject();
        obj.addProperty("error", msg);
        sendJson(ex, 400, gson.toJson(obj));
    }

    private String readBody(HttpExchange ex) throws IOException {
        try (InputStream is = ex.getRequestBody()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    /** Parse ?v=1.0,2.0,... from query string */
    private float[] parseVec(String raw) {
        if (raw == null || raw.isBlank()) return new float[0];
        String[] parts = raw.split(",");
        float[] v = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try { v[i] = Float.parseFloat(parts[i].trim()); }
            catch (NumberFormatException e) { return new float[0]; }
        }
        return v;
    }

    /** Parse a query string like k=5&metric=cosine&algo=hnsw */
    private Map<String, String> parseQuery(URI uri) {
        Map<String, String> params = new HashMap<>();
        String query = uri.getRawQuery();
        if (query == null || query.isBlank()) return params;
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0) {
                String key = java.net.URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8);
                String val = java.net.URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                params.put(key, val);
            }
        }
        return params;
    }

    /** Extract last path segment as integer (e.g. /delete/42 → 42) */
    private int parseIdFromPath(String path) {
        String[] parts = path.split("/");
        return Integer.parseInt(parts[parts.length - 1]);
    }

    private float[] parseVecFromJson(JsonObject obj, String key) {
        if (!obj.has(key)) return new float[0];
        JsonArray arr = obj.getAsJsonArray(key);
        float[] v = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) v[i] = arr.get(i).getAsFloat();
        return v;
    }

    private String vecToJsonArray(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format("%.4f", v[i]));
        }
        return sb.append(']').toString();
    }

    private void loadIndexHtml() {
        try (InputStream is = getClass().getResourceAsStream("/index.html")) {
            if (is != null) indexHtml = is.readAllBytes();
        } catch (IOException e) {
            System.err.println("[WARN] Could not load index.html from resources: " + e.getMessage());
        }
    }

    // ── Handlers ─────────────────────────────────────────────────────────

    // GET /  → serve index.html
    private void handleRoot(HttpExchange ex) throws IOException {
        // Handle CORS preflight
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            addCors(ex);
            ex.sendResponseHeaders(204, -1);
            return;
        }
        String path = ex.getRequestURI().getPath();
        // Only serve root; let other registered contexts handle their paths
        if (!"/".equals(path)) {
            ex.sendResponseHeaders(404, -1);
            return;
        }
        if (indexHtml == null) {
            String notFound = "index.html not found";
            ex.getResponseHeaders().set("Content-Type", "text/plain");
            ex.sendResponseHeaders(404, notFound.length());
            ex.getResponseBody().write(notFound.getBytes());
            ex.getResponseBody().close();
            return;
        }
        ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, indexHtml.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(indexHtml); }
    }

    // GET /search?v=...&k=5&metric=cosine&algo=hnsw
    private void handleSearch(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        Map<String, String> params = parseQuery(ex.getRequestURI());
        float[] q = parseVec(params.get("v"));
        if (q.length != vdb.dims) {
            sendError(ex, "need " + vdb.dims + "D vector"); return;
        }
        int k = 5;
        try { k = Integer.parseInt(params.getOrDefault("k", "5")); } catch (NumberFormatException ignored) {}
        String metric = params.getOrDefault("metric", "cosine");
        String algo   = params.getOrDefault("algo",   "hnsw");

        VectorStore.SearchResult out = vdb.search(q, k, metric, algo);

        StringBuilder sb = new StringBuilder("{\"results\":[");
        for (int i = 0; i < out.hits.size(); i++) {
            if (i > 0) sb.append(',');
            VectorStore.Hit h = out.hits.get(i);
            sb.append("{\"id\":").append(h.id)
              .append(",\"metadata\":").append(jsonStr(h.metadata))
              .append(",\"category\":").append(jsonStr(h.category))
              .append(",\"distance\":").append(String.format("%.6f", h.distance))
              .append(",\"embedding\":").append(vecToJsonArray(h.embedding))
              .append('}');
        }
        sb.append("],\"latencyUs\":").append(out.latencyUs)
          .append(",\"algo\":").append(jsonStr(out.algo))
          .append(",\"metric\":").append(jsonStr(out.metric))
          .append('}');

        sendJson(ex, 200, sb.toString());
    }

    // POST /insert  body: {id, label, category, vec:[...]}
    // Also accepts: {metadata, category, embedding:[...]}
    private void handleInsert(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            sendError(ex, "POST required"); return;
        }
        String body = readBody(ex);
        try {
            JsonObject obj = JsonParser.parseString(body).getAsJsonObject();
            // Support both "metadata"/"label" and "embedding"/"vec"
            String meta = obj.has("metadata") ? obj.get("metadata").getAsString()
                        : obj.has("label")    ? obj.get("label").getAsString()
                        : "";
            String cat  = obj.has("category") ? obj.get("category").getAsString() : "";
            float[] emb = parseVecFromJson(obj, "embedding");
            if (emb.length == 0) emb = parseVecFromJson(obj, "vec");

            if (meta.isBlank() || emb.length != vdb.dims) {
                sendError(ex, "invalid body — need metadata/label + " + vdb.dims + "D embedding/vec"); return;
            }
            int id = vdb.insert(meta, cat, emb, VectorStore.getDistFn("cosine"));
            sendJson(ex, 200, "{\"id\":" + id + "}");
        } catch (Exception e) {
            sendError(ex, "invalid JSON: " + e.getMessage());
        }
    }

    // DELETE /delete/:id
    private void handleDelete(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        if (!"DELETE".equalsIgnoreCase(ex.getRequestMethod())) {
            sendError(ex, "DELETE required"); return;
        }
        try {
            int id  = parseIdFromPath(ex.getRequestURI().getPath());
            boolean ok = vdb.remove(id);
            sendJson(ex, 200, "{\"ok\":" + ok + "}");
        } catch (NumberFormatException e) {
            sendError(ex, "invalid id");
        }
    }

    // GET /items
    private void handleItems(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        List<VectorStore.Item> items = vdb.all();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(',');
            VectorStore.Item v = items.get(i);
            sb.append("{\"id\":").append(v.id)
              .append(",\"metadata\":").append(jsonStr(v.metadata))
              .append(",\"category\":").append(jsonStr(v.category))
              .append(",\"embedding\":").append(vecToJsonArray(v.embedding))
              .append('}');
        }
        sb.append(']');
        sendJson(ex, 200, sb.toString());
    }

    // GET /benchmark?v=...&k=5&metric=cosine
    private void handleBenchmark(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        Map<String, String> params = parseQuery(ex.getRequestURI());
        float[] q = parseVec(params.get("v"));
        if (q.length != vdb.dims) {
            sendError(ex, "need " + vdb.dims + "D vector"); return;
        }
        int k = 5;
        try { k = Integer.parseInt(params.getOrDefault("k", "5")); } catch (NumberFormatException ignored) {}
        String metric = params.getOrDefault("metric", "cosine");

        VectorStore.BenchResult b = vdb.benchmark(q, k, metric);
        String json = "{\"bruteforceUs\":" + b.bruteforceUs
                    + ",\"kdtreeUs\":"     + b.kdtreeUs
                    + ",\"hnswUs\":"       + b.hnswUs
                    + ",\"itemCount\":"    + b.itemCount + "}";
        sendJson(ex, 200, json);
    }

    // GET /hnsw-info
    private void handleHnswInfo(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        HNSW.GraphInfo gi = vdb.hnswInfo();

        StringBuilder sb = new StringBuilder();
        sb.append("{\"topLayer\":").append(gi.topLayer)
          .append(",\"nodeCount\":").append(gi.nodeCount)
          .append(",\"nodesPerLayer\":[");
        for (int i = 0; i < gi.nodesPerLayer.length; i++) {
            if (i > 0) sb.append(','); sb.append(gi.nodesPerLayer[i]);
        }
        sb.append("],\"edgesPerLayer\":[");
        for (int i = 0; i < gi.edgesPerLayer.length; i++) {
            if (i > 0) sb.append(','); sb.append(gi.edgesPerLayer[i]);
        }
        sb.append("],\"nodes\":[");
        for (int i = 0; i < gi.nodes.size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.GraphInfo.NodeView n = gi.nodes.get(i);
            sb.append("{\"id\":").append(n.id)
              .append(",\"metadata\":").append(jsonStr(n.metadata))
              .append(",\"category\":").append(jsonStr(n.category))
              .append(",\"maxLyr\":").append(n.maxLayer)
              .append('}');
        }
        sb.append("],\"edges\":[");
        for (int i = 0; i < gi.edges.size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.GraphInfo.EdgeView e = gi.edges.get(i);
            sb.append("{\"src\":").append(e.src)
              .append(",\"dst\":").append(e.dst)
              .append(",\"lyr\":").append(e.layer)
              .append('}');
        }
        sb.append("]}");
        sendJson(ex, 200, sb.toString());
    }

    // GET /stats
    private void handleStats(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        String json = "{\"count\":"     + vdb.size()
                    + ",\"dims\":"      + vdb.dims
                    + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                    + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        sendJson(ex, 200, json);
    }

    // POST /doc/insert  body: {title, text}
    private void handleDocInsert(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            sendError(ex, "POST required"); return;
        }
        String body = readBody(ex);
        try {
            JsonObject obj   = JsonParser.parseString(body).getAsJsonObject();
            String title = obj.has("title") ? obj.get("title").getAsString() : "";
            String text  = obj.has("text")  ? obj.get("text").getAsString()  : "";

            if (title.isBlank() || text.isBlank()) {
                sendError(ex, "need title and text"); return;
            }

            List<String> chunks = OllamaClient.chunkText(text, 250, 50);
            List<Integer> ids = new ArrayList<>();

            for (int i = 0; i < chunks.size(); i++) {
                float[] emb = ollama.embed(chunks.get(i));
                if (emb.length == 0) {
                    sendError(ex, "Ollama unavailable. Install from https://ollama.com then run: " +
                                  "ollama pull nomic-embed-text && ollama pull llama3.2");
                    return;
                }
                String chunkTitle = chunks.size() > 1
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }

            StringBuilder sb = new StringBuilder("{\"ids\":[");
            for (int i = 0; i < ids.size(); i++) {
                if (i > 0) sb.append(','); sb.append(ids.get(i));
            }
            sb.append("],\"chunks\":").append(chunks.size())
              .append(",\"dims\":").append(docDB.getDims())
              .append('}');
            sendJson(ex, 200, sb.toString());

        } catch (Exception e) {
            sendError(ex, "invalid JSON: " + e.getMessage());
        }
    }

    // GET /doc/list
    private void handleDocList(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        List<DocumentStore.DocItem> docs = docDB.all();
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentStore.DocItem d = docs.get(i);
            String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
            int words = d.text.trim().split("\\s+").length;
            sb.append("{\"id\":").append(d.id)
              .append(",\"title\":").append(jsonStr(d.title))
              .append(",\"preview\":").append(jsonStr(preview))
              .append(",\"words\":").append(words)
              .append('}');
        }
        sb.append(']');
        sendJson(ex, 200, sb.toString());
    }

    // DELETE /doc/delete/:id
    private void handleDocDelete(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        if (!"DELETE".equalsIgnoreCase(ex.getRequestMethod())) {
            sendError(ex, "DELETE required"); return;
        }
        try {
            int id  = parseIdFromPath(ex.getRequestURI().getPath());
            boolean ok = docDB.remove(id);
            sendJson(ex, 200, "{\"ok\":" + ok + "}");
        } catch (NumberFormatException e) {
            sendError(ex, "invalid id");
        }
    }

    // POST /doc/ask  body: {question, k:3}
    private void handleDocAsk(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            sendError(ex, "POST required"); return;
        }
        String body = readBody(ex);
        try {
            JsonObject obj = JsonParser.parseString(body).getAsJsonObject();
            String question = obj.has("question") ? obj.get("question").getAsString() : "";
            int k = obj.has("k") ? obj.get("k").getAsInt() : 3;

            if (question.isBlank()) {
                sendError(ex, "need question"); return;
            }

            // Step 1: embed the question
            float[] qEmb = ollama.embed(question);
            if (qEmb.length == 0) {
                sendError(ex, "Ollama unavailable"); return;
            }

            // Step 2: retrieve top-k chunks
            List<AbstractMap.SimpleEntry<Float, DocumentStore.DocItem>> hits = docDB.search(qEmb, k);

            // Step 3: build prompt
            StringBuilder ctx = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                ctx.append('[').append(i + 1).append("] Source: ")
                   .append(hits.get(i).getValue().title).append("\n")
                   .append(hits.get(i).getValue().text).append("\n\n");
            }
            String prompt =
                "You are a helpful assistant. Answer the user's question directly. " +
                "Use the provided context if it contains relevant information. " +
                "If it doesn't, just use your own general knowledge. " +
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like " +
                "'the context doesn't mention'. Just answer the question naturally.\n\n" +
                "Context:\n" + ctx +
                "Question: " + question + "\n\nAnswer:";

            // Step 4: generate
            String answer = ollama.generate(prompt);

            // Step 5: return
            StringBuilder sb = new StringBuilder();
            sb.append("{\"answer\":").append(jsonStr(answer))
              .append(",\"model\":").append(jsonStr(ollama.genModel))
              .append(",\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) sb.append(',');
                DocumentStore.DocItem d = hits.get(i).getValue();
                float dist = hits.get(i).getKey();
                sb.append("{\"id\":").append(d.id)
                  .append(",\"title\":").append(jsonStr(d.title))
                  .append(",\"text\":").append(jsonStr(d.text))
                  .append(",\"distance\":").append(String.format("%.4f", dist))
                  .append('}');
            }
            sb.append("],\"docCount\":").append(docDB.size()).append('}');
            sendJson(ex, 200, sb.toString());

        } catch (Exception e) {
            sendError(ex, "invalid JSON: " + e.getMessage());
        }
    }

    // GET /status
    private void handleStatus(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        boolean up = ollama.isAvailable();
        String json = "{\"ollamaAvailable\":" + up
                    + ",\"embedModel\":"       + jsonStr(ollama.embedModel)
                    + ",\"genModel\":"         + jsonStr(ollama.genModel)
                    + ",\"docCount\":"         + docDB.size()
                    + ",\"docDims\":"          + docDB.getDims()
                    + ",\"demoDims\":"         + vdb.dims
                    + ",\"demoCount\":"        + vdb.size() + "}";
        sendJson(ex, 200, json);
    }

    // ── File upload handler ──────────────────────────────────────────────

    private void handleDocUpload(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
            sendError(ex, "POST required"); return;
        }

        String contentType = ex.getRequestHeaders().getFirst("Content-Type");
        if (contentType == null || !contentType.contains("multipart/form-data")) {
            sendError(ex, "multipart/form-data required"); return;
        }

        String boundary = null;
        for (String part : contentType.split(";")) {
            part = part.trim();
            if (part.startsWith("boundary=")) {
                boundary = part.substring("boundary=".length()).trim();
                break;
            }
        }
        if (boundary == null) { sendError(ex, "missing boundary"); return; }

        byte[] body = ex.getRequestBody().readAllBytes();
        String bodyStr = new String(body, StandardCharsets.ISO_8859_1);

        String delimiter = "--" + boundary;
        String[] rawParts = bodyStr.split(java.util.regex.Pattern.quote(delimiter));

        String title = null;
        byte[] fileBytes = null;
        String fileName = null;

        for (String rawPart : rawParts) {
            if (rawPart.trim().isEmpty() || rawPart.trim().equals("--")) continue;
            int headerEnd = rawPart.indexOf("\r\n\r\n");
            if (headerEnd < 0) continue;
            String headers = rawPart.substring(0, headerEnd);
            String partBodyStr = rawPart.substring(headerEnd + 4);
            if (partBodyStr.endsWith("\r\n")) partBodyStr = partBodyStr.substring(0, partBodyStr.length() - 2);

            String disposition = "";
            for (String h : headers.split("\r\n")) {
                if (h.toLowerCase().startsWith("content-disposition")) { disposition = h; break; }
            }

            String fieldName = extractDispositionParam(disposition, "name");
            String fn = extractDispositionParam(disposition, "filename");

            if ("title".equals(fieldName)) {
                title = partBodyStr.trim();
            } else if ("file".equals(fieldName) && fn != null && !fn.isEmpty()) {
                fileName = fn;
                fileBytes = partBodyStr.getBytes(StandardCharsets.ISO_8859_1);
            }
        }

        if (fileBytes == null || fileName == null) {
            sendError(ex, "no file found in upload"); return;
        }
        if (title == null || title.isEmpty()) title = fileName;

        String ext = fileName.contains(".") ? fileName.substring(fileName.lastIndexOf('.') + 1).toLowerCase() : "";
        String text;
        try {
            if ("pdf".equals(ext)) {
                text = com.vectordb.util.FileParser.parsePdf(fileBytes);
            } else if ("txt".equals(ext) || "md".equals(ext)) {
                text = com.vectordb.util.FileParser.parseText(new String(fileBytes, StandardCharsets.UTF_8));
            } else {
                sendError(ex, "Unsupported file type. Only PDF, TXT, and MD files are accepted."); return;
            }
        } catch (Exception e) {
            sendError(ex, "File parsing failed: " + e.getMessage()); return;
        }

        if (text.isBlank()) { sendError(ex, "File appears to be empty or unreadable."); return; }

        List<String> chunks = OllamaClient.chunkText(text, 250, 50);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) {
                sendError(ex, "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text"); return;
            }
            String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        float[] proxy = new float[vdb.dims];
        java.util.Arrays.fill(proxy, 0.5f);
        vdb.insert(title, "doc", proxy, VectorStore.getDistFn("cosine"));

        StringBuilder sb = new StringBuilder("{\"ok\":true,\"title\":");
        sb.append(jsonStr(title)).append(",\"chunks\":").append(chunks.size())
          .append(",\"dims\":").append(docDB.getDims()).append('}');
        sendJson(ex, 200, sb.toString());
    }

    private void handleDocDeleteAll(HttpExchange ex) throws IOException {
        addCors(ex);
        if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
            ex.sendResponseHeaders(204, -1); return;
        }
        docDB.removeAll();
        vdb.removeAllByCategory("doc");
        sendJson(ex, 200, "{\"ok\":true}");
    }

    private String extractDispositionParam(String disposition, String param) {
        for (String part : disposition.split(";")) {
            part = part.trim();
            if (part.startsWith(param + "=")) {
                String val = part.substring(param.length() + 1).trim();
                if (val.startsWith("\"") && val.endsWith("\"")) val = val.substring(1, val.length() - 1);
                return val;
            }
        }
        return null;
    }

    // ── JSON string escaping ──────────────────────────────────────────────

    private static String jsonStr(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n");  break;
                case '\r': sb.append("\\r");  break;
                case '\t': sb.append("\\t");  break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else          sb.append(c);
            }
        }
        return sb.append('"').toString();
    }
}
