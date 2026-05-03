package com.vectordb.ollama;

import com.google.gson.*;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

/**
 * HTTP client for the local Ollama API (http://localhost:11434).
 *
 * Endpoints used:
 * POST /api/embeddings → {model, prompt} → {embedding:[...]}
 * POST /api/generate → {model, prompt, stream:false} → {response:"..."}
 * GET /api/tags → availability check
 */
public class OllamaClient {

    public final String embedModel = "nomic-embed-text";
    public final String genModel = "gemma4:e2b";

    private final String baseUrl;
    private final HttpClient http;
    private final Gson gson = new Gson();

    public OllamaClient(String host, int port) {
        this.baseUrl = "http://" + host + ":" + port;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    public OllamaClient() {
        this("localhost", 11434);
    }

    // ── Availability check ───────────────────────────────────────────────

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/tags"))
                    .timeout(Duration.ofSeconds(3))
                    .GET()
                    .build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // ── Embedding ────────────────────────────────────────────────────────

    /**
     * Generates a text embedding via Ollama's nomic-embed-text model.
     * Returns an empty array if Ollama is unavailable or an error occurs.
     */
    public float[] embed(String text) {
        try {
            JsonObject body = new JsonObject();
            body.addProperty("model", embedModel);
            body.addProperty("prompt", text);

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(120))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                    .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200)
                return new float[0];

            JsonObject json = JsonParser.parseString(resp.body()).getAsJsonObject();
            if (!json.has("embedding"))
                return new float[0];

            JsonArray arr = json.getAsJsonArray("embedding");
            float[] result = new float[arr.size()];
            for (int i = 0; i < arr.size(); i++) {
                result[i] = arr.get(i).getAsFloat();
            }
            return result;

        } catch (Exception e) {
            return new float[0];
        }
    }

    // ── Generation ───────────────────────────────────────────────────────

    /**
     * Generates text via Ollama's llama3.2 model (non-streaming).
     * Returns an error string if Ollama is unavailable.
     */
    public String generate(String prompt) {
        try {
            JsonObject body = new JsonObject();
            body.addProperty("model", genModel);
            body.addProperty("prompt", prompt);
            body.addProperty("stream", false);

            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/generate"))
                    .timeout(Duration.ofSeconds(120))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(body)))
                    .build();

            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() != 200)
                return "ERROR: Ollama unavailable. Run: ollama serve";

            JsonObject json = JsonParser.parseString(resp.body()).getAsJsonObject();
            return json.has("response") ? json.get("response").getAsString()
                    : "ERROR: no response field";

        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve (" + e.getMessage() + ")";
        }
    }

    // ── Text chunker ─────────────────────────────────────────────────────

    /**
     * Splits text into overlapping chunks of chunkWords words with overlapWords
     * overlap.
     * Matches the C++ chunkText() logic exactly.
     */
    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0)
            return List.of();
        if (words.length <= chunkWords)
            return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i)
                    sb.append(' ');
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length)
                break;
        }
        return chunks;
    }
}
