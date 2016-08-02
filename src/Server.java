import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.*;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import edu.stanford.nlp.simple.*;
import com.google.gson.Gson;

public class Server {

    public static void main(String[] args) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);
        server.createContext("/doc", new DocumentHandler());
        server.createContext("/sent", new SentenceHandler());
        server.createContext("/dep", new DependencyHandler());
        server.createContext("/const", new ConstituencyHandler());
        server.setExecutor(null); // creates a default executor
        server.start();
    }

    private static class DocumentHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String in = get(t);
            Document doc = new Document(in);
            List<String> sentenceStrings = new ArrayList<>();
            for (Sentence sentence : doc.sentences()) {
                sentenceStrings.add(sentence.toString());
            }
            String out = new Gson().toJson(sentenceStrings);
            send(t, out);
        }
    }

    private static class SentenceHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String in = get(t);
            Sentence sent = new Sentence(in);
            List<String> words = new ArrayList<>();
            for (String word : sent.words()) {
                words.add(word);
            }
            String out = new Gson().toJson(words);
            send(t, out);
        }
    }

    private static class ConstituencyHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String in = get(t);
            Sentence sent = new Sentence(in);
            String out = sent.parse().toString();
            send(t, out);
        }
    }

    private static class DependencyHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String in = get(t);
            Sentence sent = new Sentence(in);
            List<List<Object>> deps = new ArrayList<>();
            for (int i = 0; i < sent.length(); i++) {
                String word = sent.word(i);
                if (sent.governor(i).isPresent() && sent.incomingDependencyLabel(i).isPresent()) {
                    int governor = sent.governor(i).get();
                    String label = sent.incomingDependencyLabel(i).get();
                    List<Object> curDeps = new ArrayList<>();
                    curDeps.add(word);
                    curDeps.add(governor);
                    curDeps.add(label);
                    deps.add(curDeps);
                }
                else {
                    System.out.println("error");
                    send(t, "error");
                }
            }
            String out = new Gson().toJson(deps);
            send(t, out);
        }
    }

    private static String get(HttpExchange t) {
        return convertStreamToString(t.getRequestBody());
    }

    private static void send(HttpExchange t, String response) throws IOException {
        byte[] byteResponse = response.getBytes("UTF-8");
        t.sendResponseHeaders(200, byteResponse.length);
        OutputStream os = t.getResponseBody();
        os.write(byteResponse);
        os.close();
    }

    private static String convertStreamToString(java.io.InputStream is) {
        java.util.Scanner s = new java.util.Scanner(is).useDelimiter("\\A");
        return s.hasNext() ? s.next() : "";
    }

}
