import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In dev, Vite serves the page and forwards the API to the Python server. The Origin and Host are rewritten to the
// target so the server's Host/Origin allow-list never has to be widened.
const target = "http://127.0.0.1:8765";
const forward = { target, changeOrigin: true, headers: { origin: target } };

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": forward, "/dashboard": forward, "/metrics.json": forward } },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/setup.ts"] },
});
