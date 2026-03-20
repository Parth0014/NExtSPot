const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  // Proxy Socket.IO connections to AI server (port 5001)
  app.use(
    "/socket.io",
    createProxyMiddleware({
      target: "http://localhost:5001",
      changeOrigin: true,
      ws: true, // Enable WebSocket proxying
      logLevel: "debug",
      onProxyReq: (proxyReq, req, res) => {
        console.log(`[Socket.IO Proxy] ${req.method} ${req.url}`);
      },
      onProxyReqWs: (proxyReq, req, socket, options, head) => {
        console.log(`[Socket.IO WebSocket Proxy] ${req.url}`);
      },
      onError: (err, req, res) => {
        console.error("Proxy error:", err.message);
        if (res.writeHead) {
          res.writeHead(500, {
            "Content-Type": "application/json",
          });
          res.end(
            JSON.stringify({ error: "Proxy error", details: err.message })
          );
        }
      },
    })
  );

  // Proxy AI API endpoints to AI server (port 5001)
  app.use(
    "/api/ai",
    createProxyMiddleware({
      target: "http://localhost:5001",
      changeOrigin: true,
      logLevel: "debug",
      onProxyReq: (proxyReq, req, res) => {
        console.log(`[AI API Proxy] ${req.method} ${req.url}`);
      },
    })
  );
};
