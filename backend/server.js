require("dotenv").config();
const express = require("express");
const cors = require("cors");
const http = require("http");
const { Server } = require("socket.io");
const axios = require("axios");
const mysql = require("mysql2");
const jwt = require("jsonwebtoken");

// --- Basic setup ---
const app = express();
app.use(cors());
// Increase JSON payload limit for image uploads (base64 encoded images can be large)
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// --- Routes (make sure these paths exist) ---
const authRoutes = require("./routes/auth");
const spotsRoutes = require("./routes/spots");
const bookingsRoutes = require("./routes/bookings");
const systemRoutes = require("./routes/system");
const reviewsRouter = require("./routes/reviews");
const slotsRoutes = require("./routes/slots");
const aiRoutes = require("./routes/ai");
const { runBookingMaintenanceJobs } = require("./scheduledBookingsJob");

app.use("/api/auth", authRoutes);
app.use("/api/spots", spotsRoutes);
app.use("/api/bookings", bookingsRoutes);
app.use("/api/system", systemRoutes);
app.use("/reviews", reviewsRouter);
app.use("/api/slots", slotsRoutes);
app.use("/api/ai", aiRoutes);

app.get("/", (req, res) => res.send("Parking System API is running"));

// --- Database (mysql2 pool) ---
const db = mysql.createPool({
  host: process.env.DB_HOST || "localhost",
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "",
  database: process.env.DB_NAME || "parking_db",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

// Export db for other modules
module.exports.db = db;

// --- Active sessions store (shared) ---
const activeSessions = require("./activeSessions");

// --- HTTP + Socket.IO server creation ---
const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"],
    credentials: true,
  },
  maxHttpBufferSize: 10e6, // 10 MB - adjust if needed
  transports: ["websocket", "polling"],
});

// --- Socket.IO Authentication Middleware ---
io.use((socket, next) => {
  const token = socket.handshake.auth.token;

  if (!token) {
    return next(new Error("Authentication error: No token provided"));
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    socket.user = decoded;
    next();
  } catch (err) {
    next(new Error("Authentication error: Invalid token"));
  }
});

// --- Helper: update DB after detection ---
function updateSlotInDatabase(spot_id, slot_id, status, confidence) {
  const is_available = status === "vacant" ? 1 : 0;

  db.query(
    "UPDATE parking_slots SET is_available = ? WHERE id = ?",
    [is_available, slot_id],
    (err) => {
      if (err) {
        console.error("❌ DB update error:", err);
        return;
      }

      // Update spot statistics
      db.query(
        `UPDATE parking_spots 
         SET available_slots = (
           SELECT COUNT(*) FROM parking_slots 
           WHERE parking_spot_id = ? AND is_active = 1 AND is_available = 1
         ),
         occupied_slots = (
           SELECT COUNT(*) FROM parking_slots 
           WHERE parking_spot_id = ? AND is_active = 1 AND is_available = 0
         )
         WHERE id = ?`,
        [spot_id, spot_id, spot_id],
        (err2) => {
          if (err2) console.error("⚠️ Stats update error:", err2);
        },
      );

      // Log detection
      db.query(
        `INSERT INTO ai_detection_logs 
         (parking_spot_id, slot_id, detection_type, confidence, created_at) 
         VALUES (?, ?, ?, ?, NOW())`,
        [spot_id, slot_id, status, confidence],
        (err3) => {
          if (err3) console.warn("⚠️ Log error:", err3);
        },
      );
    },
  );
}

// --- Socket.IO Connection Handling ---
io.on("connection", (socket) => {
  console.log(`🔌 Client connected: ${socket.id} (User: ${socket.user.id})`);

  // Join room for specific parking spot
  socket.on("join_spot", (spotId) => {
    socket.join(`spot_${spotId}`);
    console.log(`📍 User ${socket.user.id} joined spot ${spotId} room`);
  });

  // Leave room
  socket.on("leave_spot", (spotId) => {
    socket.leave(`spot_${spotId}`);
    console.log(`📍 User ${socket.user.id} left spot ${spotId} room`);
  });

  // Handle disconnect
  socket.on("disconnect", () => {
    console.log(`🔌 Client disconnected: ${socket.id}`);
  });
});

// --- Set Socket.IO instance for AI routes ---
aiRoutes.setSocketIO(io);

// --- Exports (if other modules should interact) ---
module.exports = {
  activeSessions,
  io,
  db,
  app,
  server,
};

// --- Start server ---
const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log("=".repeat(60));
  console.log(`🚀 Backend server running on port ${PORT}`);
  console.log(`🔌 Socket.IO ready for real-time updates`);
  console.log(`🎥 Camera worker system enabled`);
  console.log("=".repeat(60));

  // Start background job for scheduled bookings
  console.log("⏰ Starting scheduled bookings maintenance job...");
  runBookingMaintenanceJobs(); // Run immediately on startup
  setInterval(runBookingMaintenanceJobs, 60000); // Run every minute
  console.log("✅ Background job started - checking every 60 seconds");
});
