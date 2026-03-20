const express = require("express");
const router = express.Router();
const db = require("../config/db");
const auth = require("../middleware/authMiddleware");
const axios = require("axios");
const MjpegProxy = require("mjpeg-proxy").MjpegProxy;

const activeSessions = require("../activeSessions");
const {
  startCameraWorker,
  stopCameraWorker,
  getWorkerStatus,
} = require("./cameraWorker");

const PYTHON_AI_SERVER =
  process.env.PYTHON_AI_SERVER || "http://localhost:5001";

// Store io instance (set by server.js)
let io = null;
function setSocketIO(socketIO) {
  io = socketIO;
}

/**
 * Health check for Python AI server
 */
router.get("/python-health", auth(["owner"]), async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_AI_SERVER}/health`, {
      timeout: 5000,
    });
    res.json({
      success: true,
      python_server: response.data,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Python AI server not reachable",
      error: error.message,
    });
  }
});

/**
 * Get saved AI camera configuration for a parking spot
 */
router.get("/config/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;

  db.query(
    `SELECT * FROM ai_camera_config WHERE parking_spot_id = ?`,
    [spot_id],
    (err, rows) => {
      if (err) {
        return res.status(500).json({ error: err.message });
      }

      if (!rows || rows.length === 0) {
        return res.json({
          success: true,
          config: null,
          message: "No configuration found",
        });
      }

      const config = rows[0];
      res.json({
        success: true,
        config: {
          mode: config.mode,
          camera_url: config.camera_url,
          grid_config: config.grid_config,
          scan_interval: config.scan_interval,
          last_calibrated: config.last_calibrated,
        },
      });
    },
  );
});

/**
 * Save Camera URL
 */
router.post("/save-camera-url/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;
  const { camera_url } = req.body;

  console.log(`💾 Saving camera URL for spot ${spot_id}:`, camera_url);

  if (!camera_url) {
    return res.status(400).json({ message: "Camera URL is required" });
  }

  db.query(
    `INSERT INTO ai_camera_config (parking_spot_id, camera_url)
     VALUES (?, ?)
     ON DUPLICATE KEY UPDATE camera_url = ?`,
    [spot_id, camera_url, camera_url],
    (err) => {
      if (err) {
        console.error(`❌ Error saving camera URL:`, err.message);
        return res.status(500).json({ error: err.message });
      }
      console.log(`✅ Camera URL saved successfully for spot ${spot_id}`);
      res.json({ success: true, message: "Camera URL saved", camera_url });
    },
  );
});

/**
 * Get previously used camera URLs for a parking spot
 */
router.get("/previous-urls/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;

  db.query(
    `SELECT DISTINCT camera_url FROM ai_camera_config 
     WHERE parking_spot_id = ? AND camera_url IS NOT NULL AND camera_url != ''
     LIMIT 10`,
    [spot_id],
    (err, rows) => {
      if (err) {
        console.error(`❌ Error fetching previous URLs:`, err.message);
        return res.status(500).json({ success: false, error: err.message });
      }

      const urls = rows
        ? rows.map((row) => row.camera_url).filter(Boolean)
        : [];
      console.log(
        `📚 Found ${urls.length} previous camera URLs for spot ${spot_id}:`,
        urls,
      );

      res.json({
        success: true,
        urls: urls,
        message: `Found ${urls.length} previous URLs`,
      });
    },
  );
});

/**
 * Toggle AI Mode
 */
router.post("/toggle-mode/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;
  const { mode } = req.body;

  db.query(
    "SELECT owner_id, occupied_slots FROM parking_spots WHERE id = ?",
    [spot_id],
    (err, result) => {
      if (err) return res.status(500).json({ error: err.message });
      if (!result[0])
        return res.status(404).json({ message: "Spot not found" });
      if (result[0].owner_id !== req.user.id) {
        return res.status(403).json({ message: "Not authorized" });
      }

      if (mode === "ai" && result[0].occupied_slots > 0) {
        return res.status(400).json({
          message: "Cannot switch to AI mode while slots are occupied",
        });
      }

      db.query(
        `INSERT INTO ai_camera_config (parking_spot_id, mode) VALUES (?, ?)
         ON DUPLICATE KEY UPDATE mode = ?`,
        [spot_id, mode, mode],
        (err2) => {
          if (err2) return res.status(500).json({ error: err2.message });
          res.json({ message: `Switched to ${mode} mode`, mode });
        },
      );
    },
  );
});

/**
 * Save Grid Configuration
 */
router.post("/save-grid-config/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;
  const { grid_config } = req.body;

  console.log(`💾 Saving grid config for spot ${spot_id}`);
  console.log(`📹 Camera URL from grid_config:`, grid_config?.camera_url);

  if (!grid_config || !Array.isArray(grid_config.cells)) {
    return res.status(400).json({ message: "Invalid grid_config" });
  }

  const gridJson = JSON.stringify(grid_config);
  const cameraUrl = grid_config.camera_url || null;

  console.log(`🔗 Camera URL to save:`, cameraUrl);

  db.query(
    `INSERT INTO ai_camera_config (parking_spot_id, grid_config, last_calibrated, camera_url)
     VALUES (?, ?, NOW(), ?)
     ON DUPLICATE KEY UPDATE grid_config = ?, last_calibrated = NOW(), camera_url = ?`,
    [spot_id, gridJson, cameraUrl, gridJson, cameraUrl],
    (err) => {
      if (err) {
        console.error(`❌ Error saving grid config:`, err.message);
        return res.status(500).json({ error: err.message });
      }
      console.log(
        `✅ Grid configuration saved successfully for spot ${spot_id}`,
      );
      res.json({ message: "Grid configuration saved" });
    },
  );
});

/**
 * Clear Grid Config - Delete grid configuration from database
 */
router.delete("/clear-grid-config/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;

  console.log(`🗑️ Clearing grid config for spot ${spot_id}`);

  db.query(
    `UPDATE ai_camera_config SET grid_config = NULL WHERE parking_spot_id = ?`,
    [spot_id],
    (err, result) => {
      if (err) {
        console.error(`❌ Error clearing grid config:`, err.message);
        return res.status(500).json({ error: err.message });
      }
      console.log(`✅ Grid configuration cleared for spot ${spot_id}`);
      res.json({ message: "Grid configuration cleared" });
    },
  );
});

/**
 * Detect Grid - Auto-detect parking grid from frame
 */
router.post("/detect-grid", auth(["owner"]), async (req, res) => {
  const { frame, aoi } = req.body;

  if (!frame) {
    return res.status(400).json({
      success: false,
      message: "Missing frame data",
    });
  }

  try {
    const response = await axios.post(
      `${PYTHON_AI_SERVER}/detect-grid`,
      { frame, aoi },
      { timeout: 30000 },
    );

    if (response.data.success) {
      res.json({
        success: true,
        num_cells: response.data.num_cells,
        cells: response.data.cells,
        annotated_frame: response.data.annotated_frame,
      });
    } else {
      res.status(500).json({
        success: false,
        message: response.data.error || "Grid detection failed",
      });
    }
  } catch (error) {
    console.error("❌ Grid detection error:", error.message);
    res.status(500).json({
      success: false,
      message: "Cannot connect to Python AI server for grid detection",
      error: error.message,
    });
  }
});

/**
 * Start AI Detection - Backend-owned camera
 */
router.post("/start-detection", auth(["owner"]), async (req, res) => {
  const { parking_spot_id, grid_config } = req.body;

  if (!parking_spot_id) {
    return res.status(400).json({
      success: false,
      message: "Missing parking_spot_id",
    });
  }

  if (!grid_config || !grid_config.camera_url) {
    return res.status(400).json({
      success: false,
      message: "Missing grid_config or camera_url",
    });
  }

  try {
    // Fetch slot mapping from DB
    const slotsQuery = `SELECT id as slot_id, slot_number FROM parking_slots 
                        WHERE parking_spot_id = ? AND is_active = 1 
                        ORDER BY slot_number`;

    db.query(slotsQuery, [parking_spot_id], async (err, slots) => {
      if (err) {
        return res.status(500).json({ success: false, message: err.message });
      }

      const slot_mapping = {};
      slots.forEach((slot) => {
        slot_mapping[String(slot.slot_number)] = slot.slot_id;
      });

      // Initialize Python AI server session
      try {
        const response = await axios.post(
          `${PYTHON_AI_SERVER}/start-detection`,
          {
            parking_spot_id,
            grid_config,
            slot_mapping,
          },
          { timeout: 10000 },
        );

        if (response.data.success) {
          // Store session in Node.js
          activeSessions.set(parking_spot_id, {
            grid_config,
            slot_mapping,
            started_at: Date.now(),
            last_states: {},
            num_slots: response.data.num_slots,
          });

          // 🚀 START CAMERA WORKER (Backend owns camera now)
          if (!io) {
            return res.status(500).json({
              success: false,
              message: "Socket.IO not initialized",
            });
          }

          startCameraWorker({
            spotId: parking_spot_id,
            cameraUrl: grid_config.camera_url,
            pythonServer: PYTHON_AI_SERVER,
            io: io,
          });

          console.log(
            `✅ Detection started for spot ${parking_spot_id} with ${response.data.num_slots} slots`,
          );

          res.json({
            success: true,
            message: "Detection started",
            session_id: parking_spot_id,
            num_slots: response.data.num_slots,
          });
        } else {
          res.status(500).json({
            success: false,
            message: response.data.message || "Python AI failed",
          });
        }
      } catch (pythonError) {
        console.error("❌ Python AI error:", pythonError.message);
        res.status(500).json({
          success: false,
          message: "Cannot connect to Python AI server",
          error: pythonError.message,
        });
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});

/**
 * Stop AI Detection
 */
router.post("/stop-detection", auth(["owner"]), async (req, res) => {
  const { parking_spot_id } = req.body;

  if (!parking_spot_id) {
    console.error("❌ Missing parking_spot_id in stop-detection request");
    return res.status(400).json({
      success: false,
      message: "Missing parking_spot_id",
    });
  }

  try {
    // Stop Python AI session
    try {
      await axios.post(
        `${PYTHON_AI_SERVER}/stop-detection`,
        { parking_spot_id },
        { timeout: 5000 },
      );
      console.log(`✅ Python AI stopped for spot ${parking_spot_id}`);
    } catch (pythonError) {
      console.warn(
        `⚠️ Python AI stop failed (non-critical):`,
        pythonError.message,
      );
    }

    // Stop camera worker
    stopCameraWorker(parking_spot_id);

    // Remove session
    activeSessions.delete(parking_spot_id);

    console.log(`⏹️ Detection stopped for spot ${parking_spot_id}`);

    res.json({
      success: true,
      message: "Detection stopped",
    });
  } catch (error) {
    console.error("❌ Stop detection error:", error);
    res.status(500).json({
      success: false,
      message: error.message,
    });
  }
});

/**
 * Get AI Status (for auto-resume)
 */
router.get("/status/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;
  const session = activeSessions.get(parseInt(spot_id));
  const isRunning = getWorkerStatus(parseInt(spot_id));

  res.json({
    is_running: isRunning && !!session,
    started_at: session?.started_at || null,
    num_slots: session?.num_slots || 0,
    has_worker: isRunning,
  });
});

/**
 * MJPEG Stream Proxy - Allows React to view live camera feed
 */
router.get("/stream/:spot_id", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;
  const session = activeSessions.get(parseInt(spot_id));

  console.log(`📹 Stream request for spot ${spot_id}`);
  console.log(`🔍 Active session exists: ${!!session}`);

  // Try to get camera URL from active session first
  if (session && session.grid_config && session.grid_config.camera_url) {
    const cameraUrl = session.grid_config.camera_url;
    console.log(`✅ Using camera URL from active session: ${cameraUrl}`);

    // Create MJPEG proxy - it handles headers automatically
    const proxy = new MjpegProxy(cameraUrl);
    proxy.proxyRequest(req, res);

    req.on("close", () => {
      // MjpegProxy doesn't have destroy method - cleanup handled internally
      console.log(`📹 Stream closed for spot ${spot_id}`);
    });
  } else {
    // No active session - fetch camera URL from database
    console.log(`📂 Fetching camera URL from database for spot ${spot_id}`);
    db.query(
      `SELECT camera_url FROM ai_camera_config WHERE parking_spot_id = ?`,
      [spot_id],
      (err, rows) => {
        if (err) {
          console.error(`❌ Database error:`, err.message);
          return res.status(500).json({ error: err.message });
        }

        console.log(`📊 Database query result:`, rows);

        if (!rows || rows.length === 0 || !rows[0].camera_url) {
          console.error(`❌ No camera URL found for spot ${spot_id}`);
          return res.status(404).json({
            error: "Camera URL not configured for this parking spot",
            debug: {
              rows,
              hasRows: rows?.length > 0,
              cameraUrl: rows?.[0]?.camera_url,
            },
          });
        }

        const cameraUrl = rows[0].camera_url;
        console.log(`✅ Using camera URL from database: ${cameraUrl}`);

        // Create MJPEG proxy - it handles headers automatically
        const proxy = new MjpegProxy(cameraUrl);
        proxy.proxyRequest(req, res);

        req.on("close", () => {
          // MjpegProxy doesn't have destroy method - cleanup handled internally
          console.log(`📹 Stream closed for spot ${spot_id}`);
        });
      },
    );
  }
});

module.exports = router;
module.exports.setSocketIO = setSocketIO;
