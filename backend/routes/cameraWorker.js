const axios = require("axios");
const MjpegConsumer = require("mjpeg-consumer");
const { Buffer } = require("buffer");
const http = require("http");
const https = require("https");

// Map: spotId -> { camera, stopFn }
const cameraWorkers = new Map();

// 🚀 OPTIMIZATION: Frame skip settings
const FRAME_SKIP = 1; // Process every Nth frame (1 = all frames, 2 = every other)
const TARGET_FPS = 30; // Target frames per second
const MIN_FRAME_INTERVAL = 1000 / TARGET_FPS; // Minimum ms between processed frames

/**
 * Start camera worker for a parking spot
 */
function startCameraWorker({ spotId, cameraUrl, pythonServer, io }) {
  if (cameraWorkers.has(spotId)) {
    console.log(`⚠️ Camera worker already running for spot ${spotId}`);
    return;
  }

  console.log(`🎥 Starting camera worker for spot ${spotId}`);
  console.log(`📹 Camera URL: ${cameraUrl}`);

  let running = true;
  let consecutiveErrors = 0;
  const MAX_RETRIES = 50;
  let frameCount = 0;
  let processedFrameCount = 0;
  let lastFpsTime = Date.now();
  let lastProcessTime = 0;
  let currentFps = 0;
  let request = null;
  let isProcessing = false; // 🚀 Prevent overlapping requests

  // Create MJPEG consumer
  const camera = new MjpegConsumer();

  // Function to connect to camera stream
  const connectCamera = () => {
    try {
      const client = cameraUrl.startsWith("https") ? https : http;

      request = client.get(cameraUrl, (response) => {
        console.log(`✅ Connected to camera stream [spot ${spotId}]`);
        consecutiveErrors = 0; // Reset errors on successful connection
        response.pipe(camera);
      });

      request.on("error", (err) => {
        console.error(
          `❌ Camera connection error [spot ${spotId}]:`,
          err.message,
        );
        camera.emit("error", err);
      });
    } catch (err) {
      console.error(`❌ Camera connect failed [spot ${spotId}]:`, err.message);
      camera.emit("error", err);
    }
  };

  camera.on("data", async (frameBuffer) => {
    if (!running) return;

    frameCount++;
    const now = Date.now();

    // 🚀 OPTIMIZATION: Skip frames if processing too slow or too frequent
    if (isProcessing) {
      return; // Skip if still processing previous frame
    }

    // Rate limiting - ensure minimum interval between processed frames
    if (now - lastProcessTime < MIN_FRAME_INTERVAL) {
      return;
    }

    try {
      isProcessing = true;
      lastProcessTime = now;

      // Convert buffer to base64
      const base64Frame = frameBuffer.toString("base64");

      // Send to Python AI server with shorter timeout
      const response = await axios.post(
        `${pythonServer}/process-frame`,
        {
          spot_id: spotId,
          frame: base64Frame,
          timestamp: now,
        },
        { timeout: 3000 }, // Reduced timeout for faster failure
      );

      isProcessing = false;
      processedFrameCount++;

      // Broadcast processed frame with annotations
      if (response.data.processed_frame) {
        io.to(`spot_${spotId}`).emit("processed_frame", {
          spot_id: spotId,
          frame: response.data.processed_frame,
          timestamp: response.data.timestamp,
        });
      }

      // Broadcast occupancy data
      if (response.data.occupancy) {
        io.to(`spot_${spotId}`).emit("occupancy_update", {
          spot_id: spotId,
          occupancy: response.data.occupancy,
          num_slots: response.data.num_slots,
        });
      }

      // If state changed, broadcast that too
      if (response.data.state_change) {
        io.to(`spot_${spotId}`).emit("state_change", {
          spot_id: spotId,
          change: response.data.state_change,
        });
      }

      // Calculate FPS (based on processed frames)
      if (now - lastFpsTime >= 1000) {
        currentFps = processedFrameCount;
        processedFrameCount = 0;
        lastFpsTime = now;

        // Broadcast FPS
        io.to(`spot_${spotId}`).emit("fps_update", {
          spot_id: spotId,
          fps: currentFps,
        });
      }

      // Reset error counter on success
      consecutiveErrors = 0;
    } catch (err) {
      isProcessing = false;
      consecutiveErrors++;
      console.error(`❌ Frame processing error [spot ${spotId}]:`, err.message);

      if (consecutiveErrors > MAX_RETRIES) {
        console.error(`🛑 Too many errors for spot ${spotId}, stopping worker`);
        stopCameraWorker(spotId);

        // Notify clients
        io.to(`spot_${spotId}`).emit("camera_error", {
          spot_id: spotId,
          message: "Camera connection failed after multiple retries",
        });
      }
    }
  });

  camera.on("error", (err) => {
    console.error(`❌ Camera stream error [spot ${spotId}]:`, err.message);
    consecutiveErrors++;

    // Close existing request if any
    if (request) {
      request.destroy();
      request = null;
    }

    if (consecutiveErrors > MAX_RETRIES) {
      console.error(`🛑 Too many errors for spot ${spotId}, stopping worker`);
      stopCameraWorker(spotId);

      // Notify clients
      io.to(`spot_${spotId}`).emit("camera_error", {
        spot_id: spotId,
        message: "Camera stream error",
      });
    } else {
      // Exponential backoff reconnect
      const delay = Math.min(5000, 100 * consecutiveErrors);
      setTimeout(() => {
        if (running) {
          console.log(
            `🔄 Reconnecting to camera [spot ${spotId}] after ${delay}ms...`,
          );
          connectCamera();
        }
      }, delay);
    }
  });

  // Initial connection
  connectCamera();

  cameraWorkers.set(spotId, {
    camera,
    request,
    stopFn: () => {
      console.log(`⏹️ Stopping camera worker for spot ${spotId}`);
      running = false;
      camera.removeAllListeners();
      if (request) {
        request.destroy();
      }
    },
  });

  console.log(`✅ Camera worker started for spot ${spotId}`);
}

function stopCameraWorker(spotId) {
  const worker = cameraWorkers.get(spotId);
  if (worker) {
    worker.stopFn();
    cameraWorkers.delete(spotId);
    console.log(`✅ Camera worker stopped for spot ${spotId}`);
  }
}

function getWorkerStatus(spotId) {
  return cameraWorkers.has(spotId);
}

module.exports = {
  startCameraWorker,
  stopCameraWorker,
  getWorkerStatus,
};
