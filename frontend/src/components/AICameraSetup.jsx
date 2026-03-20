import React, { useState, useRef, useEffect, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Wifi,
  Play,
  Square,
  Edit3,
  Trash2,
  Save,
  Target,
  ArrowLeft,
  Grid3X3,
} from "lucide-react";
import { io } from "socket.io-client";
import API from "../api/api";

function AICameraSetup() {
  // Get spotId from URL params
  const { spotId } = useParams();
  const navigate = useNavigate();

  const [mode, setMode] = useState("manual");
  const [isDetecting, setIsDetecting] = useState(false);
  const [gridConfig, setGridConfig] = useState(null);
  const [message, setMessage] = useState("");
  const [fps, setFps] = useState(0);
  const [occupancyStatus, setOccupancyStatus] = useState({});
  const [processedFrame, setProcessedFrame] = useState(null);
  const [totalSlots, setTotalSlots] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  // Camera URL state
  const [cameraSource, setCameraSource] = useState("ip"); // "ip" or "usb"
  const [showIpModal, setShowIpModal] = useState(false);
  const [ipInputValue, setIpInputValue] = useState("");
  const [cameraUrl, setCameraUrl] = useState("");
  const [previousCameraUrls, setPreviousCameraUrls] = useState([]);

  // Drawing state
  const [isFrozen, setIsFrozen] = useState(false);
  const [frozenImage, setFrozenImage] = useState(null);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const [drawnRectangles, setDrawnRectangles] = useState([]);
  const [currentRect, setCurrentRect] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // AOI state
  const [aoiMode, setAoiMode] = useState(false);
  const [aoiRect, setAoiRect] = useState(null);
  const [isDrawingAOI, setIsDrawingAOI] = useState(false);

  // Auto-detect state
  const [autoDetecting, setAutoDetecting] = useState(false);

  // Grid adjustment state - for independent corner adjustment like document scanner
  const [isAdjustingGrids, setIsAdjustingGrids] = useState(false);
  const [adjustingGridIndex, setAdjustingGridIndex] = useState(-1);
  const [adjustingPointIndex, setAdjustingPointIndex] = useState(-1);
  const [hoveredGridIndex, setHoveredGridIndex] = useState(-1);
  const [hoveredPointIndex, setHoveredPointIndex] = useState(-1);
  const [gridCorners, setGridCorners] = useState({}); // Store independent corners for each grid

  // Refs
  const videoRef = useRef(null);
  const drawingCanvasRef = useRef(null);
  const frozenCanvasRef = useRef(null);
  const socketRef = useRef(null);

  const token = localStorage.getItem("token");
  const BACKEND_SERVER =
    process.env.REACT_APP_BACKEND_SERVER || "http://localhost:5000";

  // Include token as query param for image stream (img tags cannot set Authorization header)
  const STREAM_URL = token
    ? `${BACKEND_SERVER}/api/ai/stream/${spotId}?token=${encodeURIComponent(
        token,
      )}`
    : `${BACKEND_SERVER}/api/ai/stream/${spotId}`;

  // USB Preview URL (works without active detection session)
  const USB_PREVIEW_URL = token
    ? `${BACKEND_SERVER}/api/ai/usb-preview?token=${encodeURIComponent(token)}`
    : `${BACKEND_SERVER}/api/ai/usb-preview`;

  // Determine which stream URL to use based on camera type and detection state
  const getVideoStreamUrl = () => {
    // For USB cameras not in detection mode, use preview stream
    if ((cameraUrl === "usb" || cameraUrl.startsWith("usb:")) && !isDetecting) {
      return USB_PREVIEW_URL;
    }
    // Otherwise use the main stream URL
    return STREAM_URL;
  };

  // Normalize slot data to standard format
  const normalizeSlots = (slots) => {
    if (!Array.isArray(slots)) return [];
    return slots
      .map((slot, idx) => {
        // If slot is already in correct format
        if (slot && slot.bbox && Array.isArray(slot.bbox)) {
          return slot;
        }
        // If slot is just an array (bbox)
        if (Array.isArray(slot) && slot.length >= 4) {
          const [x1, y1, x2, y2] = slot;
          return {
            slot_number: idx + 1,
            bbox: [x1, y1, x2, y2],
            bbox_normalized: [x1 / 1280, y1 / 720, x2 / 1280, y2 / 720],
          };
        }
        // Invalid format
        console.warn(`Invalid slot format at index ${idx}:`, slot);
        return null;
      })
      .filter((slot) => slot !== null);
  };

  // Convert bbox to 4 independent corners (for document scanner-like adjustment)
  const getGridCorners = useCallback(
    (gridIndex) => {
      // Check if there are temporary adjustment corners
      if (gridCorners[gridIndex]) {
        return gridCorners[gridIndex];
      }

      // Check if slot has persisted perspective corners
      const slot = drawnRectangles[gridIndex];
      if (
        slot?.corners &&
        Array.isArray(slot.corners) &&
        slot.corners.length === 4
      ) {
        return slot.corners;
      }

      // Initialize corners from bbox
      let bbox = slot?.bbox || slot;

      if (!Array.isArray(bbox) || bbox.length < 4) {
        return null;
      }

      const [x1, y1, x2, y2] = bbox;
      return [
        { x: x1, y: y1 }, // top-left
        { x: x2, y: y1 }, // top-right
        { x: x2, y: y2 }, // bottom-right
        { x: x1, y: y2 }, // bottom-left
      ];
    },
    [gridCorners, drawnRectangles],
  );

  // Convert 4 corners back to bbox
  const cornersToBox = useCallback((corners) => {
    if (!Array.isArray(corners) || corners.length < 4) return null;

    const xCoords = corners.map((c) => c.x);
    const yCoords = corners.map((c) => c.y);

    const x1 = Math.min(...xCoords);
    const y1 = Math.min(...yCoords);
    const x2 = Math.max(...xCoords);
    const y2 = Math.max(...yCoords);

    return [x1, y1, x2, y2];
  }, []);

  // Perspective transformation: store corner points for non-rectangular grids
  const getPerspectiveCorners = useCallback(
    (gridIndex) => {
      if (gridCorners[gridIndex]) {
        return gridCorners[gridIndex];
      }

      // Initialize from bbox
      const slot = drawnRectangles[gridIndex];
      let bbox = slot?.bbox || slot;

      if (!Array.isArray(bbox) || bbox.length < 4) {
        return null;
      }

      const [x1, y1, x2, y2] = bbox;
      return [
        { x: x1, y: y1 }, // top-left
        { x: x2, y: y1 }, // top-right
        { x: x2, y: y2 }, // bottom-right
        { x: x1, y: y2 }, // bottom-left
      ];
    },
    [gridCorners, drawnRectangles],
  );

  // Draw perspective quadrilateral filled (for rendering distorted grids)
  const drawPerspectiveQuad = useCallback(
    (ctx, corners, color, label, index) => {
      if (!corners || corners.length < 4) return;

      // Draw the quadrilateral
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      ctx.lineTo(corners[1].x, corners[1].y);
      ctx.lineTo(corners[2].x, corners[2].y);
      ctx.lineTo(corners[3].x, corners[3].y);
      ctx.closePath();
      ctx.stroke();

      // Draw label at center
      const centerX =
        (corners[0].x + corners[1].x + corners[2].x + corners[3].x) / 4;
      const centerY =
        (corners[0].y + corners[1].y + corners[2].y + corners[3].y) / 4;

      ctx.fillStyle = color;
      ctx.font = "bold 24px Arial";
      ctx.fillText(label, centerX - 12, centerY + 8);
    },
    [],
  );

  // Determine which grid point is closest to clicked position
  const getGridPointAtCoords = (x, y, threshold = 15) => {
    const safeRects = drawnRectangles || [];
    for (let i = 0; i < safeRects.length; i++) {
      const corners = getGridCorners(i);
      if (!corners) continue;

      for (let pointIdx = 0; pointIdx < corners.length; pointIdx++) {
        const corner = corners[pointIdx];
        const distance = Math.sqrt(
          Math.pow(corner.x - x, 2) + Math.pow(corner.y - y, 2),
        );
        if (distance <= threshold) {
          return { gridIndex: i, pointIndex: pointIdx };
        }
      }
    }
    return null;
  };

  // Load saved configuration on mount
  useEffect(() => {
    const loadSavedConfig = async () => {
      setIsLoading(true);
      try {
        // Load AI camera config
        const configRes = await fetch(
          `${BACKEND_SERVER}/api/ai/config/${spotId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          },
        );
        const configData = await configRes.json();

        if (configData.success && configData.config) {
          const cfg = configData.config;

          // ⚠️ DO NOT restore camera URL - IP addresses change frequently
          // User must manually configure camera URL each session
          // if (cfg.camera_url) {
          //   setCameraUrl(cfg.camera_url);
          //   console.log("📹 Camera URL restored:", cfg.camera_url);
          // }

          // Restore mode
          if (cfg.mode) {
            setMode(cfg.mode);
          }

          // Restore grid config and drawn rectangles
          if (cfg.grid_config && cfg.grid_config.cells) {
            setGridConfig(cfg.grid_config);
            setDrawnRectangles(cfg.grid_config.cells);
            console.log(
              "📐 Grid configuration restored:",
              cfg.grid_config.cells.length,
              "slots",
            );

            // Restore AOI if present
            if (cfg.grid_config.aoi) {
              const aoi = cfg.grid_config.aoi;
              if (aoi.bbox) {
                setAoiRect({
                  x1: aoi.bbox[0],
                  y1: aoi.bbox[1],
                  x2: aoi.bbox[2],
                  y2: aoi.bbox[3],
                  x1_norm: aoi.bbox_normalized?.[0],
                  y1_norm: aoi.bbox_normalized?.[1],
                  x2_norm: aoi.bbox_normalized?.[2],
                  y2_norm: aoi.bbox_normalized?.[3],
                });
              }
            }
          }

          setMessage("✅ Configuration loaded");
        }

        // Load total slots count
        const slotsRes = await fetch(
          `${BACKEND_SERVER}/api/slots/parking-spot/${spotId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          },
        );
        const slotsData = await slotsRes.json();
        if (slotsData && Array.isArray(slotsData)) {
          setTotalSlots(slotsData.length);
        }

        // Check if detection is running
        const statusRes = await fetch(
          `${BACKEND_SERVER}/api/ai/status/${spotId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          },
        );
        const statusData = await statusRes.json();
        if (statusData?.is_running) {
          setIsDetecting(true);
          setMessage("🔄 Detection is active");
        }
      } catch (error) {
        console.error("Failed to load config:", error);
        setMessage("⚠️ Could not load saved configuration");
      } finally {
        setIsLoading(false);
      }
    };

    if (spotId) {
      loadSavedConfig();
    }
  }, [spotId, token, BACKEND_SERVER]);

  // Socket connection
  useEffect(() => {
    // Always create a fresh socket connection when component mounts
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    socketRef.current = io(BACKEND_SERVER, {
      transports: ["websocket", "polling"],
      auth: { token },
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
    });

    socketRef.current.on("connect", () => {
      console.log("🟢 Socket connected");
      // Join room for this parking spot
      socketRef.current.emit("join_spot", spotId);
    });

    socketRef.current.on("disconnect", () => {
      console.log("🔴 Socket disconnected");
    });

    socketRef.current.on("error", (data) => {
      setMessage(`❌ ${data.message}`);
    });

    socketRef.current.on("occupancy_update", (data) => {
      if (data.spot_id == spotId) {
        setOccupancyStatus(data.occupancy || {});
      }
    });

    socketRef.current.on("state_change", (data) => {
      if (data.spot_id == spotId && data.change) {
        setMessage(
          `🔄 Slot ${data.change.slot_number}: ${data.change.old_status} → ${data.change.new_status}`,
        );
      }
    });

    socketRef.current.on("fps_update", (data) => {
      if (data.spot_id == spotId) {
        setFps(data.fps || 0);
      }
    });

    socketRef.current.on("camera_error", (data) => {
      if (data.spot_id == spotId) {
        setMessage(`❌ Camera Error: ${data.message}`);
        setIsDetecting(false);
      }
    });

    socketRef.current.on("processed_frame", (data) => {
      if (data.spot_id == spotId && data.frame) {
        setProcessedFrame(data.frame);
      }
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.emit("leave_spot", spotId);
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, [spotId, token, BACKEND_SERVER]);

  // Handle back navigation
  const handleClose = useCallback(() => {
    navigate(-1);
  }, [navigate]);

  // --- IP CAMERA SETUP ---

  const startIPWebcam = async () => {
    setIpInputValue("");
    setShowIpModal(true);

    // Fetch previous camera URLs from database
    try {
      console.log(`🔍 Fetching previous URLs for spot ${spotId}...`);
      const response = await fetch(
        `${BACKEND_SERVER}/api/ai/previous-urls/${spotId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      const data = await response.json();
      console.log("📡 API Response:", data);

      if (data.success && Array.isArray(data.urls)) {
        console.log(`✅ Loaded ${data.urls.length} previous URLs:`, data.urls);
        setPreviousCameraUrls(data.urls);
      } else {
        console.warn("⚠️ No URLs in response or invalid format:", data);
        setPreviousCameraUrls([]);
      }
    } catch (error) {
      console.error("❌ Failed to fetch previous URLs:", error);
      setPreviousCameraUrls([]);
    }
  };

  const selectPreviousUrl = (url) => {
    setIpInputValue(url);
  };

  const handleIpSubmit = async () => {
    let url = ipInputValue.trim();
    if (!url) {
      setMessage("⚠️ Please enter a valid IP address");
      return;
    }

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      url = "http://" + url;
    }

    url = url.replace(/\/+$/, "");

    if (!/\/video(?:\b|$)/i.test(url)) {
      url = url + "/video";
    }

    console.log("📹 Camera URL configured:", url);

    // Save to database immediately
    try {
      const response = await fetch(
        `${BACKEND_SERVER}/api/ai/save-camera-url/${spotId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ camera_url: url }),
        },
      );

      const data = await response.json();

      if (response.ok && data.success) {
        setCameraUrl(url);
        setShowIpModal(false);
        setMessage("✅ Camera URL saved to database");
        console.log("✅ Camera URL saved to database:", url);
      } else {
        setMessage(
          "⚠️ Failed to save camera URL: " + (data.message || "Unknown error"),
        );
        console.error("Failed to save camera URL:", data);
      }
    } catch (error) {
      setMessage("❌ Error saving camera URL: " + error.message);
      console.error("Error saving camera URL:", error);
    }
  };

  // --- FREEZE FRAME ---

  const freezeFrame = () => {
    const img = videoRef.current;
    if (!img || !img.complete || !img.naturalWidth) {
      setMessage("⚠️ Wait for video to load");
      return;
    }

    const w = img.naturalWidth;
    const h = img.naturalHeight;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = w;
    tempCanvas.height = h;
    const tempCtx = tempCanvas.getContext("2d");

    try {
      tempCtx.drawImage(img, 0, 0, w, h);
      const frozenDataUrl = tempCanvas.toDataURL("image/jpeg", 0.95);

      setFrozenImage(frozenDataUrl);
      setIsFrozen(true);
      setMessage(`📸 Frame frozen (${w}x${h})`);
    } catch (err) {
      setMessage("Failed to freeze frame: " + err.message);
    }
  };

  const unfreezeFrame = () => {
    setIsFrozen(false);
    setFrozenImage(null);
    setIsDrawingMode(false);
    setAoiMode(false);
    setMessage("▶️ Live feed resumed");
  };

  // --- AOI DRAWING ---

  const startDrawingAOI = () => {
    if (!isFrozen) {
      setMessage("⚠️ Please freeze the frame first");
      return;
    }
    setAoiMode(true);
    setIsDrawingMode(false);
    setMessage("🎯 Draw Area of Interest");
  };

  const clearAOI = () => {
    setAoiRect(null);
    setAoiMode(false);
    setMessage("🗑️ AOI cleared");
    // Redraw without AOI - need to do this after state update
    setTimeout(() => {
      const canvas = drawingCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Redraw only the slots
        (drawnRectangles || []).forEach((slot, index) => {
          const [x1, y1, x2, y2] = slot.bbox;
          ctx.strokeStyle = "#10B981";
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = "#10B981";
          ctx.font = "bold 24px Arial";
          ctx.fillText(`#${index + 1}`, x1 + 10, y1 + 30);
        });
      }
    }, 0);
  };

  // --- SLOT DRAWING ---

  const startDrawing = () => {
    if (drawnRectangles.length >= totalSlots) {
      setMessage(`⚠️ Maximum ${totalSlots} slots reached`);
      return;
    }
    if (!isFrozen) {
      setMessage("⚠️ Please freeze the frame first");
      return;
    }
    setIsDrawingMode(true);
    setAoiMode(false);
    setMessage("✏️ Click and drag to draw slot");
  };

  // Adjustment Mode Canvas Handlers (Document Scanner Style)
  const handleAdjustmentCanvasMouseDown = (e) => {
    if (!isAdjustingGrids || !isFrozen) return;

    const { x, y } = getCanvasCoords(e);
    const result = getGridPointAtCoords(x, y, 15);

    if (result) {
      setAdjustingGridIndex(result.gridIndex);
      setAdjustingPointIndex(result.pointIndex);
    }
  };

  const handleAdjustmentCanvasMouseMove = (e) => {
    if (!isAdjustingGrids || !isFrozen) return;

    const { x, y } = getCanvasCoords(e);

    // Check for hover
    const hoverResult = getGridPointAtCoords(x, y, 15);
    if (hoverResult) {
      setHoveredGridIndex(hoverResult.gridIndex);
      setHoveredPointIndex(hoverResult.pointIndex);
      drawingCanvasRef.current.style.cursor = "grab";
    } else {
      setHoveredGridIndex(null);
      setHoveredPointIndex(null);
      drawingCanvasRef.current.style.cursor = "default";
    }

    // Handle dragging
    if (adjustingGridIndex !== null && adjustingPointIndex !== null) {
      drawingCanvasRef.current.style.cursor = "grabbing";

      const corners =
        gridCorners[adjustingGridIndex] || getGridCorners(adjustingGridIndex);
      if (corners) {
        const newCorners = [...corners];
        // Constrain to canvas bounds
        newCorners[adjustingPointIndex] = {
          x: Math.max(0, Math.min(x, drawingCanvasRef.current.width)),
          y: Math.max(0, Math.min(y, drawingCanvasRef.current.height)),
        };

        setGridCorners({
          ...gridCorners,
          [adjustingGridIndex]: newCorners,
        });

        // Trigger redraw with updated corners
        redrawRectangles();
      }
    }
  };

  const handleAdjustmentCanvasMouseUp = (e) => {
    if (adjustingGridIndex !== null && adjustingPointIndex !== null) {
      // Persist the adjusted corners back to drawnRectangles
      const corners = gridCorners[adjustingGridIndex];
      if (corners) {
        const newBox = cornersToBox(corners);
        if (newBox) {
          const updatedRects = drawnRectangles.map((rect, idx) => {
            if (idx === adjustingGridIndex) {
              const canvas = drawingCanvasRef.current;
              return {
                ...rect,
                bbox: newBox,
                bbox_normalized: [
                  newBox[0] / canvas.width,
                  newBox[1] / canvas.height,
                  newBox[2] / canvas.width,
                  newBox[3] / canvas.height,
                ],
                // Persist perspective corners for rendering
                corners: corners,
              };
            }
            return rect;
          });
          setDrawnRectangles(updatedRects);
          setMessage(`🎯 Grid #${adjustingGridIndex + 1} adjusted`);
        }
      }
    }

    setAdjustingGridIndex(null);
    setAdjustingPointIndex(null);
    drawingCanvasRef.current.style.cursor = "default";
  };

  const handleAdjustmentCanvasMouseLeave = (e) => {
    if (adjustingGridIndex !== null && adjustingPointIndex !== null) {
      // Save pending adjustments
      const corners = gridCorners[adjustingGridIndex];
      if (corners) {
        const newBox = cornersToBox(corners);
        if (newBox) {
          const updatedRects = drawnRectangles.map((rect, idx) => {
            if (idx === adjustingGridIndex) {
              const canvas = drawingCanvasRef.current;
              return {
                ...rect,
                bbox: newBox,
                bbox_normalized: [
                  newBox[0] / canvas.width,
                  newBox[1] / canvas.height,
                  newBox[2] / canvas.width,
                  newBox[3] / canvas.height,
                ],
                // Persist perspective corners for rendering
                corners: corners,
              };
            }
            return rect;
          });
          setDrawnRectangles(updatedRects);
        }
      }

      setAdjustingGridIndex(null);
      setAdjustingPointIndex(null);
    }

    setHoveredGridIndex(null);
    setHoveredPointIndex(null);
    drawingCanvasRef.current.style.cursor = "default";
  };

  const handleCanvasMouseDown = (e) => {
    if (!isFrozen) return;

    // Route to adjustment handler if in adjustment mode
    if (isAdjustingGrids) {
      handleAdjustmentCanvasMouseDown(e);
      return;
    }

    const { x, y } = getCanvasCoords(e);

    if (aoiMode) {
      setCurrentRect({ x1: x, y1: y, x2: x, y2: y });
      setIsDrawingAOI(true);
      return;
    }

    if (!isDrawingMode) return;
    setCurrentRect({ x1: x, y1: y, x2: x, y2: y });
    setIsDrawing(true);
  };

  const handleCanvasMouseMove = (e) => {
    if (isAdjustingGrids) {
      handleAdjustmentCanvasMouseMove(e);
      return;
    }

    if (!currentRect) return;
    const { x, y } = getCanvasCoords(e);

    if (isDrawingAOI) {
      setCurrentRect({ ...currentRect, x2: x, y2: y });
      redrawRectangles(drawnRectangles, { ...currentRect, x2: x, y2: y }, true);
      return;
    }

    if (!isDrawing) return;
    setCurrentRect({ ...currentRect, x2: x, y2: y });
    redrawRectangles(drawnRectangles, { ...currentRect, x2: x, y2: y }, false);
  };

  const handleCanvasMouseUp = (e) => {
    if (isAdjustingGrids) {
      handleAdjustmentCanvasMouseUp(e);
      return;
    }

    if (isDrawingAOI) {
      const { x, y } = getCanvasCoords(e);
      const finalRect = { ...currentRect, x2: x, y2: y };
      const width = Math.abs(finalRect.x2 - finalRect.x1);
      const height = Math.abs(finalRect.y2 - finalRect.y1);

      if (width > 50 && height > 50) {
        const canvas = drawingCanvasRef.current;
        const x1 = Math.min(finalRect.x1, finalRect.x2);
        const y1 = Math.min(finalRect.y1, finalRect.y2);
        const x2 = Math.max(finalRect.x1, finalRect.x2);
        const y2 = Math.max(finalRect.y1, finalRect.y2);

        setAoiRect({
          x1,
          y1,
          x2,
          y2,
          x1_norm: x1 / canvas.width,
          y1_norm: y1 / canvas.height,
          x2_norm: x2 / canvas.width,
          y2_norm: y2 / canvas.height,
        });
        setMessage("✅ AOI defined");
        setAoiMode(false);
      } else {
        setMessage("⚠️ AOI too small");
      }

      setCurrentRect(null);
      setIsDrawingAOI(false);
      return;
    }

    if (!isDrawing) return;

    const { x, y } = getCanvasCoords(e);
    const finalRect = { ...currentRect, x2: x, y2: y };
    const width = Math.abs(finalRect.x2 - finalRect.x1);
    const height = Math.abs(finalRect.y2 - finalRect.y1);

    if (width > 30 && height > 30) {
      const canvas = drawingCanvasRef.current;
      const x1 = Math.min(finalRect.x1, finalRect.x2);
      const y1 = Math.min(finalRect.y1, finalRect.y2);
      const x2 = Math.max(finalRect.x1, finalRect.x2);
      const y2 = Math.max(finalRect.y1, finalRect.y2);

      const newSlot = {
        slot_number: drawnRectangles.length + 1,
        bbox: [x1, y1, x2, y2],
        bbox_normalized: [
          x1 / canvas.width,
          y1 / canvas.height,
          x2 / canvas.width,
          y2 / canvas.height,
        ],
      };

      setDrawnRectangles([...drawnRectangles, newSlot]);
      setMessage(`✅ Slot ${drawnRectangles.length + 1} drawn`);
      setIsDrawingMode(false);
    } else {
      setMessage("⚠️ Rectangle too small");
    }

    setCurrentRect(null);
    setIsDrawing(false);
  };

  const getCanvasCoords = (e) => {
    const canvas = drawingCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const redrawRectangles = useCallback(
    (rects = drawnRectangles, tempRect = null, isAOI = false) => {
      const canvas = drawingCanvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw AOI
      if (aoiRect) {
        const { x1, y1, x2, y2 } = aoiRect;
        ctx.strokeStyle = "#FFFF00";
        ctx.lineWidth = 4;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = "#FFFF00";
        ctx.font = "bold 20px Arial";
        ctx.fillText("AOI", x1 + 10, y1 + 30);
      }

      // Draw slots - ensure rects is always an array
      const safeRects = rects || [];
      safeRects.forEach((slot, index) => {
        let bbox = slot?.bbox || slot;

        // Handle drawing preview format (x1, y1, x2, y2 properties)
        if (bbox && typeof bbox === "object" && bbox.x1 !== undefined) {
          bbox = [bbox.x1, bbox.y1, bbox.x2, bbox.y2];
        }

        if (!Array.isArray(bbox) || bbox.length < 4) return;

        // Use perspective corners for rendering (hexagonal/document scanner style adjustment)
        const perspectiveCorners = gridCorners[index] || getGridCorners(index);
        if (perspectiveCorners && perspectiveCorners.length === 4) {
          // Render as perspective quadrilateral
          drawPerspectiveQuad(
            ctx,
            perspectiveCorners,
            "#10B981",
            `#${index + 1}`,
            index,
          );
        } else {
          // Fallback to axis-aligned rectangle if no perspective data
          const [x1, y1, x2, y2] = bbox;
          ctx.strokeStyle = "#10B981";
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = "#10B981";
          ctx.font = "bold 24px Arial";
          ctx.fillText(`#${index + 1}`, x1 + 10, y1 + 30);
        }

        // Draw adjustment handles if in adjustment mode or hovering
        if (isAdjustingGrids || hoveredGridIndex === index) {
          const corners = gridCorners[adjustingGridIndex]
            ? gridCorners[adjustingGridIndex]
            : getGridCorners(index);
          if (corners) {
            const cornerLabels = ["↖", "↗", "↘", "↙"];
            corners.forEach((corner, pointIdx) => {
              const isActive =
                adjustingGridIndex === index &&
                adjustingPointIndex === pointIdx;
              const isHovered =
                hoveredGridIndex === index && hoveredPointIndex === pointIdx;

              const size = isActive ? 10 : isHovered ? 8 : 6;
              const color = isActive
                ? "#FF0000"
                : isHovered
                  ? "#FFC107"
                  : "#00FF00";

              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(corner.x, corner.y, size, 0, 2 * Math.PI);
              ctx.fill();

              // Draw label
              ctx.fillStyle = color;
              ctx.font = "bold 12px Arial";
              ctx.fillText(cornerLabels[pointIdx], corner.x + 12, corner.y - 8);
            });
          }
        }
      });

      // Draw live quadrilateral when actively adjusting
      if (
        isAdjustingGrids &&
        adjustingGridIndex !== null &&
        gridCorners[adjustingGridIndex]
      ) {
        const corners = gridCorners[adjustingGridIndex];
        ctx.strokeStyle = "#FF0000";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y);
        ctx.lineTo(corners[1].x, corners[1].y);
        ctx.lineTo(corners[2].x, corners[2].y);
        ctx.lineTo(corners[3].x, corners[3].y);
        ctx.closePath();
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw temp rectangle
      if (tempRect) {
        const x = Math.min(tempRect.x1, tempRect.x2);
        const y = Math.min(tempRect.y1, tempRect.y2);
        const w = Math.abs(tempRect.x2 - tempRect.x1);
        const h = Math.abs(tempRect.y2 - tempRect.y1);

        ctx.strokeStyle = isAOI ? "#FFFF00" : "#F59E0B";
        ctx.lineWidth = isAOI ? 4 : 2;
        ctx.strokeRect(x, y, w, h);
      }
    },
    [
      drawnRectangles,
      aoiRect,
      isAdjustingGrids,
      adjustingGridIndex,
      adjustingPointIndex,
      hoveredGridIndex,
      hoveredPointIndex,
      gridCorners,
      getGridCorners,
    ],
  );

  const deleteLastRectangle = () => {
    if (drawnRectangles.length === 0) return;
    setDrawnRectangles(drawnRectangles.slice(0, -1));
    setMessage(`🗑️ Deleted slot ${drawnRectangles.length}`);
    // Redraw canvas
    setTimeout(() => {
      const canvas = drawingCanvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Redraw AOI
        if (aoiRect) {
          const { x1, y1, x2, y2 } = aoiRect;
          ctx.strokeStyle = "#FFFF00";
          ctx.lineWidth = 4;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = "#FFFF00";
          ctx.font = "bold 20px Arial";
          ctx.fillText("AOI", x1 + 10, y1 + 30);
        }
        // Redraw remaining slots (minus last one)
        const remaining = drawnRectangles.slice(0, -1);
        remaining.forEach((slot, index) => {
          const [x1, y1, x2, y2] = slot.bbox;
          ctx.strokeStyle = "#10B981";
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          ctx.fillStyle = "#10B981";
          ctx.font = "bold 24px Arial";
          ctx.fillText(`#${index + 1}`, x1 + 10, y1 + 30);
        });
      }
    }, 0);
  };

  const clearAllRectangles = async () => {
    setDrawnRectangles([]);
    setGridConfig(null); // Also clear grid config
    setAoiRect(null); // Clear AOI too
    setMessage("🗑️ All slots and AOI cleared");

    // Clear canvas
    const canvas = drawingCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Also clear from database so it doesn't reload on page refresh
    try {
      await API.delete(`/ai/clear-grid-config/${spotId}`);
      console.log("✅ Grid config cleared from database");
    } catch (err) {
      console.warn(
        "⚠️ Could not clear grid config from database:",
        err.message,
      );
    }
  };

  // --- AUTO-DETECT GRID ---

  const autoDetectGrid = async () => {
    if (!isFrozen || !frozenImage) {
      setMessage("⚠️ Please freeze the frame first");
      return;
    }

    setAutoDetecting(true);
    setMessage("🔍 Auto-detecting grid...");

    try {
      const base64Frame = frozenImage.split(",")[1];

      const requestBody = { frame: base64Frame };
      if (aoiRect) {
        requestBody.aoi = {
          bbox: [aoiRect.x1, aoiRect.y1, aoiRect.x2, aoiRect.y2],
          bbox_normalized: [
            aoiRect.x1_norm,
            aoiRect.y1_norm,
            aoiRect.x2_norm,
            aoiRect.y2_norm,
          ],
        };
      }

      const res = await fetch(`${BACKEND_SERVER}/api/ai/detect-grid`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(requestBody),
      });

      const result = await res.json();

      if (result.success && result.cells && result.cells.length > 0) {
        setDrawnRectangles(result.cells);
        setMessage(`✅ Auto-detected ${result.num_cells} slots`);

        if (result.annotated_frame) {
          setFrozenImage(result.annotated_frame);
        }
      } else {
        setMessage(result.message || "Could not detect grid");
      }
    } catch (err) {
      setMessage(`❌ Auto-detect failed: ${err.message}`);
    } finally {
      setAutoDetecting(false);
    }
  };

  // --- SAVE GRID ---

  const saveGridConfiguration = async () => {
    if (drawnRectangles.length === 0) {
      setMessage("⚠️ Please draw at least one slot");
      return;
    }

    if (!cameraUrl) {
      setMessage("⚠️ Please configure camera URL first");
      return;
    }

    const canvas = drawingCanvasRef.current || frozenCanvasRef.current;
    const frameWidth = canvas?.width || 1280;
    const frameHeight = canvas?.height || 720;

    const cfg = {
      spot_id: spotId,
      cells: drawnRectangles,
      frame_width: frameWidth,
      frame_height: frameHeight,
      camera_url: cameraUrl, // 🔥 Include camera URL
      detected_at: Math.floor(Date.now() / 1000),
    };

    if (aoiRect) {
      cfg.aoi = {
        bbox: [aoiRect.x1, aoiRect.y1, aoiRect.x2, aoiRect.y2],
        bbox_normalized: [
          aoiRect.x1_norm,
          aoiRect.y1_norm,
          aoiRect.x2_norm,
          aoiRect.y2_norm,
        ],
      };
    }

    try {
      const res = await fetch(
        `${BACKEND_SERVER}/api/ai/save-grid-config/${spotId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ grid_config: cfg }),
        },
      );

      if (res.ok) {
        setGridConfig(cfg);
        setMessage("✅ Grid saved - Ready to start detection!");
      } else {
        setMessage("❌ Failed to save grid");
      }
    } catch (err) {
      setMessage(`❌ Error: ${err.message}`);
    }
  };

  // --- START/STOP DETECTION ---

  const startDetection = async () => {
    if (!gridConfig) {
      setMessage("⚠️ Save grid configuration first");
      return;
    }

    try {
      const res = await fetch(`${BACKEND_SERVER}/api/ai/start-detection`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          parking_spot_id: spotId,
          grid_config: gridConfig,
        }),
      });

      const result = await res.json();
      if (result.success) {
        setIsDetecting(true);
        setIsFrozen(false);
        setFrozenImage(null);
        setMessage(`🚀 Detection started with ${result.num_slots} slots`);
      } else {
        setMessage(`❌ ${result.message}`);
      }
    } catch (err) {
      setMessage(`❌ ${err.message}`);
    }
  };

  const stopDetection = async () => {
    try {
      const res = await fetch(`${BACKEND_SERVER}/api/ai/stop-detection`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ parking_spot_id: spotId }),
      });

      if (res.ok) {
        setIsDetecting(false);
        setFps(0);
        setMessage("⏹️ Detection stopped");
      }
    } catch (err) {
      setMessage(`❌ ${err.message}`);
    }
  };

  // --- MODE TOGGLE ---

  const toggleMode = async (newMode) => {
    try {
      const res = await fetch(
        `${BACKEND_SERVER}/api/ai/toggle-mode/${spotId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ mode: newMode }),
        },
      );

      const data = await res.json();
      if (res.ok) {
        setMode(newMode);
        setMessage(`Switched to ${newMode} mode`);
      } else {
        setMessage(`❌ ${data.message}`);
      }
    } catch (err) {
      setMessage(`❌ ${err.message}`);
    }
  };

  // Setup drawing canvas when frozen image loads
  useEffect(() => {
    if (isFrozen && frozenImage && frozenCanvasRef.current) {
      const img = new Image();
      img.onload = () => {
        const canvas = frozenCanvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;

        if (drawingCanvasRef.current) {
          drawingCanvasRef.current.width = img.width;
          drawingCanvasRef.current.height = img.height;
          redrawRectangles();
        }
      };
      img.src = frozenImage;
    }
  }, [isFrozen, frozenImage]);

  // Redraw when rectangles or AOI change
  useEffect(() => {
    if (drawingCanvasRef.current) {
      redrawRectangles();
    }
  }, [drawnRectangles, aoiRect]);

  // --- STYLES ---
  const styles = {
    container: {
      fontFamily:
        "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      width: "100vw",
      height: "100vh",
      display: "flex",
      flexDirection: "column",
      backgroundColor: "#f6f6f8",
      color: "#0d121b",
      overflow: "hidden",
      zIndex: 100,
    },
    header: {
      height: "64px",
      borderBottom: "1px solid #e7ebf3",
      backgroundColor: "#ffffff",
      padding: "0 24px",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      flexShrink: 0,
      boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
    },
    headerLeft: {
      display: "flex",
      alignItems: "center",
      gap: "16px",
    },
    headerTitle: {
      fontSize: "18px",
      fontWeight: 700,
      margin: 0,
    },
    statusChips: {
      display: "flex",
      alignItems: "center",
      gap: "12px",
    },
    statusChip: {
      display: "flex",
      height: "32px",
      alignItems: "center",
      gap: "8px",
      borderRadius: "9999px",
      padding: "0 12px",
      fontSize: "12px",
      fontWeight: 600,
    },
    statusChipActive: {
      backgroundColor: "rgba(34, 197, 94, 0.1)",
      border: "1px solid rgba(34, 197, 94, 0.2)",
      color: "#15803d",
    },
    statusChipBlue: {
      backgroundColor: "rgba(59, 130, 246, 0.1)",
      border: "1px solid rgba(59, 130, 246, 0.2)",
      color: "#1d4ed8",
    },
    dot: {
      width: "8px",
      height: "8px",
      borderRadius: "50%",
      backgroundColor: "#22c55e",
    },
    mainLayout: {
      display: "flex",
      flex: 1,
      overflow: "hidden",
    },
    sidebar: {
      width: "320px",
      backgroundColor: "#ffffff",
      borderRight: "1px solid #e7ebf3",
      display: "flex",
      flexDirection: "column",
      flexShrink: 0,
      overflowY: "auto",
      padding: "24px",
    },
    button: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "10px 16px",
      borderRadius: "8px",
      border: "1px solid #e5e7eb",
      backgroundColor: "#ffffff",
      fontSize: "14px",
      fontWeight: 600,
      cursor: "pointer",
      transition: "all 0.2s",
      width: "100%",
      justifyContent: "center",
      marginBottom: "8px",
    },
    buttonPrimary: {
      backgroundColor: "#135bec",
      color: "#ffffff",
      border: "none",
      boxShadow: "0 4px 6px -1px rgba(19, 91, 236, 0.2)",
    },
    buttonSuccess: {
      backgroundColor: "#10B981",
      color: "#ffffff",
      border: "none",
    },
    buttonDanger: {
      backgroundColor: "#EF4444",
      color: "#ffffff",
      border: "none",
    },
    buttonWarning: {
      backgroundColor: "#F59E0B",
      color: "#ffffff",
      border: "none",
    },
    buttonPurple: {
      backgroundColor: "#8B5CF6",
      color: "#ffffff",
      border: "none",
    },
    buttonInfo: {
      backgroundColor: "#3B82F6",
      color: "#ffffff",
      border: "none",
    },
    buttonDisabled: {
      backgroundColor: "#E5E7EB",
      color: "#9CA3AF",
      cursor: "not-allowed",
      border: "none",
    },
    mainContent: {
      flex: 1,
      display: "flex",
      flexDirection: "column",
      minWidth: 0,
      backgroundColor: "#f6f6f8",
      position: "relative",
      padding: "24px",
    },
    videoContainer: {
      position: "relative",
      width: "100%",
      height: "100%",
      backgroundColor: "#000000",
      borderRadius: "12px",
      overflow: "hidden",
      boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
      border: "1px solid #e5e7eb",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    freezeButton: {
      position: "absolute",
      bottom: "16px",
      left: "16px",
      zIndex: 20,
      display: "flex",
      alignItems: "center",
      gap: "8px",
      backgroundColor: "rgba(0, 0, 0, 0.6)",
      backdropFilter: "blur(8px)",
      color: "#ffffff",
      padding: "8px 12px",
      borderRadius: "8px",
      border: "none",
      cursor: "pointer",
      fontSize: "12px",
      fontWeight: 500,
    },
    floatingActionButton: {
      position: "absolute",
      bottom: "16px",
      right: "16px",
      zIndex: 10,
      display: "flex",
      alignItems: "center",
      gap: "8px",
      backgroundColor: "rgba(0, 0, 0, 0.6)",
      backdropFilter: "blur(8px)",
      color: "#ffffff",
      padding: "12px 16px",
      borderRadius: "8px",
      border: "none",
      cursor: "pointer",
      fontSize: "13px",
      fontWeight: 500,
      pointerEvents: "none",
    },
    occupancyGrid: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
      gap: "8px",
      marginTop: "16px",
      padding: "16px",
      backgroundColor: "#ffffff",
      borderRadius: "8px",
    },
    occupancyCard: {
      padding: "12px",
      borderRadius: "8px",
      textAlign: "center",
    },
  };

  // --- RENDER ---

  // Loading state
  if (isLoading) {
    return (
      <div
        style={{
          ...styles.container,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: "16px",
        }}
      >
        <div
          style={{
            width: "50px",
            height: "50px",
            border: "4px solid #e5e7eb",
            borderTop: "4px solid #3B82F6",
            borderRadius: "50%",
            animation: "spin 1s linear infinite",
          }}
        />
        <p style={{ color: "#6b7280" }}>Loading configuration...</p>
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* IP Modal */}
      {showIpModal && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1000,
          }}
          onClick={() => setShowIpModal(false)}
        >
          <div
            style={{
              backgroundColor: "#ffffff",
              borderRadius: "16px",
              padding: "24px",
              width: "420px",
              maxHeight: "80vh",
              overflowY: "auto",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ marginTop: 0, marginBottom: "16px" }}>
              Connect IP Webcam
            </h3>
            <input
              type="text"
              value={ipInputValue}
              onChange={(e) => setIpInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleIpSubmit();
              }}
              placeholder="192.168.1.100:8080"
              style={{
                width: "100%",
                padding: "12px",
                border: "1px solid #e5e7eb",
                borderRadius: "8px",
                marginBottom: "16px",
                boxSizing: "border-box",
                fontSize: "14px",
              }}
              autoFocus
            />

            {/* Previous URLs */}
            {previousCameraUrls.length > 0 && (
              <div style={{ marginBottom: "16px" }}>
                <label
                  style={{
                    display: "block",
                    fontSize: "12px",
                    fontWeight: "600",
                    color: "#6b7280",
                    marginBottom: "8px",
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                  }}
                >
                  📚 Previously Used URLs
                </label>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "8px",
                  }}
                >
                  {previousCameraUrls.map((url, idx) => (
                    <button
                      key={idx}
                      onClick={() => selectPreviousUrl(url)}
                      style={{
                        padding: "10px 12px",
                        backgroundColor:
                          ipInputValue === url ? "#dbeafe" : "#f3f4f6",
                        border:
                          ipInputValue === url
                            ? "2px solid #3b82f6"
                            : "1px solid #e5e7eb",
                        borderRadius: "8px",
                        cursor: "pointer",
                        textAlign: "left",
                        fontSize: "13px",
                        color: "#374151",
                        fontFamily: "monospace",
                        transition: "all 0.2s",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = "#eff6ff";
                        e.currentTarget.style.borderColor = "#3b82f6";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor =
                          ipInputValue === url ? "#dbeafe" : "#f3f4f6";
                        e.currentTarget.style.borderColor =
                          ipInputValue === url ? "#3b82f6" : "#e5e7eb";
                      }}
                      title={url}
                    >
                      ✓ {url}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: "flex", gap: "8px" }}>
              <button
                onClick={() => setShowIpModal(false)}
                style={{ ...styles.button, flex: 1 }}
              >
                Cancel
              </button>
              <button
                onClick={handleIpSubmit}
                style={{ ...styles.button, ...styles.buttonPrimary, flex: 1 }}
              >
                <Wifi size={16} />
                Connect
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <h1 style={styles.headerTitle}>AI Camera Setup - Spot #{spotId}</h1>
        </div>

        <div style={styles.statusChips}>
          <div
            style={{
              ...styles.statusChip,
              ...(cameraUrl
                ? styles.statusChipActive
                : { border: "1px solid #e5e7eb" }),
            }}
          >
            <div
              style={
                cameraUrl
                  ? styles.dot
                  : {
                      width: "8px",
                      height: "8px",
                      borderRadius: "50%",
                      backgroundColor: "#ef4444",
                    }
              }
            ></div>
            <span>{cameraUrl ? "Camera Configured" : "No Camera"}</span>
          </div>
          {isDetecting && (
            <div style={{ ...styles.statusChip, ...styles.statusChipBlue }}>
              <span>⚡</span>
              <span>{fps} FPS</span>
            </div>
          )}
        </div>

        <button
          onClick={handleClose}
          style={{
            padding: "8px 16px",
            background: "#EF4444",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          <ArrowLeft size={16} />
          Back
        </button>
      </header>

      <div style={styles.mainLayout}>
        {/* Sidebar */}
        <aside style={styles.sidebar} className="ai-camera-setup-sidebar">
          <h3>Setup Wizard</h3>

          {/* Mode */}
          <div style={{ marginBottom: "16px" }}>
            <button
              onClick={() => toggleMode("manual")}
              style={{
                ...styles.button,
                ...(mode === "manual" ? styles.buttonPrimary : {}),
              }}
            >
              Manual Mode
            </button>
            <button
              onClick={() => toggleMode("ai")}
              style={{
                ...styles.button,
                ...(mode === "ai" ? styles.buttonPrimary : {}),
              }}
            >
              AI Mode
            </button>
          </div>

          <hr />

          {/* Camera URL */}
          <h4>📹 Camera</h4>
          <button
            onClick={startIPWebcam}
            style={{ ...styles.button, ...styles.buttonSuccess }}
          >
            <Wifi size={16} /> Configure Camera URL
          </button>

          <hr />

          {cameraUrl && !isDetecting && (
            <>
              {/* Frame Controls */}
              <h4>📸 Frame</h4>
              <button
                onClick={isFrozen ? unfreezeFrame : freezeFrame}
                style={{
                  ...styles.button,
                  ...(isFrozen ? styles.buttonSuccess : styles.buttonPrimary),
                }}
              >
                {isFrozen ? "▶️ Unfreeze" : "⏸️ Freeze"}
              </button>

              {/* AOI */}
              <h4>🎯 Area of Interest</h4>
              {!aoiRect ? (
                <button
                  onClick={startDrawingAOI}
                  disabled={!isFrozen}
                  style={{
                    ...styles.button,
                    ...(!isFrozen
                      ? styles.buttonDisabled
                      : styles.buttonWarning),
                  }}
                >
                  <Target size={16} /> Draw AOI
                </button>
              ) : (
                <button
                  onClick={clearAOI}
                  style={{ ...styles.button, ...styles.buttonWarning }}
                >
                  <Trash2 size={16} /> Clear AOI
                </button>
              )}

              {/* Slots */}
              <h4>
                🅿️ Parking Slots ({drawnRectangles.length}/{totalSlots})
              </h4>
              <button
                onClick={autoDetectGrid}
                disabled={autoDetecting || !isFrozen}
                style={{
                  ...styles.button,
                  ...(autoDetecting || !isFrozen
                    ? styles.buttonDisabled
                    : styles.buttonPurple),
                }}
              >
                🤖 {autoDetecting ? "Detecting..." : "Auto-Detect"}
              </button>
              <button
                onClick={startDrawing}
                disabled={drawnRectangles.length >= totalSlots || !isFrozen}
                style={{
                  ...styles.button,
                  ...(drawnRectangles.length >= totalSlots || !isFrozen
                    ? styles.buttonDisabled
                    : styles.buttonPurple),
                }}
              >
                <Edit3 size={16} /> Draw Slot
              </button>
              <button
                onClick={deleteLastRectangle}
                disabled={drawnRectangles.length === 0}
                style={{
                  ...styles.button,
                  ...(drawnRectangles.length === 0
                    ? styles.buttonDisabled
                    : styles.buttonWarning),
                }}
              >
                <Trash2 size={16} /> Undo
              </button>
              <button
                onClick={clearAllRectangles}
                disabled={
                  drawnRectangles.length === 0 && !aoiRect && !gridConfig
                }
                style={{
                  ...styles.button,
                  ...(drawnRectangles.length === 0 && !aoiRect && !gridConfig
                    ? styles.buttonDisabled
                    : styles.buttonDanger),
                }}
              >
                <Trash2 size={16} /> Clear All
              </button>

              {/* Adjust Grids Button */}
              <button
                onClick={() => {
                  setIsAdjustingGrids(!isAdjustingGrids);
                  setMessage(
                    isAdjustingGrids
                      ? "❌ Adjustment mode disabled"
                      : "🎯 Adjustment mode - drag corners like a document scanner",
                  );
                }}
                disabled={drawnRectangles.length === 0 || !isFrozen}
                style={{
                  ...styles.button,
                  ...(drawnRectangles.length === 0 || !isFrozen
                    ? styles.buttonDisabled
                    : isAdjustingGrids
                      ? styles.buttonWarning
                      : styles.buttonInfo),
                }}
              >
                <Grid3X3 size={16} />{" "}
                {isAdjustingGrids ? "Stop Adjusting" : "Adjust Grids"}
              </button>

              {/* Save Grid */}
              <hr />
              <button
                onClick={saveGridConfiguration}
                disabled={drawnRectangles.length === 0}
                style={{
                  ...styles.button,
                  ...(drawnRectangles.length === 0
                    ? styles.buttonDisabled
                    : styles.buttonSuccess),
                }}
              >
                <Save size={16} /> Save Grid
              </button>
            </>
          )}

          {/* Detection Controls */}
          {gridConfig && (
            <>
              <hr />
              <h4>🚀 Detection</h4>
              {!isDetecting ? (
                <button
                  onClick={startDetection}
                  style={{ ...styles.button, ...styles.buttonSuccess }}
                >
                  <Play size={16} /> Start Detection
                </button>
              ) : (
                <button
                  onClick={stopDetection}
                  style={{ ...styles.button, ...styles.buttonDanger }}
                >
                  <Square size={16} /> Stop Detection
                </button>
              )}
            </>
          )}
        </aside>

        {/* Main Content */}
        <main style={styles.mainContent}>
          <div style={styles.videoContainer}>
            {/* Processed Frame with AI Annotations (when detecting and have processed frame) */}
            {isDetecting && processedFrame && (
              <img
                src={processedFrame}
                alt="AI Processed Stream"
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  maxWidth: "100%",
                  maxHeight: "100%",
                  width: "auto",
                  height: "auto",
                  objectFit: "contain",
                }}
              />
            )}

            {/* Live Stream - keep mounted but hide when frozen or when processed frame is shown */}
            {cameraUrl && (
              <img
                ref={videoRef}
                src={STREAM_URL}
                alt="Live Stream"
                crossOrigin="anonymous"
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  maxWidth: "100%",
                  maxHeight: "100%",
                  width: "auto",
                  height: "auto",
                  objectFit: "contain",
                  display:
                    isFrozen || (isDetecting && processedFrame)
                      ? "none"
                      : "block",
                }}
                onLoad={() => {
                  if (drawingCanvasRef.current && videoRef.current) {
                    const w = videoRef.current.naturalWidth || 1280;
                    const h = videoRef.current.naturalHeight || 720;
                    drawingCanvasRef.current.width = w;
                    drawingCanvasRef.current.height = h;
                  }
                }}
              />
            )}

            {/* Frozen Frame */}
            {isFrozen && frozenImage && (
              <>
                <img
                  src={frozenImage}
                  alt="Frozen"
                  style={{
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    transform: "translate(-50%, -50%)",
                    maxWidth: "100%",
                    maxHeight: "100%",
                    width: "auto",
                    height: "auto",
                    objectFit: "contain",
                  }}
                />
                <canvas ref={frozenCanvasRef} style={{ display: "none" }} />
              </>
            )}

            {/* Drawing Canvas */}
            {!isDetecting && (isFrozen || cameraUrl) && (
              <canvas
                ref={drawingCanvasRef}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleAdjustmentCanvasMouseLeave}
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  maxWidth: "100%",
                  maxHeight: "100%",
                  cursor:
                    isAdjustingGrids && isFrozen
                      ? "grab"
                      : isDrawingMode || aoiMode
                        ? "crosshair"
                        : "default",
                  pointerEvents: isFrozen ? "auto" : "none",
                }}
              />
            )}

            {/* Freeze Button */}
            {cameraUrl && !isDetecting && (
              <button
                style={styles.freezeButton}
                onClick={isFrozen ? unfreezeFrame : freezeFrame}
              >
                {isFrozen ? "▶️ Unfreeze" : "⏸ Freeze"}
              </button>
            )}

            {/* Action Status - only show when not actively drawing */}
            {isFrozen && !isDrawingMode && !aoiMode && (
              <div style={styles.floatingActionButton}>
                {aoiRect ? "✏️ Draw Slot" : "🎯 Draw Area of Interest"}
              </div>
            )}

            {/* No Camera Message */}
            {!cameraUrl && (
              <div style={{ color: "#9ca3af", fontSize: "18px" }}>
                📹 Configure camera URL to start
              </div>
            )}
          </div>

          {/* Occupancy Status */}
          {isDetecting && occupancyStatus?.slots && (
            <div style={styles.occupancyGrid}>
              {Object.entries(occupancyStatus.slots).map(([slotNum, data]) => {
                const status = data?.status || "unknown";
                const bgColor =
                  status === "occupied"
                    ? "#FEE2E2"
                    : status === "vacant"
                      ? "#ECFDF5"
                      : "#F3F4F6";
                const textColor =
                  status === "occupied"
                    ? "#B91C1C"
                    : status === "vacant"
                      ? "#065F46"
                      : "#374151";

                return (
                  <div
                    key={slotNum}
                    style={{
                      ...styles.occupancyCard,
                      backgroundColor: bgColor,
                    }}
                  >
                    <div style={{ fontWeight: 600, color: textColor }}>
                      Slot #{slotNum}
                    </div>
                    <div
                      style={{ fontSize: "12px", textTransform: "capitalize" }}
                    >
                      {status}
                    </div>
                    <div style={{ fontSize: "11px", color: "#9CA3AF" }}>
                      {data?.confidence || "N/A"}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </main>
      </div>

      {/* Message Toast */}
      {message && (
        <div
          style={{
            position: "fixed",
            bottom: "20px",
            left: "50%",
            transform: "translateX(-50%)",
            backgroundColor: "#020617",
            color: "#fff",
            padding: "12px 24px",
            borderRadius: "8px",
            fontSize: "14px",
            fontWeight: 500,
            boxShadow: "0 10px 25px rgba(0,0,0,0.3)",
            zIndex: 1000,
          }}
          onClick={() => setMessage("")}
        >
          {message}
        </div>
      )}
    </div>
  );
}

export default AICameraSetup;
