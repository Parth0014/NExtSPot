// routes/spots.js - Multi-Slot Parking Spots Management
const express = require("express");
const router = express.Router();
const db = require("../config/db");
const auth = require("../middleware/authMiddleware");

// ================= USER ENDPOINTS =================

// Get all spots with availability (for USER map view)
router.get("/available", auth(["user"]), (req, res) => {
  const query = `
    SELECT 
      ps.id, 
      ps.name, 
      ps.latitude, 
      ps.longitude, 
      ps.price, 
      ps.type,
      ps.total_slots,
      ps.active_slots,
      ps.available_slots,
      COALESCE(a.occupied_slots, 0) as occupied_slots,
      CASE 
        WHEN ps.available_slots > 0 THEN 'available'
        WHEN ps.available_slots = 0 AND ps.active_slots > 0 THEN 'occupied'
        ELSE 'unavailable'
      END as status
    FROM parking_spots ps
    LEFT JOIN availability a ON ps.id = a.parking_id
    WHERE ps.active_slots > 0
  `;

  db.query(query, (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(result);
  });
});

// Get spot details with comprehensive slot information
router.get("/:id/details", auth(["user", "owner"]), (req, res) => {
  const { id } = req.params;

  const query = `
    SELECT 
      ps.id,
      ps.name,
      ps.latitude,
      ps.longitude,
      ps.price,
      ps.type,
      ps.total_slots,
      ps.active_slots,
      ps.available_slots,
      COALESCE(a.occupied_slots, 0) as occupied_slots,
      (
        SELECT COUNT(*) 
        FROM parking_slots 
        WHERE parking_spot_id = ps.id AND is_active = 1 AND is_available = 1
      ) as bookable_slots,
      CASE 
        WHEN ps.active_slots = 0 THEN 'disabled'
        WHEN ps.available_slots > 0 THEN 'available'
        WHEN ps.available_slots = 0 AND ps.active_slots > 0 THEN 'occupied'
        ELSE 'unavailable'
      END as status
    FROM parking_spots ps
    LEFT JOIN availability a ON ps.id = a.parking_id
    WHERE ps.id = ?
  `;

  db.query(query, [id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!result.length)
      return res.status(404).json({ message: "Spot not found" });
    res.json(result[0]);
  });
});

// ================= OWNER ENDPOINTS =================

// Get all spots of the owner (OWNER dashboard)
router.get("/owner", auth(["owner"]), (req, res) => {
  const query = `
    SELECT 
      ps.id,
      ps.owner_id,
      ps.name,
      ps.latitude,
      ps.longitude,
      ps.price,
      ps.type,
      ps.total_slots,
      ps.active_slots,
      ps.available_slots,
      ps.created_at,
      COALESCE(a.occupied_slots, 0) as occupied_slots,
      CASE 
        WHEN ps.active_slots = 0 THEN 'disabled'
        WHEN ps.available_slots > 0 THEN 'available'
        WHEN ps.available_slots = 0 AND ps.active_slots > 0 THEN 'occupied'
        ELSE 'unavailable'
      END as status
    FROM parking_spots ps
    LEFT JOIN availability a ON ps.id = a.parking_id
    WHERE ps.owner_id = ?
    ORDER BY ps.name
  `;

  db.query(query, [req.user.id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(result);
  });
});

// Add new parking spot with initial slots (OWNER only)
router.post("/add", auth(["owner"]), (req, res) => {
  const {
    name,
    latitude,
    longitude,
    price,
    type,
    initial_slots = 1,
  } = req.body;

  if (!name || !latitude || !longitude || !price || !type) {
    return res.status(400).json({ message: "All fields required" });
  }

  if (initial_slots < 1 || initial_slots > 100) {
    return res
      .status(400)
      .json({ message: "Initial slots must be between 1 and 100" });
  }

  // Insert parking spot with initial counts set to 0
  db.query(
    `INSERT INTO parking_spots 
     (owner_id, name, latitude, longitude, price, type, total_slots, active_slots, available_slots, created_at) 
     VALUES (?,?,?,?,?,?,?,?,?, NOW())`,
    [
      req.user.id,
      name,
      latitude,
      longitude,
      price,
      type,
      initial_slots,
      initial_slots,
      initial_slots,
    ],
    (err, result) => {
      if (err) return res.status(500).json({ error: err.message });
      const parkingId = result.insertId;

      // Initialize availability record
      db.query(
        `INSERT INTO availability 
         (parking_id, total_slots, active_slots, occupied_slots, available_slots, updated_at) 
         VALUES (?, ?, ?, 0, ?, NOW())`,
        [parkingId, initial_slots, initial_slots, initial_slots],
        (err2) => {
          if (err2) return res.status(500).json({ error: err2.message });

          // Create initial slots
          const slotValues = [];
          for (let i = 1; i <= initial_slots; i++) {
            slotValues.push([parkingId, i, 1, 1]); // is_active=1, is_available=1
          }

          db.query(
            "INSERT INTO parking_slots (parking_spot_id, slot_number, is_active, is_available) VALUES ?",
            [slotValues],
            (err3) => {
              if (err3) return res.status(500).json({ error: err3.message });

              res.status(201).json({
                message: "Parking spot added with slots",
                id: parkingId,
                initial_slots: initial_slots,
              });
            },
          );
        },
      );
    },
  );
});

// Delete spot (only if no active bookings on any slot)
router.delete("/:id", auth(["owner"]), (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) return res.status(400).json({ message: "Invalid spot ID" });

  const checkQuery = `
    SELECT 
      ps.owner_id,
      COUNT(DISTINCT b.id) as active_booking_count
    FROM parking_spots ps
    LEFT JOIN parking_slots sl ON ps.id = sl.parking_spot_id
    LEFT JOIN bookings b ON sl.id = b.slot_id AND b.status = 'active'
    WHERE ps.id = ?
    GROUP BY ps.owner_id
  `;

  db.query(checkQuery, [id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!result.length)
      return res.status(404).json({ message: "Spot not found" });

    if (result[0].owner_id !== req.user.id)
      return res.status(403).json({ message: "Not your spot" });

    if (result[0].active_booking_count > 0)
      return res.status(400).json({
        message: `Cannot delete spot. There are ${result[0].active_booking_count} active booking(s).`,
        active_bookings: result[0].active_booking_count,
      });

    // Delete availability record first
    db.query("DELETE FROM availability WHERE parking_id=?", [id], (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });

      // Delete all parking slots (CASCADE will handle this through foreign key)
      db.query(
        "DELETE FROM parking_slots WHERE parking_spot_id=?",
        [id],
        (err3) => {
          if (err3) return res.status(500).json({ error: err3.message });

          // Delete parking spot
          db.query("DELETE FROM parking_spots WHERE id=?", [id], (err4) => {
            if (err4) return res.status(500).json({ error: err4.message });
            res.json({
              message: "Parking spot and all its slots removed successfully",
            });
          });
        },
      );
    });
  });
});

// Get owner's bookings with slot information (OWNER only)
router.get("/owner/bookings", auth(["owner"]), (req, res) => {
  const query = `
    SELECT 
      b.id as booking_id,
      b.reserved_at,
      b.expires_at,
      b.duration_minutes,
      b.total_price,
      b.status,
      ps.id as spot_id,
      ps.name as spot_name,
      ps.total_slots,
      ps.available_slots,
      sl.id as slot_id,
      sl.slot_number,
      u.name as user_name,
      u.email as user_email
    FROM bookings b
    JOIN parking_spots ps ON b.parking_id = ps.id
    LEFT JOIN parking_slots sl ON b.slot_id = sl.id
    JOIN users u ON b.user_id = u.id
    WHERE ps.owner_id = ?
    ORDER BY b.reserved_at DESC
  `;

  db.query(query, [req.user.id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(result);
  });
});

module.exports = router;
