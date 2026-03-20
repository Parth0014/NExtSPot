// routes/slots.js - Individual Slot Management
const express = require("express");
const router = express.Router();
const db = require("../config/db");
const auth = require("../middleware/authMiddleware");

// ================= GET SLOT INFORMATION =================

// Get all slots for a specific parking spot (with details)
router.get("/parking-spot/:spot_id", auth(["owner", "user"]), (req, res) => {
  const { spot_id } = req.params;

  // Check if user is owner (for owner endpoints)
  const ownerCheck = req.user.role === "owner" ? "AND ps.owner_id = ?" : "";

  const params = req.user.role === "owner" ? [spot_id, req.user.id] : [spot_id];

  const query = `
    SELECT 
      sl.id as slot_id,
      sl.parking_spot_id,
      sl.slot_number,
      sl.is_active,
      sl.is_available,
      sl.created_at,
      ps.name as spot_name,
      ps.owner_id,
      b.id as current_booking_id,
      b.user_id as current_user_id,
      b.reserved_at,
      b.expires_at,
      u.name as current_user_name,
      u.email as current_user_email,
      CASE 
        WHEN sl.is_active = 0 THEN 'disabled'
        WHEN sl.is_available = 0 AND b.id IS NOT NULL THEN 'occupied'
        WHEN sl.is_available = 1 THEN 'available'
        ELSE 'unknown'
      END as status
    FROM parking_slots sl
    JOIN parking_spots ps ON sl.parking_spot_id = ps.id
    LEFT JOIN bookings b ON sl.id = b.slot_id AND b.status = 'active'
    LEFT JOIN users u ON b.user_id = u.id
    WHERE sl.parking_spot_id = ? ${ownerCheck}
    ORDER BY sl.slot_number ASC
  `;

  db.query(query, params, (err, result) => {
    if (err) return res.status(500).json({ error: err.message });

    if (!result.length) {
      return res
        .status(404)
        .json({ message: "No slots found for this parking spot" });
    }

    res.json(result);
  });
});

// Get single slot details
router.get("/:slot_id", auth(["owner", "user"]), (req, res) => {
  const { slot_id } = req.params;

  const query = `
    SELECT 
      sl.id as slot_id,
      sl.parking_spot_id,
      sl.slot_number,
      sl.is_active,
      sl.is_available,
      sl.created_at,
      ps.name as spot_name,
      ps.latitude,
      ps.longitude,
      ps.price,
      ps.type,
      ps.owner_id,
      b.id as current_booking_id,
      b.user_id as current_user_id,
      b.reserved_at,
      b.expires_at,
      b.duration_minutes,
      b.total_price,
      u.name as current_user_name,
      u.email as current_user_email,
      CASE 
        WHEN sl.is_active = 0 THEN 'disabled'
        WHEN sl.is_available = 0 AND b.id IS NOT NULL THEN 'occupied'
        WHEN sl.is_available = 1 THEN 'available'
        ELSE 'unknown'
      END as status
    FROM parking_slots sl
    JOIN parking_spots ps ON sl.parking_spot_id = ps.id
    LEFT JOIN bookings b ON sl.id = b.slot_id AND b.status = 'active'
    LEFT JOIN users u ON b.user_id = u.id
    WHERE sl.id = ?
  `;

  db.query(query, [slot_id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!result.length)
      return res.status(404).json({ message: "Slot not found" });

    res.json(result[0]);
  });
});

// ================= OWNER: ADD SLOTS =================

// Add multiple new slots to a parking spot
router.post("/parking-spot/:spot_id/add", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;
  const { count = 1 } = req.body; // Number of slots to add

  if (count < 1 || count > 50) {
    return res
      .status(400)
      .json({ message: "Can add between 1 and 50 slots at a time" });
  }

  // Verify ownership
  db.query(
    "SELECT owner_id, total_slots FROM parking_spots WHERE id = ?",
    [spot_id],
    (err, result) => {
      if (err) return res.status(500).json({ error: err.message });
      if (!result.length)
        return res.status(404).json({ message: "Parking spot not found" });

      if (result[0].owner_id !== req.user.id) {
        return res
          .status(403)
          .json({ message: "Not authorized to modify this parking spot" });
      }

      const currentTotal = result[0].total_slots;

      // Check if adding would exceed maximum
      if (currentTotal + count > 100) {
        return res.status(400).json({
          message: `Cannot exceed 100 total slots. Current: ${currentTotal}, Attempting to add: ${count}`,
        });
      }

      // Get the next slot number
      db.query(
        "SELECT COALESCE(MAX(slot_number), 0) as max_slot FROM parking_slots WHERE parking_spot_id = ?",
        [spot_id],
        (err2, maxResult) => {
          if (err2) return res.status(500).json({ error: err2.message });

          const startNumber = maxResult[0].max_slot + 1;

          // Create insert values for multiple slots
          const values = [];
          for (let i = 0; i < count; i++) {
            values.push([spot_id, startNumber + i, 1, 1]);
          }

          const insertQuery = `
            INSERT INTO parking_slots (parking_spot_id, slot_number, is_active, is_available)
            VALUES ?
          `;

          db.query(insertQuery, [values], (err3) => {
            if (err3) return res.status(500).json({ error: err3.message });

            // Update parking_spots statistics
            db.query(
              `UPDATE parking_spots 
               SET total_slots = total_slots + ?,
                   active_slots = active_slots + ?,
                   available_slots = available_slots + ?
               WHERE id = ?`,
              [count, count, count, spot_id],
              (err4) => {
                if (err4) return res.status(500).json({ error: err4.message });

                // Update availability table
                db.query(
                  `UPDATE availability 
                   SET total_slots = total_slots + ?,
                       active_slots = active_slots + ?,
                       available_slots = available_slots + ?,
                       updated_at = NOW()
                   WHERE parking_id = ?`,
                  [count, count, count, spot_id],
                  (err5) => {
                    if (err5)
                      return res.status(500).json({ error: err5.message });

                    res.status(201).json({
                      message: `Successfully added ${count} slot(s)`,
                      slots_added: count,
                      starting_slot_number: startNumber,
                      total_slots: currentTotal + count,
                    });
                  },
                );
              },
            );
          });
        },
      );
    },
  );
});

// ================= OWNER: DELETE SLOTS =================

// Delete a specific slot (only if not occupied)
router.delete("/:slot_id", auth(["owner"]), (req, res) => {
  const { slot_id } = req.params;

  // Check ownership and slot status
  const checkQuery = `
    SELECT 
      sl.id,
      sl.parking_spot_id,
      sl.slot_number,
      sl.is_active,
      sl.is_available,
      ps.owner_id,
      ps.total_slots,
      b.id as active_booking
    FROM parking_slots sl
    JOIN parking_spots ps ON sl.parking_spot_id = ps.id
    LEFT JOIN bookings b ON sl.id = b.slot_id AND b.status = 'active'
    WHERE sl.id = ?
  `;

  db.query(checkQuery, [slot_id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!result.length)
      return res.status(404).json({ message: "Slot not found" });

    const slot = result[0];

    if (slot.owner_id !== req.user.id) {
      return res
        .status(403)
        .json({ message: "Not authorized to delete this slot" });
    }

    if (slot.active_booking) {
      return res.status(400).json({
        message: "Cannot delete slot with active booking",
        booking_id: slot.active_booking,
      });
    }

    // Prevent deleting the last slot
    if (slot.total_slots <= 1) {
      return res.status(400).json({
        message:
          "Cannot delete the last slot. A parking spot must have at least 1 slot.",
      });
    }

    // Delete the slot
    db.query("DELETE FROM parking_slots WHERE id = ?", [slot_id], (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });

      // Update parking_spots statistics
      const activeDecrement = slot.is_active ? 1 : 0;
      const availableDecrement = slot.is_active && slot.is_available ? 1 : 0;

      db.query(
        `UPDATE parking_spots 
         SET total_slots = total_slots - 1,
             active_slots = active_slots - ?,
             available_slots = available_slots - ?
         WHERE id = ?`,
        [activeDecrement, availableDecrement, slot.parking_spot_id],
        (err3) => {
          if (err3) return res.status(500).json({ error: err3.message });

          // Update availability table
          db.query(
            `UPDATE availability 
             SET total_slots = total_slots - 1,
                 active_slots = active_slots - ?,
                 available_slots = available_slots - ?,
                 updated_at = NOW()
             WHERE parking_id = ?`,
            [activeDecrement, availableDecrement, slot.parking_spot_id],
            (err4) => {
              if (err4) return res.status(500).json({ error: err4.message });

              res.json({
                message: "Slot deleted successfully",
                deleted_slot_number: slot.slot_number,
                remaining_slots: slot.total_slots - 1,
              });
            },
          );
        },
      );
    });
  });
});

// ================= OWNER: TOGGLE SLOT VISIBILITY =================

// Enable/disable a specific slot
router.put("/:slot_id/toggle", auth(["owner"]), (req, res) => {
  const { slot_id } = req.params;
  const { is_active } = req.body; // true to enable, false to disable

  if (typeof is_active !== "boolean") {
    return res
      .status(400)
      .json({ message: "is_active must be a boolean value" });
  }

  // Check ownership and current status
  const checkQuery = `
    SELECT 
      sl.id,
      sl.parking_spot_id,
      sl.slot_number,
      sl.is_active,
      sl.is_available,
      ps.owner_id,
      ps.active_slots,
      b.id as active_booking
    FROM parking_slots sl
    JOIN parking_spots ps ON sl.parking_spot_id = ps.id
    LEFT JOIN bookings b ON sl.id = b.slot_id AND b.status = 'active'
    WHERE sl.id = ?
  `;

  db.query(checkQuery, [slot_id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!result.length)
      return res.status(404).json({ message: "Slot not found" });

    const slot = result[0];

    if (slot.owner_id !== req.user.id) {
      return res.status(403).json({ message: "Not authorized" });
    }

    // Can't disable if it's occupied
    if (!is_active && slot.active_booking) {
      return res.status(400).json({
        message: "Cannot disable slot with active booking",
        booking_id: slot.active_booking,
      });
    }
    // Update slot status
    db.query(
      "UPDATE parking_slots SET is_active = ? WHERE id = ?",
      [is_active ? 1 : 0, slot_id],
      (err2) => {
        if (err2) return res.status(500).json({ error: err2.message });

        // Calculate changes for statistics
        const wasActive = slot.is_active;
        const wasAvailable = slot.is_available;
        const willBeActive = is_active;

        let activeChange = 0;
        let availableChange = 0;

        if (wasActive && !willBeActive) {
          // Disabling
          activeChange = -1;
          if (wasAvailable) availableChange = -1;
        } else if (!wasActive && willBeActive) {
          // Enabling
          activeChange = 1;
          if (wasAvailable) availableChange = 1;
        }

        // Update parking_spots statistics
        db.query(
          `UPDATE parking_spots 
           SET active_slots = active_slots + ?,
               available_slots = available_slots + ?
           WHERE id = ?`,
          [activeChange, availableChange, slot.parking_spot_id],
          (err3) => {
            if (err3) return res.status(500).json({ error: err3.message });

            // Update availability table
            db.query(
              `UPDATE availability 
               SET active_slots = active_slots + ?,
                   available_slots = available_slots + ?,
                   updated_at = NOW()
               WHERE parking_id = ?`,
              [activeChange, availableChange, slot.parking_spot_id],
              (err4) => {
                if (err4) return res.status(500).json({ error: err4.message });

                res.json({
                  message: `Slot ${
                    is_active ? "enabled" : "disabled"
                  } successfully`,
                  slot_number: slot.slot_number,
                  is_active: is_active,
                });
              },
            );
          },
        );
      },
    );
  });
});

// ================= STATISTICS & SUMMARY =================

// Force reset all slots for a parking spot (emergency override)
router.post(
  "/parking-spot/:spot_id/force-reset",
  auth(["owner"]),
  (req, res) => {
    const { spot_id } = req.params;

    // First verify ownership
    db.query(
      "SELECT id, owner_id, total_slots, name FROM parking_spots WHERE id = ?",
      [spot_id],
      (err, result) => {
        if (err) return res.status(500).json({ error: err.message });
        if (!result[0])
          return res.status(404).json({ message: "Spot not found" });
        if (result[0].owner_id !== req.user.id) {
          return res.status(403).json({ message: "Not authorized" });
        }

        const spot = result[0];

        // Step 1: Cancel all active bookings for this spot
        db.query(
          "UPDATE bookings SET status = 'cancelled' WHERE parking_id = ? AND status = 'active'",
          [spot_id],
          (err2, bookingResult) => {
            if (err2) return res.status(500).json({ error: err2.message });

            // Step 2: Reset all slots to available
            db.query(
              "UPDATE parking_slots SET is_available = 1 WHERE parking_spot_id = ?",
              [spot_id],
              (err3, slotResult) => {
                if (err3) return res.status(500).json({ error: err3.message });

                // Step 3: Update parking_spots statistics
                db.query(
                  `UPDATE parking_spots 
                 SET available_slots = active_slots,
                     occupied_slots = 0
                 WHERE id = ?`,
                  [spot_id],
                  (err4) => {
                    if (err4)
                      return res.status(500).json({ error: err4.message });

                    // Step 4: Update availability table
                    db.query(
                      `UPDATE availability 
                     SET occupied_slots = 0,
                         available_slots = (SELECT active_slots FROM parking_spots WHERE id = ?),
                         updated_at = NOW()
                     WHERE parking_id = ?`,
                      [spot_id, spot_id],
                      (err5) => {
                        if (err5)
                          return res.status(500).json({ error: err5.message });

                        console.log(
                          `🔄 Force reset completed for spot ${spot_id} (${spot.name})`,
                        );
                        console.log(
                          `   - Cancelled ${bookingResult.affectedRows} active bookings`,
                        );
                        console.log(
                          `   - Reset ${slotResult.affectedRows} slots to available`,
                        );

                        res.json({
                          success: true,
                          message: `Force reset completed for ${spot.name}`,
                          cancelled_bookings: bookingResult.affectedRows,
                          reset_slots: slotResult.affectedRows,
                        });
                      },
                    );
                  },
                );
              },
            );
          },
        );
      },
    );
  },
);

// Get detailed statistics for a parking spot
router.get("/parking-spot/:spot_id/stats", auth(["owner"]), (req, res) => {
  const { spot_id } = req.params;

  const statsQuery = `
    SELECT 
      ps.id as parking_spot_id,
      ps.name,
      ps.total_slots,
      ps.active_slots,
      ps.available_slots,
      COUNT(sl.id) as actual_slot_count,
      COUNT(CASE WHEN sl.is_active = 1 THEN 1 END) as enabled_slots,
      COUNT(CASE WHEN sl.is_active = 0 THEN 1 END) as disabled_slots,
      COUNT(CASE WHEN sl.is_active = 1 AND sl.is_available = 1 THEN 1 END) as truly_available,
      COUNT(CASE WHEN sl.is_active = 1 AND sl.is_available = 0 THEN 1 END) as occupied_slots,
      a.occupied_slots as availability_occupied
    FROM parking_spots ps
    LEFT JOIN parking_slots sl ON ps.id = sl.parking_spot_id
    LEFT JOIN availability a ON ps.id = a.parking_id
    WHERE ps.id = ? AND ps.owner_id = ?
    GROUP BY ps.id, ps.name, ps.total_slots, ps.active_slots, ps.available_slots, a.occupied_slots
  `;

  db.query(statsQuery, [spot_id, req.user.id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!result.length)
      return res
        .status(404)
        .json({ message: "Parking spot not found or not authorized" });

    res.json(result[0]);
  });
});

module.exports = router;
