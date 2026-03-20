// routes/bookings.js - Multi-Slot Booking System
const express = require("express");
const router = express.Router();
const db = require("../config/db");
const auth = require("../middleware/authMiddleware");

// Reserve spot (USER) - Automatically assigns an available slot
// Accepts optional `start_at` (ISO datetime) and `duration_minutes` (int)
router.post("/reserve/:parking_id", auth(["user"]), (req, res) => {
  const { parking_id } = req.params;
  const { duration_minutes = 60, start_at } = req.body;

  const MIN_MINUTES = 30; // allow 30 minutes minimum
  const MAX_MINUTES = 7 * 24 * 60; // 7 days by default

  if (
    typeof duration_minutes !== "number" ||
    duration_minutes < MIN_MINUTES ||
    duration_minutes > MAX_MINUTES
  ) {
    return res.status(400).json({
      message: `Duration must be between ${MIN_MINUTES} and ${MAX_MINUTES} minutes`,
    });
  }

  // Parse start_at if provided, otherwise start now
  let startAt = start_at ? new Date(start_at) : new Date();
  if (isNaN(startAt.getTime())) {
    return res.status(400).json({ message: "Invalid start_at datetime" });
  }
  // Do not allow bookings that start in the past
  const now = new Date();
  if (startAt < now) {
    return res.status(400).json({ message: "start_at cannot be in the past" });
  }

  // Check if parking spot exists and has available slots
  const checkQuery = `
    SELECT 
      ps.id, 
      ps.price, 
      ps.total_slots,
      ps.active_slots,
      ps.available_slots
    FROM parking_spots ps
    WHERE ps.id = ?
  `;

  db.query(checkQuery, [parking_id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });

    if (!result[0]) {
      return res.status(404).json({ message: "Parking spot not found" });
    }

    const spot = result[0];

    // Check if there are available slots
    if (spot.available_slots <= 0) {
      return res.status(400).json({
        message: "No parking slots available at this location",
        total_slots: spot.total_slots,
        active_slots: spot.active_slots,
        available_slots: spot.available_slots,
      });
    }

    // Convert startAt to MySQL format - CORRECTLY preserving local timezone
    const year = startAt.getFullYear();
    const month = String(startAt.getMonth() + 1).padStart(2, "0");
    const day = String(startAt.getDate()).padStart(2, "0");
    const hours = String(startAt.getHours()).padStart(2, "0");
    const minutes = String(startAt.getMinutes()).padStart(2, "0");
    const seconds = String(startAt.getSeconds()).padStart(2, "0");
    const startAtSql = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;

    // Calculate end time for overlap checking
    const endAt = new Date(startAt.getTime() + duration_minutes * 60000);
    const endYear = endAt.getFullYear();
    const endMonth = String(endAt.getMonth() + 1).padStart(2, "0");
    const endDay = String(endAt.getDate()).padStart(2, "0");
    const endHours = String(endAt.getHours()).padStart(2, "0");
    const endMinutes = String(endAt.getMinutes()).padStart(2, "0");
    const endSeconds = String(endAt.getSeconds()).padStart(2, "0");
    const endAtSql = `${endYear}-${endMonth}-${endDay} ${endHours}:${endMinutes}:${endSeconds}`;

    // Find an available slot that doesn't have conflicting future bookings
    const findSlotQuery = `
      SELECT ps.id, ps.slot_number
      FROM parking_slots ps
      WHERE ps.parking_spot_id = ?
        AND ps.is_active = 1
        AND ps.is_available = 1
        AND NOT EXISTS (
          SELECT 1 FROM bookings b
          WHERE b.slot_id = ps.id
            AND b.status = 'active'
            AND (
              (b.start_at < ? AND b.expires_at > ?)
              OR (b.start_at >= ? AND b.start_at < ?)
            )
        )
      ORDER BY ps.slot_number ASC
      LIMIT 1
    `;

    db.query(
      findSlotQuery,
      [parking_id, endAtSql, startAtSql, startAtSql, endAtSql],
      (err2, slotResult) => {
        if (err2) return res.status(500).json({ error: err2.message });

        if (!slotResult.length) {
          return res.status(400).json({
            message: "No available slots found for the requested time period.",
          });
        }

        const slot = slotResult[0];
        const total_price = +(spot.price * (duration_minutes / 60)).toFixed(2);

        // Determine if this is scheduled (future) or immediate booking
        const isScheduled = startAt > now;

        if (isScheduled) {
          // SCHEDULED BOOKING - slot stays available until start_at arrives
          db.query(
            `INSERT INTO bookings 
           (user_id, parking_id, slot_id, reserved_at, start_at, expires_at, duration_minutes, total_price, status) 
           VALUES (?, ?, ?, NOW(), ?, DATE_ADD(?, INTERVAL ? MINUTE), ?, ?, 'active')`,
            [
              req.user.id,
              parking_id,
              slot.id,
              startAtSql,
              startAtSql,
              duration_minutes,
              duration_minutes,
              total_price,
            ],
            (err3, result3) => {
              if (err3) return res.status(500).json({ error: err3.message });

              const expiresAt = new Date(
                startAt.getTime() + duration_minutes * 60000,
              );

              res.status(201).json({
                message:
                  "Parking slot reserved for a future time. It will be occupied when the reservation starts.",
                booking_id: result3.insertId,
                slot_number: slot.slot_number,
                slot_id: slot.id,
                total_price,
                duration_minutes,
                start_at: startAt.toISOString(),
                expires_at: expiresAt.toISOString(),
                slots_remaining: spot.available_slots,
                is_scheduled: true,
              });
            },
          );
        } else {
          // IMMEDIATE BOOKING - slot becomes unavailable right away
          db.query(
            `INSERT INTO bookings 
           (user_id, parking_id, slot_id, reserved_at, start_at, expires_at, duration_minutes, total_price, status) 
           VALUES (?, ?, ?, NOW(), NOW(), DATE_ADD(NOW(), INTERVAL ? MINUTE), ?, ?, 'active')`,
            [
              req.user.id,
              parking_id,
              slot.id,
              duration_minutes,
              duration_minutes,
              total_price,
            ],
            (err3, result3) => {
              if (err3) return res.status(500).json({ error: err3.message });

              // Mark slot as unavailable immediately
              db.query(
                "UPDATE parking_slots SET is_available = 0 WHERE id = ?",
                [slot.id],
                (err4) => {
                  if (err4) {
                    console.error("Error updating slot availability:", err4);
                  }

                  // Update parking spot statistics
                  db.query(
                    `UPDATE parking_spots 
                   SET available_slots = available_slots - 1,
                       occupied_slots = occupied_slots + 1
                   WHERE id = ?`,
                    [parking_id],
                    (err5) => {
                      if (err5) {
                        console.error("Error updating spot statistics:", err5);
                      }

                      // Update availability table
                      db.query(
                        `UPDATE availability 
                       SET occupied_slots = occupied_slots + 1,
                           available_slots = available_slots - 1,
                           updated_at = NOW()
                       WHERE parking_id = ?`,
                        [parking_id],
                        (err6) => {
                          if (err6) {
                            console.error(
                              "Error updating availability table:",
                              err6,
                            );
                          }

                          const expiresAt = new Date(
                            Date.now() + duration_minutes * 60000,
                          );

                          res.status(201).json({
                            message: "Parking slot reserved successfully",
                            booking_id: result3.insertId,
                            slot_number: slot.slot_number,
                            slot_id: slot.id,
                            total_price,
                            duration_minutes,
                            start_at: new Date().toISOString(),
                            expires_at: expiresAt.toISOString(),
                            slots_remaining: spot.available_slots - 1,
                            is_scheduled: false,
                          });
                        },
                      );
                    },
                  );
                },
              );
            },
          );
        }
      },
    );
  });
});

// Get user's reservations with slot details
router.get("/my-reservations", auth(["user"]), (req, res) => {
  const query = `
    SELECT 
      b.id as booking_id,
      b.parking_id,
      b.slot_id,
      b.reserved_at,
      COALESCE(b.start_at, b.reserved_at) as start_at,
      b.expires_at,
      b.duration_minutes as duration_minutes,
      b.total_price,
      b.status,
      ps.name as spot_name,
      ps.latitude,
      ps.longitude,
      ps.type,
      ps.price as spot_price,
      ps.total_slots,
      ps.available_slots,
      sl.slot_number,
      CASE 
        WHEN b.status = 'active' AND COALESCE(b.start_at, b.reserved_at) > NOW() THEN 'scheduled'
        WHEN b.status = 'active' AND b.expires_at > NOW() THEN 'active'
        WHEN b.status = 'active' AND b.expires_at <= NOW() THEN 'expired'
        ELSE b.status
      END as actual_status
    FROM bookings b
    JOIN parking_spots ps ON b.parking_id = ps.id
    LEFT JOIN parking_slots sl ON b.slot_id = sl.id
    WHERE b.user_id = ?
    ORDER BY b.reserved_at DESC
  `;

  db.query(query, [req.user.id], (err, result) => {
    if (err) {
      console.error("Error fetching my-reservations:", err, "SQL:", query);
      return res.status(500).json({ error: err.message });
    }
    res.json(result);
  });
});

// Cancel reservation - only within 5 minutes of booking
router.delete("/cancel/:booking_id", auth(["user"]), (req, res) => {
  const { booking_id } = req.params;

  db.query("SELECT * FROM bookings WHERE id=?", [booking_id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });

    const booking = result[0];
    if (
      !booking ||
      booking.user_id !== req.user.id ||
      booking.status !== "active"
    ) {
      return res.status(400).json({ message: "Cannot cancel booking" });
    }

    // Determine cancellation policy:
    // - If the booking start is in the future: allow cancellation until 1 hour before start
    // - Otherwise (immediate/started bookings): allow cancellation only within 5 minutes of reservation
    const currentTime = new Date();
    const startAt = booking.start_at
      ? new Date(booking.start_at)
      : new Date(booking.reserved_at);

    if (startAt > currentTime) {
      const minutesUntilStart = (startAt - currentTime) / (1000 * 60);
      if (minutesUntilStart < 60) {
        return res.status(400).json({
          message:
            "Cancellation not allowed less than 1 hour before the scheduled start time.",
          canCancel: false,
        });
      }
      // allowed: scheduled booking and more than 1 hour before start
    } else {
      // booking already started or starts now — keep strict short cancellation window
      const bookingTime = new Date(booking.reserved_at);
      const timeDifferenceMinutes = (currentTime - bookingTime) / (1000 * 60);

      if (timeDifferenceMinutes > 5) {
        return res.status(400).json({
          message:
            "Cancellation not allowed. You can only cancel within 5 minutes of booking.",
          canCancel: false,
        });
      }
    }

    // Cancel booking
    db.query(
      "UPDATE bookings SET status='cancelled', cancelled_at=NOW() WHERE id=?",
      [booking_id],
      (err2) => {
        if (err2) return res.status(500).json({ error: err2.message });

        // Free up the slot
        if (booking.slot_id) {
          db.query(
            "UPDATE parking_slots SET is_available = 1 WHERE id = ?",
            [booking.slot_id],
            (err3) => {
              if (err3) {
                console.error("Error freeing slot:", err3);
              }

              // Update parking spot statistics
              db.query(
                `UPDATE parking_spots 
                   SET available_slots = available_slots + 1 
                   WHERE id = ?`,
                [booking.parking_id],
                (err4) => {
                  if (err4) {
                    console.error("Error updating spot statistics:", err4);
                  }

                  // Update availability table
                  db.query(
                    `UPDATE availability 
                       SET occupied_slots = occupied_slots - 1,
                           available_slots = available_slots + 1,
                           updated_at = NOW()
                       WHERE parking_id = ?`,
                    [booking.parking_id],
                    (err5) => {
                      if (err5) {
                        console.error(
                          "Error updating availability table:",
                          err5,
                        );
                      }

                      res.json({
                        message:
                          "Booking cancelled successfully. The parking slot is now available.",
                        canCancel: true,
                        slot_id: booking.slot_id,
                      });
                    },
                  );
                },
              );
            },
          );
        } else {
          res.json({
            message: "Booking cancelled successfully.",
            canCancel: true,
          });
        }
      },
    );
  });
});

// Get booking details with slot information
router.get("/:booking_id", auth(["user"]), (req, res) => {
  const { booking_id } = req.params;

  const query = `
    SELECT 
      b.id as booking_id,
      b.parking_id,
      b.slot_id,
      b.reserved_at,
      COALESCE(b.start_at, b.reserved_at) as start_at,
      b.expires_at,
      b.cancelled_at,
      b.duration_minutes as duration_minutes,
      b.total_price,
      b.status,
      ps.name as spot_name,
      ps.latitude,
      ps.longitude,
      ps.type,
      ps.price as spot_price,
      ps.total_slots,
      ps.available_slots,
      sl.slot_number,
      sl.is_active as slot_is_active,
      CASE 
        WHEN b.status = 'active' AND b.start_at > NOW() THEN 'scheduled'
        WHEN b.status = 'active' AND b.expires_at > NOW() THEN 'active'
        WHEN b.status = 'active' AND b.expires_at <= NOW() THEN 'expired'
        ELSE b.status
      END as actual_status
    FROM bookings b
    JOIN parking_spots ps ON b.parking_id = ps.id
    LEFT JOIN parking_slots sl ON b.slot_id = sl.id
    WHERE b.id = ? AND b.user_id = ?
  `;

  db.query(query, [booking_id, req.user.id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!result.length)
      return res.status(404).json({ message: "Booking not found" });
    res.json(result[0]);
  });
});

// Get real-time slot availability for a parking spot
router.get(
  "/spot/:parking_id/availability",
  auth(["user", "owner"]),
  (req, res) => {
    const { parking_id } = req.params;

    const query = `
    SELECT 
      ps.id as parking_spot_id,
      ps.name,
      ps.total_slots,
      ps.active_slots,
      ps.available_slots,
      COALESCE(a.occupied_slots, 0) as occupied_slots,
      (
        SELECT COUNT(*) 
        FROM parking_slots 
        WHERE parking_spot_id = ps.id 
          AND is_active = 1 
          AND is_available = 1
      ) as real_time_available_slots,
      (
        SELECT JSON_ARRAYAGG(
          JSON_OBJECT(
            'slot_id', id,
            'slot_number', slot_number,
            'is_active', is_active,
            'is_available', is_available,
            'status', CASE 
              WHEN is_active = 0 THEN 'disabled'
              WHEN is_available = 0 THEN 'occupied'
              ELSE 'available'
            END
          )
        )
        FROM parking_slots
        WHERE parking_spot_id = ps.id
        ORDER BY slot_number
      ) as slots_detail
    FROM parking_spots ps
    LEFT JOIN availability a ON ps.id = a.parking_id
    WHERE ps.id = ?
  `;

    db.query(query, [parking_id], (err, result) => {
      if (err) return res.status(500).json({ error: err.message });
      if (!result.length)
        return res.status(404).json({ message: "Parking spot not found" });

      const spot = result[0];
      // Parse JSON string if necessary
      if (typeof spot.slots_detail === "string") {
        spot.slots_detail = JSON.parse(spot.slots_detail);
      }

      res.json(spot);
    });
  },
);

// Get active bookings for a specific parking spot (OWNER endpoint)
router.get("/spot/:parking_id/active-bookings", auth(["owner"]), (req, res) => {
  const { parking_id } = req.params;

  // Verify ownership
  db.query(
    "SELECT owner_id FROM parking_spots WHERE id = ?",
    [parking_id],
    (err, ownerCheck) => {
      if (err) return res.status(500).json({ error: err.message });
      if (!ownerCheck.length)
        return res.status(404).json({ message: "Parking spot not found" });

      if (ownerCheck[0].owner_id !== req.user.id) {
        return res.status(403).json({ message: "Not authorized" });
      }

      const query = `
        SELECT 
          b.id as booking_id,
          b.reserved_at,
          COALESCE(b.start_at, b.reserved_at) as start_at,
          b.expires_at,
          b.duration_minutes as duration_minutes,
          b.total_price,
          b.status,
          sl.id as slot_id,
          sl.slot_number,
          u.id as user_id,
          u.name as user_name,
          u.email as user_email
        FROM bookings b
        JOIN parking_slots sl ON b.slot_id = sl.id
        JOIN users u ON b.user_id = u.id
        WHERE b.parking_id = ? 
          AND b.status = 'active'
        ORDER BY sl.slot_number ASC
      `;

      db.query(query, [parking_id], (err2, result) => {
        if (err2) return res.status(500).json({ error: err2.message });
        res.json(result);
      });
    },
  );
});

// Get booking schedule/calendar for a parking spot (PUBLIC - for users to see availability)
router.get("/spot/:parking_id/schedule", (req, res) => {
  const { parking_id } = req.params;
  const { days = 7 } = req.query; // Default to next 7 days

  const query = `
    SELECT 
      DATE(b.start_at) as booking_date,
      HOUR(b.start_at) as booking_hour,
      COUNT(DISTINCT b.slot_id) as slots_booked,
      ps.total_slots,
      ps.active_slots,
      GROUP_CONCAT(
        CONCAT(
          TIME_FORMAT(b.start_at, '%H:%i'), 
          '-', 
          TIME_FORMAT(b.expires_at, '%H:%i'),
          ' (Slot ', sl.slot_number, ')'
        ) 
        ORDER BY b.start_at 
        SEPARATOR '; '
      ) as booking_details
    FROM bookings b
    JOIN parking_slots sl ON b.slot_id = sl.id
    JOIN parking_spots ps ON b.parking_id = ps.id
    WHERE b.parking_id = ?
      AND b.status = 'active'
      AND DATE(b.start_at) BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL ? DAY)
    GROUP BY DATE(b.start_at), HOUR(b.start_at), ps.total_slots, ps.active_slots
    ORDER BY booking_date ASC, booking_hour ASC
  `;

  db.query(query, [parking_id, days], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });

    // Get spot info
    db.query(
      "SELECT id, name, total_slots, active_slots, available_slots, price FROM parking_spots WHERE id = ?",
      [parking_id],
      (err2, spotInfo) => {
        if (err2) return res.status(500).json({ error: err2.message });
        if (!spotInfo.length) {
          return res.status(404).json({ message: "Parking spot not found" });
        }

        res.json({
          spot: spotInfo[0],
          schedule: result,
        });
      },
    );
  });
});

module.exports = router;
