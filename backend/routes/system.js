const express = require("express");
const router = express.Router();
const db = require("../config/db");

router.post("/recompute-availability", (req, res) => {
  // Step 1: For active bookings that start in the future, ensure their slots remain available
  const futureBookingsQuery = `
    SELECT DISTINCT slot_id
    FROM bookings
    WHERE status = 'active' AND start_at > NOW() AND slot_id IS NOT NULL
  `;

  db.query(futureBookingsQuery, (err, futureRows) => {
    if (err) return res.status(500).json({ error: err.message });

    const futureSlotIds = futureRows.map((r) => r.slot_id).filter(Boolean);

    const tasks = [];

    // If any slots were incorrectly marked unavailable, mark them available
    if (futureSlotIds.length) {
      tasks.push(
        new Promise((resolve) => {
          db.query(
            "UPDATE parking_slots SET is_available = 1 WHERE id IN (?)",
            [futureSlotIds],
            (err2) => {
              if (err2) console.error("Error restoring future slots:", err2);
              resolve();
            },
          );
        }),
      );
    }

    // Step 2: Recompute per-spot aggregates from parking_slots
    tasks.push(
      new Promise((resolve) => {
        const recomputeQuery = `
          UPDATE parking_spots ps
          SET
            ps.available_slots = (
              SELECT COUNT(*) FROM parking_slots sl WHERE sl.parking_spot_id = ps.id AND sl.is_active = 1 AND sl.is_available = 1
            ),
            ps.occupied_slots = (
              SELECT COUNT(*) FROM parking_slots sl WHERE sl.parking_spot_id = ps.id AND sl.is_active = 1 AND sl.is_available = 0
            )
        `;

        db.query(recomputeQuery, (err3) => {
          if (err3)
            console.error("Error recomputing parking_spots stats:", err3);
          resolve();
        });
      }),
    );

    // Step 3: Sync availability table to parking_spots
    tasks.push(
      new Promise((resolve) => {
        const syncAvailability = `
          UPDATE availability a
          JOIN parking_spots ps ON a.parking_id = ps.id
          SET a.available_slots = ps.available_slots,
              a.occupied_slots = ps.occupied_slots,
              a.updated_at = NOW()
        `;

        db.query(syncAvailability, (err4) => {
          if (err4) console.error("Error syncing availability table:", err4);
          resolve();
        });
      }),
    );

    Promise.all(tasks)
      .then(() => res.json({ message: "Recomputed availability successfully" }))
      .catch((e) => res.status(500).json({ error: e.message }));
  });
});

// Activate scheduled bookings that are now due to start
router.post("/activate-scheduled-bookings", (req, res) => {
  const findScheduledQuery = `
    SELECT b.id, b.slot_id, b.parking_id, b.start_at
    FROM bookings b
    WHERE b.status = 'active' 
      AND b.start_at IS NOT NULL 
      AND b.start_at <= NOW()
      AND b.expires_at > NOW()
  `;

  db.query(findScheduledQuery, (err, scheduledBookings) => {
    if (err) return res.status(500).json({ error: err.message });

    if (scheduledBookings.length === 0) {
      return res.json({
        message: "No scheduled bookings to activate",
        activated_count: 0,
      });
    }

    const slotIds = scheduledBookings.map((b) => b.slot_id).filter(Boolean);
    const parkingIds = [...new Set(scheduledBookings.map((b) => b.parking_id))];

    // Mark slots as unavailable
    if (slotIds.length > 0) {
      db.query(
        "UPDATE parking_slots SET is_available = 0 WHERE id IN (?)",
        [slotIds],
        (err2) => {
          if (err2) console.error("Error marking slots unavailable:", err2);

          // Update parking spot statistics
          parkingIds.forEach((parkingId) => {
            db.query(
              `UPDATE parking_spots 
               SET available_slots = available_slots - 1,
                   occupied_slots = occupied_slots + 1
               WHERE id = ?`,
              [parkingId],
            );

            db.query(
              `UPDATE availability 
               SET occupied_slots = occupied_slots + 1,
                   available_slots = available_slots - 1,
                   updated_at = NOW()
               WHERE parking_id = ?`,
              [parkingId],
            );
          });

          res.json({
            message: "Activated scheduled bookings",
            activated_count: scheduledBookings.length,
            slot_ids: slotIds,
          });
        },
      );
    } else {
      res.json({
        message: "No slots to activate",
        activated_count: 0,
      });
    }
  });
});
// One-time force restore for a single future booking (admin tool)
router.post("/force-restore-booking/:booking_id", (req, res) => {
  const { booking_id } = req.params;

  db.query(
    "SELECT id, slot_id, parking_id, start_at, reserved_at, status FROM bookings WHERE id = ?",
    [booking_id],
    (err, rows) => {
      if (err) return res.status(500).json({ error: err.message });
      if (!rows.length)
        return res.status(404).json({ message: "Booking not found" });

      const b = rows[0];
      if (!b.slot_id)
        return res
          .status(400)
          .json({ message: "Booking has no assigned slot" });
      if (b.status !== "active")
        return res.status(400).json({ message: "Booking is not active" });

      const startAt = b.start_at
        ? new Date(b.start_at)
        : new Date(b.reserved_at);
      const now = new Date();
      if (startAt <= now) {
        return res.status(400).json({
          message:
            "Booking already started or in the past; cannot force-restore",
        });
      }

      // Restore the slot availability and recompute aggregates
      db.query(
        "UPDATE parking_slots SET is_available = 1 WHERE id = ?",
        [b.slot_id],
        (err2) => {
          if (err2) return res.status(500).json({ error: err2.message });

          // Recompute spot stats and availability similar to recompute-availability
          db.query(
            `
                UPDATE parking_spots ps
                SET
                  ps.available_slots = (
                    SELECT COUNT(*) FROM parking_slots sl WHERE sl.parking_spot_id = ps.id AND sl.is_active = 1 AND sl.is_available = 1
                  ),
                  ps.occupied_slots = (
                    SELECT COUNT(*) FROM parking_slots sl WHERE sl.parking_spot_id = ps.id AND sl.is_active = 1 AND sl.is_available = 0
                  )
                WHERE ps.id = ?
                `,
            [b.parking_id],
            (err3) => {
              if (err3)
                console.error("Error recomputing parking_spots stats:", err3);

              db.query(
                `
                    UPDATE availability a
                    JOIN parking_spots ps ON a.parking_id = ps.id
                    SET a.available_slots = ps.available_slots,
                        a.occupied_slots = ps.occupied_slots,
                        a.updated_at = NOW()
                    WHERE a.parking_id = ?
                    `,
                [b.parking_id],
                (err4) => {
                  if (err4)
                    console.error("Error syncing availability table:", err4);
                  return res.json({
                    message: "Forced restore applied for booking",
                    booking_id,
                    slot_id: b.slot_id,
                  });
                },
              );
            },
          );
        },
      );
    },
  );
});

// Refresh expired bookings
router.post("/refresh-availability", (req, res) => {
  // First, find all expired bookings with their slot info
  const findExpiredQuery = `
    SELECT b.id, b.slot_id, b.parking_id 
    FROM bookings b
    WHERE b.status = 'active' AND b.expires_at < NOW()
  `;

  db.query(findExpiredQuery, (err, expiredBookings) => {
    if (err) return res.status(500).json({ error: err.message });

    if (expiredBookings.length === 0) {
      return res.json({
        message: "No expired bookings to refresh",
        expired_bookings: 0,
      });
    }

    // Update bookings to expired
    const expireQuery = `
      UPDATE bookings 
      SET status = 'expired' 
      WHERE status = 'active' AND expires_at < NOW()
    `;

    db.query(expireQuery, (err2, result) => {
      if (err2) return res.status(500).json({ error: err2.message });

      // Free up the slots
      const slotIds = expiredBookings.map((b) => b.slot_id).filter(Boolean);

      if (slotIds.length > 0) {
        db.query(
          "UPDATE parking_slots SET is_available = 1 WHERE id IN (?)",
          [slotIds],
          (err3) => {
            if (err3) console.error("Error freeing slots:", err3);

            // Update parking_spots statistics
            const parkingIds = [
              ...new Set(expiredBookings.map((b) => b.parking_id)),
            ];

            parkingIds.forEach((parkingId) => {
              db.query(
                `UPDATE parking_spots 
                 SET available_slots = (
                   SELECT COUNT(*) FROM parking_slots 
                   WHERE parking_spot_id = ? AND is_active = 1 AND is_available = 1
                 )
                 WHERE id = ?`,
                [parkingId, parkingId],
              );

              db.query(
                `UPDATE availability 
                 SET occupied_slots = (
                   SELECT COUNT(*) FROM bookings 
                   WHERE parking_id = ? AND status = 'active'
                 ),
                 available_slots = (
                   SELECT COUNT(*) FROM parking_slots 
                   WHERE parking_spot_id = ? AND is_active = 1 AND is_available = 1
                 ),
                 updated_at = NOW()
                 WHERE parking_id = ?`,
                [parkingId, parkingId, parkingId],
              );
            });

            res.json({
              message: "System refresh completed",
              expired_bookings: result.affectedRows,
              slots_freed: slotIds.length,
            });
          },
        );
      } else {
        res.json({
          message: "System refresh completed",
          expired_bookings: result.affectedRows,
          slots_freed: 0,
        });
      }
    });
  });
});

// Dashboard stats - Updated for multi-slot system
router.get("/stats", (req, res) => {
  const statsQuery = `
    SELECT 
      COUNT(DISTINCT ps.id) as total_spots,
      COALESCE(SUM(ps.total_slots), 0) as total_slots,
      COALESCE(SUM(ps.active_slots), 0) as active_slots,
      COALESCE(SUM(ps.available_slots), 0) as available_spots,
      COALESCE(SUM(a.occupied_slots), 0) as reserved_spots,
      COALESCE(SUM(CASE WHEN ps.active_slots = 0 THEN ps.total_slots ELSE 0 END), 0) as unavailable_spots,
      (
        SELECT COUNT(*) 
        FROM bookings 
        WHERE status = 'active' AND expires_at > NOW()
      ) as active_bookings,
      (
        SELECT COUNT(*) 
        FROM bookings 
        WHERE status = 'active' AND expires_at <= NOW()
      ) as expired_bookings_count,
      (
        SELECT COUNT(*) 
        FROM bookings
      ) as total_bookings
    FROM parking_spots ps
    LEFT JOIN availability a ON ps.id = a.parking_id
  `;

  db.query(statsQuery, (err, result) => {
    if (err) {
      console.error("System stats error:", err);
      return res.status(500).json({ error: err.message });
    }

    if (!result.length) {
      return res.json({
        total_spots: 0,
        total_slots: 0,
        active_slots: 0,
        available_spots: 0,
        reserved_spots: 0,
        unavailable_spots: 0,
        active_bookings: 0,
        expired_bookings_count: 0,
        total_bookings: 0,
      });
    }

    res.json(result[0]);
  });
});

// Get owner-specific statistics
router.get("/owner/stats/:owner_id", (req, res) => {
  const { owner_id } = req.params;

  const ownerStatsQuery = `
    SELECT 
      COUNT(DISTINCT ps.id) as total_spots,
      COALESCE(SUM(ps.total_slots), 0) as total_slots,
      COALESCE(SUM(ps.active_slots), 0) as active_slots,
      COALESCE(SUM(ps.available_slots), 0) as available_spots,
      COALESCE(SUM(a.occupied_slots), 0) as reserved_spots,
      COALESCE(SUM(CASE WHEN ps.active_slots = 0 THEN 1 ELSE 0 END), 0) as unavailable_spots,
      (
        SELECT COUNT(*) 
        FROM bookings b
        JOIN parking_spots ps2 ON b.parking_id = ps2.id
        WHERE ps2.owner_id = ? AND b.status = 'active' AND b.expires_at > NOW()
      ) as active_bookings,
      (
        SELECT COALESCE(SUM(b.total_price), 0)
        FROM bookings b
        JOIN parking_spots ps2 ON b.parking_id = ps2.id
        WHERE ps2.owner_id = ? AND b.status IN ('active', 'completed')
      ) as total_revenue
    FROM parking_spots ps
    LEFT JOIN availability a ON ps.id = a.parking_id
    WHERE ps.owner_id = ?
  `;

  db.query(ownerStatsQuery, [owner_id, owner_id, owner_id], (err, result) => {
    if (err) {
      console.error("Owner stats error:", err);
      return res.status(500).json({ error: err.message });
    }

    if (!result.length) {
      return res.json({
        total_spots: 0,
        total_slots: 0,
        active_slots: 0,
        available_spots: 0,
        reserved_spots: 0,
        unavailable_spots: 0,
        active_bookings: 0,
        total_revenue: 0,
      });
    }

    res.json(result[0]);
  });
});

module.exports = router;
