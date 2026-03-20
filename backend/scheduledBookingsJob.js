// scheduledBookingsJob.js - Background job to manage scheduled bookings
const db = require("./config/db");

/**
 * Activate scheduled bookings that have reached their start time
 * Marks slots as unavailable when booking time arrives
 */
function activateScheduledBookings() {
  const query = `
    SELECT 
      b.id as booking_id,
      b.slot_id,
      b.parking_id,
      b.start_at,
      ps.slot_number
    FROM bookings b
    JOIN parking_slots ps ON b.slot_id = ps.id
    WHERE b.status = 'active'
      AND b.start_at <= NOW()
      AND b.start_at > b.reserved_at
      AND ps.is_available = 1
  `;

  db.query(query, (err, bookings) => {
    if (err) {
      console.error("❌ Error fetching scheduled bookings:", err);
      return;
    }

    if (bookings.length === 0) {
      return; // No bookings to activate
    }

    console.log(
      `🔄 Activating ${bookings.length} scheduled booking(s) that have reached their start time...`,
    );

    bookings.forEach((booking) => {
      // Mark slot as unavailable
      db.query(
        "UPDATE parking_slots SET is_available = 0 WHERE id = ?",
        [booking.slot_id],
        (err1) => {
          if (err1) {
            console.error(
              `❌ Error marking slot ${booking.slot_number} unavailable:`,
              err1,
            );
            return;
          }

          // Update parking spot statistics
          db.query(
            `UPDATE parking_spots 
             SET available_slots = available_slots - 1,
                 occupied_slots = occupied_slots + 1
             WHERE id = ?`,
            [booking.parking_id],
            (err2) => {
              if (err2) {
                console.error(
                  `❌ Error updating spot statistics for booking ${booking.booking_id}:`,
                  err2,
                );
                return;
              }

              // Update availability table
              db.query(
                `UPDATE availability 
                 SET occupied_slots = occupied_slots + 1,
                     available_slots = available_slots - 1,
                     updated_at = NOW()
                 WHERE parking_id = ?`,
                [booking.parking_id],
                (err3) => {
                  if (err3) {
                    console.error(
                      `❌ Error updating availability for booking ${booking.booking_id}:`,
                      err3,
                    );
                  } else {
                    console.log(
                      `✅ Activated booking ${booking.booking_id} - Slot ${booking.slot_number} now marked as occupied`,
                    );
                  }
                },
              );
            },
          );
        },
      );
    });
  });
}

/**
 * Expire bookings that have passed their expiration time
 * Frees up slots and updates statistics
 */
function expireBookings() {
  const query = `
    SELECT 
      b.id as booking_id,
      b.slot_id,
      b.parking_id,
      b.expires_at,
      ps.slot_number
    FROM bookings b
    JOIN parking_slots ps ON b.slot_id = ps.id
    WHERE b.status = 'active'
      AND b.expires_at <= NOW()
  `;

  db.query(query, (err, expiredBookings) => {
    if (err) {
      console.error("❌ Error fetching expired bookings:", err);
      return;
    }

    if (expiredBookings.length === 0) {
      return; // No expired bookings
    }

    console.log(
      `🔄 Expiring ${expiredBookings.length} booking(s) that have passed their expiration time...`,
    );

    expiredBookings.forEach((booking) => {
      // Update booking status to expired
      db.query(
        "UPDATE bookings SET status = 'expired' WHERE id = ?",
        [booking.booking_id],
        (err1) => {
          if (err1) {
            console.error(
              `❌ Error expiring booking ${booking.booking_id}:`,
              err1,
            );
            return;
          }

          // Mark slot as available
          db.query(
            "UPDATE parking_slots SET is_available = 1 WHERE id = ?",
            [booking.slot_id],
            (err2) => {
              if (err2) {
                console.error(
                  `❌ Error freeing slot ${booking.slot_number}:`,
                  err2,
                );
                return;
              }

              // Update parking spot statistics
              db.query(
                `UPDATE parking_spots 
                 SET available_slots = available_slots + 1,
                     occupied_slots = GREATEST(occupied_slots - 1, 0)
                 WHERE id = ?`,
                [booking.parking_id],
                (err3) => {
                  if (err3) {
                    console.error(
                      `❌ Error updating spot statistics for expired booking ${booking.booking_id}:`,
                      err3,
                    );
                    return;
                  }

                  // Update availability table
                  db.query(
                    `UPDATE availability 
                     SET occupied_slots = GREATEST(occupied_slots - 1, 0),
                         available_slots = available_slots + 1,
                         updated_at = NOW()
                     WHERE parking_id = ?`,
                    [booking.parking_id],
                    (err4) => {
                      if (err4) {
                        console.error(
                          `❌ Error updating availability for expired booking ${booking.booking_id}:`,
                          err4,
                        );
                      } else {
                        console.log(
                          `✅ Expired booking ${booking.booking_id} - Slot ${booking.slot_number} now available`,
                        );
                      }
                    },
                  );
                },
              );
            },
          );
        },
      );
    });
  });
}

/**
 * Run both jobs - activate scheduled bookings and expire finished ones
 */
function runBookingMaintenanceJobs() {
  console.log("⏰ Running booking maintenance jobs...");
  activateScheduledBookings();
  expireBookings();
}

// Export for use in server.js
module.exports = {
  activateScheduledBookings,
  expireBookings,
  runBookingMaintenanceJobs,
};

// Run immediately if executed directly
if (require.main === module) {
  console.log("🚀 Starting scheduled bookings job...");
  runBookingMaintenanceJobs();

  // Run every minute
  setInterval(runBookingMaintenanceJobs, 60000);
}
