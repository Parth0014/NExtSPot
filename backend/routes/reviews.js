// routes/reviews.js - Fully Fixed for ai_parking Database Schema
const express = require("express");
const router = express.Router();
const db = require("../config/db");
const auth = require("../middleware/authMiddleware");

// ================= CONSTANTS =================
const MIN_RATING = 1;
const MAX_RATING = 5;
const MAX_COMMENT_LENGTH = 1000;
const MAX_RESPONSE_LENGTH = 500;

// ================= VALIDATION HELPERS =================

const validateRating = (rating) =>
  rating &&
  Number.isInteger(rating) &&
  rating >= MIN_RATING &&
  rating <= MAX_RATING;

const validateReviewData = (req, res, next) => {
  const {
    rating,
    comment,
    cleanliness_rating,
    safety_rating,
    accessibility_rating,
  } = req.body;

  if (!validateRating(rating)) {
    return res.status(400).json({
      message: `Rating is required and must be between ${MIN_RATING} and ${MAX_RATING}`,
    });
  }

  const optionalRatings = [
    cleanliness_rating,
    safety_rating,
    accessibility_rating,
  ];
  for (let r of optionalRatings) {
    if (r !== undefined && r !== null && !validateRating(r)) {
      return res.status(400).json({
        message: `All ratings must be between ${MIN_RATING} and ${MAX_RATING}`,
      });
    }
  }

  if (comment && comment.length > MAX_COMMENT_LENGTH) {
    return res.status(400).json({
      message: `Comment must not exceed ${MAX_COMMENT_LENGTH} characters`,
    });
  }

  next();
};

// ================= USER ENDPOINTS =================

// Get reviews for a specific parking spot (public)
router.get("/spot/:parkingId", (req, res) => {
  const { parkingId } = req.params;
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 10;
  const offset = (page - 1) * limit;

  if (!parkingId || isNaN(parkingId)) {
    return res.status(400).json({ message: "Invalid parking spot ID" });
  }

  const query = `
    SELECT 
      r.id,
      r.rating,
      r.comment,
      r.cleanliness_rating,
      r.safety_rating,
      r.accessibility_rating,
      r.created_at,
      r.updated_at,
      u.name AS user_name,
      COALESCE(SUM(CASE WHEN rv.is_helpful = 1 THEN 1 ELSE 0 END), 0) AS helpful_count,
      COALESCE(SUM(CASE WHEN rv.is_helpful = 0 THEN 1 ELSE 0 END), 0) AS not_helpful_count,
      rr.response_text AS owner_response,
      rr.created_at AS response_created_at,
      ou.name AS owner_name
    FROM reviews r
    JOIN users u ON r.user_id = u.id
    LEFT JOIN review_votes rv ON r.id = rv.review_id
    LEFT JOIN review_responses rr ON r.id = rr.review_id
    LEFT JOIN users ou ON rr.owner_id = ou.id
    WHERE r.parking_id = ?
    GROUP BY r.id
    ORDER BY r.created_at DESC
    LIMIT ? OFFSET ?;
  `;

  db.query(query, [parkingId, limit, offset], (err, reviews) => {
    if (err) {
      console.error("Error fetching reviews:", err);
      return res.status(500).json({ message: "Error fetching reviews" });
    }

    db.query(
      "SELECT COUNT(*) AS total FROM reviews WHERE parking_id = ?",
      [parkingId],
      (err2, countResult) => {
        if (err2) {
          console.error("Error counting reviews:", err2);
          return res.status(500).json({ message: "Error counting reviews" });
        }

        res.json({
          reviews,
          pagination: {
            page,
            limit,
            total: countResult[0].total,
            totalPages: Math.ceil(countResult[0].total / limit),
          },
        });
      },
    );
  });
});

// Get average ratings for a parking spot
router.get("/spot/:parkingId/summary", (req, res) => {
  const { parkingId } = req.params;

  if (!parkingId || isNaN(parkingId)) {
    return res.status(400).json({ message: "Invalid parking spot ID" });
  }

  const query = `
    SELECT 
      COUNT(*) AS total_reviews,
      COALESCE(AVG(rating), 0) AS average_rating,
      COALESCE(AVG(cleanliness_rating), 0) AS avg_cleanliness,
      COALESCE(AVG(safety_rating), 0) AS avg_safety,
      COALESCE(AVG(accessibility_rating), 0) AS avg_accessibility,
      SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) AS five_star,
      SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) AS four_star,
      SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) AS three_star,
      SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) AS two_star,
      SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) AS one_star
    FROM reviews
    WHERE parking_id = ?;
  `;

  db.query(query, [parkingId], (err, result) => {
    if (err) {
      console.error("Error fetching summary:", err);
      return res.status(500).json({ message: "Error fetching review summary" });
    }
    res.json(result[0] || {});
  });
});

// Submit a review (only for expired bookings)
router.post("/submit", auth(["user"]), validateReviewData, (req, res) => {
  const {
    parking_id,
    booking_id,
    rating,
    comment,
    cleanliness_rating,
    safety_rating,
    accessibility_rating,
  } = req.body;

  if (!booking_id) {
    return res.status(400).json({
      message:
        "Booking ID is required. You can only review after completing a reservation.",
    });
  }

  if (!parking_id || isNaN(parking_id)) {
    return res.status(400).json({ message: "Invalid parking spot ID" });
  }

  // ✅ Fix for ai_parking: Use expires_at + 'expired'
  const verifyBookingQuery = `
    SELECT b.id, b.status, b.expires_at
    FROM bookings b
    WHERE b.id = ?
      AND b.user_id = ?
      AND b.parking_id = ?
      AND b.status = 'expired'
      AND b.expires_at < NOW();
  `;

  db.query(
    verifyBookingQuery,
    [booking_id, req.user.id, parking_id],
    (err, bookingResult) => {
      if (err) {
        console.error("Error verifying booking:", err);
        return res.status(500).json({ message: "Error verifying booking" });
      }

      if (!bookingResult.length) {
        return res.status(403).json({
          message:
            "Invalid booking. You can only review expired reservations that have ended.",
        });
      }

      db.query(
        "SELECT id FROM reviews WHERE user_id = ? AND parking_id = ?",
        [req.user.id, parking_id],
        (err2, existingReview) => {
          if (err2) {
            console.error("Error checking existing review:", err2);
            return res
              .status(500)
              .json({ message: "Error checking existing review" });
          }

          if (existingReview.length > 0) {
            return res.status(400).json({
              message: "You have already reviewed this parking spot.",
            });
          }

          const insertQuery = `
            INSERT INTO reviews 
            (user_id, parking_id, booking_id, rating, comment, cleanliness_rating, safety_rating, accessibility_rating, is_verified_booking)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1);
          `;

          db.query(
            insertQuery,
            [
              req.user.id,
              parking_id,
              booking_id,
              rating,
              comment || null,
              cleanliness_rating || null,
              safety_rating || null,
              accessibility_rating || null,
            ],
            (err3, insertResult) => {
              if (err3) {
                console.error("Error inserting review:", err3);
                return res
                  .status(500)
                  .json({ message: "Error submitting review" });
              }

              res.status(201).json({
                message: "Review submitted successfully",
                review_id: insertResult.insertId,
              });
            },
          );
        },
      );
    },
  );
});

// Update a review
router.put("/:reviewId", auth(["user"]), validateReviewData, (req, res) => {
  const { reviewId } = req.params;
  const {
    rating,
    comment,
    cleanliness_rating,
    safety_rating,
    accessibility_rating,
  } = req.body;

  if (!reviewId || isNaN(reviewId)) {
    return res.status(400).json({ message: "Invalid review ID" });
  }

  db.query(
    "SELECT user_id FROM reviews WHERE id = ?",
    [reviewId],
    (err, result) => {
      if (err) {
        console.error("Error fetching review:", err);
        return res.status(500).json({ message: "Error fetching review" });
      }

      if (!result.length) {
        return res.status(404).json({ message: "Review not found" });
      }

      if (result[0].user_id !== req.user.id) {
        return res
          .status(403)
          .json({ message: "You can only update your own reviews" });
      }

      const updateQuery = `
      UPDATE reviews 
      SET rating = ?, 
          comment = ?, 
          cleanliness_rating = ?, 
          safety_rating = ?, 
          accessibility_rating = ?,
          updated_at = NOW()
      WHERE id = ?;
    `;

      db.query(
        updateQuery,
        [
          rating,
          comment || null,
          cleanliness_rating || null,
          safety_rating || null,
          accessibility_rating || null,
          reviewId,
        ],
        (err2) => {
          if (err2) {
            console.error("Error updating review:", err2);
            return res.status(500).json({ message: "Error updating review" });
          }
          res.json({ message: "Review updated successfully" });
        },
      );
    },
  );
});

// Delete a review
router.delete("/:reviewId", auth(["user"]), (req, res) => {
  const { reviewId } = req.params;

  if (!reviewId || isNaN(reviewId)) {
    return res.status(400).json({ message: "Invalid review ID" });
  }

  db.query(
    "SELECT user_id FROM reviews WHERE id = ?",
    [reviewId],
    (err, result) => {
      if (err) {
        console.error("Error fetching review:", err);
        return res.status(500).json({ message: "Error fetching review" });
      }

      if (!result.length) {
        return res.status(404).json({ message: "Review not found" });
      }

      if (result[0].user_id !== req.user.id) {
        return res
          .status(403)
          .json({ message: "You can only delete your own reviews" });
      }

      db.query("DELETE FROM reviews WHERE id = ?", [reviewId], (err2) => {
        if (err2) {
          console.error("Error deleting review:", err2);
          return res.status(500).json({ message: "Error deleting review" });
        }
        res.json({ message: "Review deleted successfully" });
      });
    },
  );
});

// Vote on review helpfulness
router.post("/:reviewId/vote", auth(["user"]), (req, res) => {
  const { reviewId } = req.params;
  const { is_helpful } = req.body;

  if (!reviewId || isNaN(reviewId)) {
    return res.status(400).json({ message: "Invalid review ID" });
  }

  if (typeof is_helpful !== "boolean") {
    return res.status(400).json({
      message:
        "is_helpful must be a boolean (true for helpful, false for not helpful)",
    });
  }

  db.query("SELECT id FROM reviews WHERE id = ?", [reviewId], (err, result) => {
    if (err) {
      console.error("Error checking review:", err);
      return res.status(500).json({ message: "Error checking review" });
    }

    if (!result.length) {
      return res.status(404).json({ message: "Review not found" });
    }

    const query = `
      INSERT INTO review_votes (review_id, user_id, is_helpful)
      VALUES (?, ?, ?)
      ON DUPLICATE KEY UPDATE is_helpful = VALUES(is_helpful);
    `;

    db.query(query, [reviewId, req.user.id, is_helpful ? 1 : 0], (err2) => {
      if (err2) {
        console.error("Error recording vote:", err2);
        return res.status(500).json({ message: "Error recording vote" });
      }

      res.json({ message: "Vote recorded successfully" });
    });
  });
});

// User's own reviews (removed ps.address)
router.get("/my-reviews", auth(["user"]), (req, res) => {
  const query = `
    SELECT 
      r.id,
      r.rating,
      r.comment,
      r.cleanliness_rating,
      r.safety_rating,
      r.accessibility_rating,
      r.created_at,
      r.updated_at,
      ps.name AS spot_name,
      ps.id AS parking_id,
      rr.response_text AS owner_response,
      rr.created_at AS response_date,
      ou.name AS owner_name
    FROM reviews r
    JOIN parking_spots ps ON r.parking_id = ps.id
    LEFT JOIN review_responses rr ON r.id = rr.review_id
    LEFT JOIN users ou ON rr.owner_id = ou.id
    WHERE r.user_id = ?
    ORDER BY r.created_at DESC;
  `;

  db.query(query, [req.user.id], (err, result) => {
    if (err) {
      console.error("Error fetching user reviews:", err);
      return res.status(500).json({ message: "Error fetching your reviews" });
    }

    res.json(result);
  });
});

// Check if user can review a spot
router.get("/can-review/:parkingId", auth(["user"]), (req, res) => {
  const { parkingId } = req.params;

  if (!parkingId || isNaN(parkingId)) {
    return res.status(400).json({ message: "Invalid parking spot ID" });
  }

  db.query(
    "SELECT id FROM reviews WHERE user_id = ? AND parking_id = ?",
    [req.user.id, parkingId],
    (err, existingReview) => {
      if (err) {
        console.error("Error checking review:", err);
        return res
          .status(500)
          .json({ message: "Error checking review status" });
      }

      if (existingReview.length > 0) {
        return res.json({
          canReview: false,
          reason: "already_reviewed",
          message: "You have already reviewed this parking spot",
        });
      }

      const bookingQuery = `
        SELECT b.id, b.expires_at
        FROM bookings b
        WHERE b.user_id = ?
          AND b.parking_id = ?
          AND b.status = 'expired'
          AND b.expires_at < NOW()
        ORDER BY b.expires_at DESC
        LIMIT 1;
      `;

      db.query(bookingQuery, [req.user.id, parkingId], (err2, bookings) => {
        if (err2) {
          console.error("Error checking bookings:", err2);
          return res
            .status(500)
            .json({ message: "Error checking booking status" });
        }

        if (!bookings.length) {
          return res.json({
            canReview: false,
            reason: "no_completed_booking",
            message:
              "You need to complete a reservation before reviewing this spot",
          });
        }

        res.json({
          canReview: true,
          booking_id: bookings[0].id,
          message: "You can review this parking spot",
        });
      });
    },
  );
});

// ================= OWNER ENDPOINTS =================

// Get owner's parking spots (for dropdown filter)
router.get("/owner/spots", auth(["owner"]), (req, res) => {
  const query = `
    SELECT id, name
    FROM parking_spots
    WHERE owner_id = ?
    ORDER BY name ASC;
  `;

  db.query(query, [req.user.id], (err, spots) => {
    if (err) {
      console.error("Error fetching owner spots:", err);
      return res.status(500).json({ message: "Error fetching parking spots" });
    }
    res.json(spots);
  });
});

router.get("/owner/reviews", auth(["owner"]), (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const limit = parseInt(req.query.limit) || 20;
  const offset = (page - 1) * limit;
  const spotId = req.query.spot_id;

  // Build dynamic query based on whether spot_id is provided
  let query = `
    SELECT 
      r.id,
      r.rating,
      r.comment,
      r.cleanliness_rating,
      r.safety_rating,
      r.accessibility_rating,
      r.created_at,
      r.updated_at,
      u.name AS user_name,
      ps.name AS spot_name,
      ps.id AS parking_id,
      rr.response_text AS my_response,
      rr.created_at AS response_date
    FROM reviews r
    JOIN parking_spots ps ON r.parking_id = ps.id
    JOIN users u ON r.user_id = u.id
    LEFT JOIN review_responses rr ON r.id = rr.review_id
    WHERE ps.owner_id = ?
  `;

  // Add spot filter if spot_id is provided and not "all"
  const queryParams = [req.user.id];
  if (spotId && spotId !== "all") {
    query += ` AND ps.id = ?`;
    queryParams.push(spotId);
  }

  query += ` ORDER BY r.created_at DESC LIMIT ? OFFSET ?`;
  queryParams.push(limit, offset);

  db.query(query, queryParams, (err, reviews) => {
    if (err) {
      console.error("Error fetching owner reviews:", err);
      return res.status(500).json({ message: "Error fetching reviews" });
    }

    // Count query with same filter
    let countQuery = `
      SELECT COUNT(*) AS total 
      FROM reviews r
      JOIN parking_spots ps ON r.parking_id = ps.id
      WHERE ps.owner_id = ?
    `;
    const countParams = [req.user.id];
    if (spotId && spotId !== "all") {
      countQuery += ` AND ps.id = ?`;
      countParams.push(spotId);
    }

    db.query(countQuery, countParams, (err2, countResult) => {
      if (err2) {
        console.error("Error counting reviews:", err2);
        return res.status(500).json({ message: "Error counting reviews" });
      }

      res.json({
        reviews,
        pagination: {
          page,
          limit,
          total: countResult[0].total,
          totalPages: Math.ceil(countResult[0].total / limit),
        },
      });
    });
  });
});

// Owner responds to a review
router.post("/:reviewId/respond", auth(["owner"]), (req, res) => {
  const { reviewId } = req.params;
  const { response_text } = req.body;

  if (!reviewId || isNaN(reviewId)) {
    return res.status(400).json({ message: "Invalid review ID" });
  }

  if (!response_text || response_text.trim().length === 0) {
    return res.status(400).json({ message: "Response text is required" });
  }

  if (response_text.length > MAX_RESPONSE_LENGTH) {
    return res
      .status(400)
      .json({
        message: `Response must not exceed ${MAX_RESPONSE_LENGTH} characters`,
      });
  }

  const checkQuery = `
    SELECT ps.owner_id
    FROM reviews r
    JOIN parking_spots ps ON r.parking_id = ps.id
    WHERE r.id = ?;
  `;

  db.query(checkQuery, [reviewId], (err, result) => {
    if (err) {
      console.error("Error verifying ownership:", err);
      return res.status(500).json({ message: "Error verifying review" });
    }

    if (!result.length) {
      return res.status(404).json({ message: "Review not found" });
    }

    if (result[0].owner_id !== req.user.id) {
      return res.status(403).json({
        message: "You can only respond to reviews for your own parking spots",
      });
    }

    const insertQuery = `
      INSERT INTO review_responses (review_id, owner_id, response_text)
      VALUES (?, ?, ?)
      ON DUPLICATE KEY UPDATE 
        response_text = VALUES(response_text),
        updated_at = NOW();
    `;

    db.query(
      insertQuery,
      [reviewId, req.user.id, response_text.trim()],
      (err2) => {
        if (err2) {
          console.error("Error saving response:", err2);
          return res.status(500).json({ message: "Error saving response" });
        }

        res.json({ message: "Response added successfully" });
      },
    );
  });
});

// Owner ratings summary across all spots
router.get("/owner/summary", auth(["owner"]), (req, res) => {
  const query = `
    SELECT 
      COUNT(r.id) AS total_reviews,
      COALESCE(AVG(r.rating), 0) AS average_rating,
      COALESCE(AVG(r.cleanliness_rating), 0) AS avg_cleanliness,
      COALESCE(AVG(r.safety_rating), 0) AS avg_safety,
      COALESCE(AVG(r.accessibility_rating), 0) AS avg_accessibility,
      COUNT(DISTINCT r.parking_id) AS spots_with_reviews,
      SUM(CASE WHEN rr.response_text IS NOT NULL THEN 1 ELSE 0 END) AS responded_count,
      SUM(CASE WHEN rr.response_text IS NULL THEN 1 ELSE 0 END) AS pending_response_count
    FROM reviews r
    JOIN parking_spots ps ON r.parking_id = ps.id
    LEFT JOIN review_responses rr ON r.id = rr.review_id
    WHERE ps.owner_id = ?;
  `;

  db.query(query, [req.user.id], (err, result) => {
    if (err) {
      console.error("Error fetching owner summary:", err);
      return res.status(500).json({ message: "Error fetching summary" });
    }

    res.json(result[0] || {});
  });
});

// Get reviews for specific owner's parking spot
router.get("/owner/spot/:parkingId", auth(["owner"]), (req, res) => {
  const { parkingId } = req.params;

  if (!parkingId || isNaN(parkingId)) {
    return res.status(400).json({ message: "Invalid parking spot ID" });
  }

  db.query(
    "SELECT owner_id FROM parking_spots WHERE id = ?",
    [parkingId],
    (err, spotResult) => {
      if (err) {
        console.error("Error verifying spot:", err);
        return res.status(500).json({ message: "Error verifying spot" });
      }

      if (!spotResult.length) {
        return res.status(404).json({ message: "Parking spot not found" });
      }

      if (spotResult[0].owner_id !== req.user.id) {
        return res.status(403).json({
          message: "You can only view reviews for your own parking spots",
        });
      }

      const query = `
        SELECT 
          r.id,
          r.rating,
          r.comment,
          r.cleanliness_rating,
          r.safety_rating,
          r.accessibility_rating,
          r.created_at,
          r.updated_at,
          u.name AS user_name,
          rr.response_text AS my_response,
          rr.created_at AS response_date
        FROM reviews r
        JOIN users u ON r.user_id = u.id
        LEFT JOIN review_responses rr ON r.id = rr.review_id
        WHERE r.parking_id = ?
        ORDER BY r.created_at DESC;
      `;

      db.query(query, [parkingId], (err2, reviews) => {
        if (err2) {
          console.error("Error fetching reviews:", err2);
          return res.status(500).json({ message: "Error fetching reviews" });
        }

        res.json(reviews);
      });
    },
  );
});

module.exports = router;
