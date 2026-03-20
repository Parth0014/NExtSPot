import React, { useState, useEffect, useMemo } from "react";
import {
  Calendar,
  Clock,
  MapPin,
  X,
  CheckCircle,
  AlertCircle,
  Star,
  Bell,
  Plus,
} from "lucide-react";

import { useLocation } from "react-router-dom";

function UserReservations() {
  const [reservations, setReservations] = useState([]);
  const [message, setMessage] = useState("");
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [bookingToCancel, setBookingToCancel] = useState(null);
  const [reviewEligibility, setReviewEligibility] = useState({});
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [currentReview, setCurrentReview] = useState({
    parking_id: null,
    booking_id: null,
    spot_name: "",
    rating: 0,
    comment: "",
    cleanliness_rating: 0,
    safety_rating: 0,
    accessibility_rating: 0,
  });
  const [reviewSubmitting, setReviewSubmitting] = useState(false);
  const [highlightedBookingId, setHighlightedBookingId] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [activeTab, setActiveTab] = useState("upcoming");

  const token = localStorage.getItem("token");

  // Check URL parameters for highlighted booking
  // Highlight booking and optionally switch to history tab if needed
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const bookingId = urlParams.get("bookingId");
    const spotId = urlParams.get("spotId");

    if (bookingId) {
      setHighlightedBookingId(parseInt(bookingId));
      // Try to find the booking in reservations
      const found = reservations.find(
        (r) => String(r.booking_id) === String(bookingId),
      );
      if (found) {
        // If it's not in upcoming, switch to history
        if (found.status !== "active" && found.status !== "scheduled") {
          setActiveTab("history");
        }
      } else {
        // If not found yet, wait for reservations to load
        setTimeout(() => {
          const el = document.getElementById(`booking-${bookingId}`);
          if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "center" });
          }
        }, 800);
      }
      // Always scroll after a short delay
      setTimeout(() => {
        const el = document.getElementById(`booking-${bookingId}`);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 800);
    }
  }, [reservations]);

  // Update time every second for real-time countdowns
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Fetch reservations from API
  const fetchReservations = () => {
    fetch("http://localhost:5000/api/bookings/my-reservations", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(async (res) => {
        const txt = await res.text();
        try {
          const json = txt ? JSON.parse(txt) : null;
          if (!res.ok)
            throw new Error((json && json.message) || `Error ${res.status}`);
          return json;
        } catch (e) {
          if (!res.ok) throw new Error(`Error ${res.status}: ${txt}`);
          return null;
        }
      })
      .then((data) => {
        setReservations(Array.isArray(data) ? data : []);
      })
      .catch((err) => setMessage(err.message));
  };

  useEffect(() => {
    fetchReservations();
    const poll = setInterval(fetchReservations, 30000);
    return () => clearInterval(poll);
  }, []);

  // Check review eligibility for expired bookings
  const checkReviewEligibility = (parkingId) => {
    if (!parkingId) return;
    fetch(`http://localhost:5000/reviews/can-review/${parkingId}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        setReviewEligibility((prev) => ({
          ...prev,
          [parkingId]: data,
        }));
      })
      .catch((err) => console.error("Error checking review eligibility:", err));
  };

  useEffect(() => {
    reservations.forEach((reservation) => {
      const parkingId = reservation.parking_id || reservation.spot_id;
      const actualStatus = reservation.actual_status || reservation.status;
      if (actualStatus === "expired" && parkingId) {
        checkReviewEligibility(parkingId);
      }
    });
  }, [reservations]);

  // Calculate tab counts
  const counts = useMemo(() => {
    const activeOrScheduled = reservations.filter(
      (r) => r.status === "active" || r.status === "scheduled",
    );
    const history = reservations.filter(
      (r) => r.status !== "active" && r.status !== "scheduled",
    );
    return { upcoming: activeOrScheduled.length, history: history.length };
  }, [reservations]);

  // Helper functions
  const parseDate = (d) => (d ? new Date(d) : null);

  const formatFull = (d) => {
    const dt = parseDate(d);
    return dt ? dt.toLocaleString() : "-";
  };

  const formatDuration = (minutes) => {
    if (minutes == null) return "-";
    const m = parseInt(minutes, 10);
    if (isNaN(m)) return "-";
    const hrs = Math.floor(m / 60);
    const mins = m % 60;
    if (hrs > 0) return `${hrs}h ${mins}m`;
    return `${mins}m`;
  };

  // Cancellation logic
  const canCancelBooking = (booking) => {
    if (!booking) return false;
    const createdAt = booking.reserved_at;
    const startAtRaw = booking.start_at || booking.reserved_at;
    if (!createdAt) return false;

    const bookingTime = new Date(createdAt);
    if (isNaN(bookingTime.getTime())) return false;

    const startAt = new Date(startAtRaw);
    const nowDate = new Date(now);

    // For scheduled bookings: can cancel until 1 hour before start
    if (!isNaN(startAt.getTime()) && startAt > nowDate) {
      const minutesUntilStart = (startAt - nowDate) / (1000 * 60);
      return minutesUntilStart >= 60;
    }

    // For active bookings: can cancel within 5 minutes of booking creation
    const timeDifferenceMinutes = (nowDate - bookingTime) / (1000 * 60);
    return timeDifferenceMinutes <= 5;
  };

  const getRemainingCancelTime = (booking) => {
    if (!booking) return null;
    const createdAt = booking.reserved_at;
    const startAtRaw = booking.start_at || booking.reserved_at;
    if (!createdAt) return null;

    const bookingTime = new Date(createdAt);
    if (isNaN(bookingTime.getTime())) return null;

    const startAt = new Date(startAtRaw);
    const nowDate = new Date(now);

    // For scheduled bookings
    if (!isNaN(startAt.getTime()) && startAt > nowDate) {
      const minutesUntilStart = (startAt - nowDate) / (1000 * 60);
      const remainingMinutes = Math.max(0, minutesUntilStart - 60);
      if (remainingMinutes <= 0) return null;
      const hrs = Math.floor(remainingMinutes / 60);
      const mins = Math.floor(remainingMinutes % 60);
      return hrs > 0 ? `${hrs}h ${mins}m` : `${mins}m`;
    }

    // For active bookings
    const timeDifferenceMinutes = (nowDate - bookingTime) / (1000 * 60);
    const remainingMinutes = Math.max(0, 5 - timeDifferenceMinutes);
    if (remainingMinutes <= 0) return null;
    const minutes = Math.floor(remainingMinutes);
    const seconds = Math.floor((remainingMinutes - minutes) * 60);
    return `${minutes}m ${seconds}s`;
  };

  const cancelBooking = (bookingId) => {
    const booking = reservations.find((r) => r.booking_id === bookingId);
    if (!booking) {
      setMessage("Booking not found");
      return;
    }
    if (!canCancelBooking(booking)) {
      const startAtRaw = booking.start_at || booking.reserved_at;
      const startAt = new Date(startAtRaw);
      const nowDate = new Date(now);
      if (!isNaN(startAt.getTime()) && startAt > nowDate) {
        setMessage(
          "Cancellation not allowed: you can only cancel until 1 hour before the scheduled start.",
        );
      } else {
        setMessage(
          "Cancellation not allowed. You can only cancel within 5 minutes of booking.",
        );
      }
      return;
    }
    setBookingToCancel(bookingId);
    setShowConfirmModal(true);
  };

  const confirmCancellation = () => {
    fetch(`http://localhost:5000/api/bookings/cancel/${bookingToCancel}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        setMessage(data.message || "Reservation cancelled");
        fetchReservations();
        setShowConfirmModal(false);
        setBookingToCancel(null);
      })
      .catch(() => {
        setMessage("Cancellation failed");
        setShowConfirmModal(false);
        setBookingToCancel(null);
      });
  };

  const getStatusStyle = (status) => {
    const baseStyle = {
      padding: "6px 10px",
      borderRadius: "999px",
      fontSize: "12px",
      fontWeight: "700",
      display: "inline-flex",
      alignItems: "center",
      gap: "8px",
      border: "1px solid",
      textTransform: "capitalize",
    };

    switch (status) {
      case "active":
        return {
          ...baseStyle,
          backgroundColor: "#dcfce7",
          color: "#166534",
          borderColor: "#86efac",
        };
      case "cancelled":
        return {
          ...baseStyle,
          backgroundColor: "#fee2e2",
          color: "#dc2626",
          borderColor: "#fca5a5",
        };
      case "scheduled":
        return {
          ...baseStyle,
          backgroundColor: "#fef3c7",
          color: "#92400e",
          borderColor: "#fde68a",
        };
      case "expired":
        return {
          ...baseStyle,
          backgroundColor: "#f3f4f6",
          color: "#374151",
          borderColor: "#d1d5db",
        };
      case "completed":
        return {
          ...baseStyle,
          backgroundColor: "#dbeafe",
          color: "#1e40af",
          borderColor: "#93c5fd",
        };
      default:
        return {
          ...baseStyle,
          backgroundColor: "#f3f4f6",
          color: "#374151",
          borderColor: "#d1d5db",
        };
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "active":
        return <CheckCircle style={{ width: "14px", height: "14px" }} />;
      case "scheduled":
        return <Clock style={{ width: "14px", height: "14px" }} />;
      case "cancelled":
        return <X style={{ width: "14px", height: "14px" }} />;
      default:
        return <AlertCircle style={{ width: "14px", height: "14px" }} />;
    }
  };

  const openReviewModal = (reservation) => {
    const parkingId = reservation.parking_id || reservation.spot_id;
    setCurrentReview({
      parking_id: parkingId,
      booking_id: reservation.booking_id,
      spot_name: reservation.spot_name,
      rating: 0,
      comment: "",
      cleanliness_rating: 0,
      safety_rating: 0,
      accessibility_rating: 0,
    });
    setShowReviewModal(true);
  };

  const closeReviewModal = () => {
    setShowReviewModal(false);
    setCurrentReview({
      parking_id: null,
      booking_id: null,
      spot_name: "",
      rating: 0,
      comment: "",
      cleanliness_rating: 0,
      safety_rating: 0,
      accessibility_rating: 0,
    });
  };

  const submitReview = () => {
    if (currentReview.rating === 0) {
      setMessage("Please provide an overall rating");
      return;
    }

    setReviewSubmitting(true);

    fetch("http://localhost:5000/reviews/submit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        parking_id: currentReview.parking_id,
        booking_id: currentReview.booking_id,
        rating: currentReview.rating,
        comment: currentReview.comment.trim() || null,
        cleanliness_rating: currentReview.cleanliness_rating || null,
        safety_rating: currentReview.safety_rating || null,
        accessibility_rating: currentReview.accessibility_rating || null,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setMessage(data.message || "Review submitted successfully!");
        closeReviewModal();
        checkReviewEligibility(currentReview.parking_id);
      })
      .catch((err) => {
        setMessage("Failed to submit review: " + err.message);
      })
      .finally(() => {
        setReviewSubmitting(false);
      });
  };

  const StarRating = ({ value, onChange, label }) => {
    return (
      <div style={{ marginBottom: "16px" }}>
        <label
          style={{
            display: "block",
            fontSize: "14px",
            fontWeight: "500",
            color: "#374151",
            marginBottom: "8px",
          }}
        >
          {label}{" "}
          {label === "Overall Rating" && (
            <span style={{ color: "#dc2626" }}>*</span>
          )}
        </label>
        <div style={{ display: "flex", gap: "8px" }}>
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              type="button"
              onClick={() => onChange(star)}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: "4px",
                transition: "transform 0.2s ease",
              }}
              onMouseOver={(e) =>
                (e.currentTarget.style.transform = "scale(1.1)")
              }
              onMouseOut={(e) => (e.currentTarget.style.transform = "scale(1)")}
            >
              <Star
                style={{
                  width: "32px",
                  height: "32px",
                  fill: star <= value ? "#fbbf24" : "none",
                  stroke: star <= value ? "#fbbf24" : "#d1d5db",
                  strokeWidth: 2,
                }}
              />
            </button>
          ))}
        </div>
      </div>
    );
  };

  // Partition reservations by tab
  const upcomingReservations = reservations.filter(
    (r) => r.status === "active" || r.status === "scheduled",
  );
  const historyReservations = reservations.filter(
    (r) => r.status !== "active" && r.status !== "scheduled",
  );

  const getProgressAndCountdown = (r) => {
    const actualStatus = r.actual_status || r.status;

    const reservedAt = parseDate(r.reserved_at);
    const startAt = parseDate(r.start_at || r.reserved_at);
    const durationMs = (parseInt(r.duration_minutes || 0, 10) || 0) * 60 * 1000;
    const nowDate = new Date(now);

    // DEBUG - Now actualStatus is defined
    console.log("Booking #" + r.booking_id + " Debug:", {
      actualStatus,
      reserved_at_raw: r.reserved_at,
      start_at_raw: r.start_at,
      reservedAt: reservedAt,
      startAt: startAt,
      nowDate: nowDate,
      duration_minutes: r.duration_minutes,
      time_until_start_hours: (startAt - nowDate) / (1000 * 60 * 60),
    });

    let remainingMs = null;
    let percent = 0;
    let isWaitingToStart = false;

    const formatHHMMSS = (ms) => {
      if (ms == null || ms <= 0) return "00:00:00";
      const totalSec = Math.floor(ms / 1000);
      const h = String(Math.floor(totalSec / 3600)).padStart(2, "0");
      const m = String(Math.floor((totalSec % 3600) / 60)).padStart(2, "0");
      const s = String(totalSec % 60).padStart(2, "0");
      return `${h}:${m}:${s}`;
    };

    // For scheduled bookings - show countdown until start
    if (actualStatus === "scheduled" && startAt && startAt > nowDate) {
      isWaitingToStart = true;
      remainingMs = startAt - nowDate;
      // Progress bar shows how much time has passed until start
      // If booking was made 2 hours ago and starts in 1 hour, we're 66% there
      if (reservedAt) {
        const totalWaitTime = startAt - reservedAt;
        const elapsedWaitTime = nowDate - reservedAt;
        percent = Math.max(
          0,
          Math.min(100, (elapsedWaitTime / totalWaitTime) * 100),
        );
      }
    }
    // For active bookings - show countdown until end
    else if (actualStatus === "active" && startAt && durationMs > 0) {
      const endAt = new Date(startAt.getTime() + durationMs);
      remainingMs = endAt - nowDate;
      const elapsed = nowDate - startAt;
      percent = Math.max(0, Math.min(100, (elapsed / durationMs) * 100));
    }

    return {
      remainingMs,
      percent: Math.round(percent),
      countdown: formatHHMMSS(remainingMs),
      isWaitingToStart,
    };
  };

  // Calculate time until scheduled booking starts
  const getStartsInText = (startDate) => {
    const startAt = parseDate(startDate);
    if (!startAt) return "-";

    const nowDate = new Date(now);
    const startsInMs = startAt - nowDate;

    if (startsInMs <= 0) return "Starting now";

    const totalSec = Math.floor(startsInMs / 1000);
    const days = Math.floor(totalSec / 86400);
    const hours = Math.floor((totalSec % 86400) / 3600);
    const mins = Math.floor((totalSec % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#ffffff",
        fontFamily: "Inter, system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Header */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          backgroundColor: "#ffffff",
          padding: "16px 40px",
          borderBottom: "1px solid #f1f5f9",
          position: "sticky",
          top: 0,
          zIndex: 50,
          boxShadow: "0 1px 3px rgba(0, 0, 0, 0.05)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <h2
            style={{
              fontSize: "25px",
              fontWeight: "1000",
              color: "#1e293b",
              margin: 0,
            }}
          >
            NExtSPot
          </h2>
        </div>
        <div
          style={{
            display: "flex",
            flex: 1,
            justifyContent: "flex-end",
            gap: "32px",
          }}
        >
          <nav style={{ display: "flex", alignItems: "center", gap: "36px" }}>
            <a
              href="/user/dashboard"
              style={{
                color: "#64748b",
                fontSize: "14px",
                fontWeight: "500",
                textDecoration: "none",
                transition: "color 0.2s ease",
              }}
              onMouseOver={(e) => (e.currentTarget.style.color = "#3b82f6")}
              onMouseOut={(e) => (e.currentTarget.style.color = "#64748b")}
            >
              Find Parking
            </a>
            <a
              style={{
                color: "#3b82f6",
                fontSize: "14px",
                fontWeight: "500",
                textDecoration: "none",
                borderBottom: "2px solid #3b82f6",
                paddingBottom: "2px",
              }}
            >
              My Reservations
            </a>
          </nav>
        </div>
      </header>

      <main
        style={{
          flex: 1,
          padding: "40px 160px",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            maxWidth: "1000px",
            flex: 1,
            width: "100%",
          }}
        >
          {/* Page Header */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: "16px",
              paddingBottom: "32px",
              flexWrap: "wrap",
            }}
          >
            <h1
              style={{
                fontSize: "36px",
                fontWeight: "900",
                color: "#1e293b",
                margin: 0,
                letterSpacing: "-0.02em",
              }}
            >
              My Reservations
            </h1>
            <a
              href="/user/dashboard"
              style={{
                backgroundColor: "#3b82f6",
                color: "white",
                fontSize: "14px",
                fontWeight: "700",
                padding: "12px 24px",
                borderRadius: "12px",
                border: "none",
                cursor: "pointer",
                boxShadow: "0 4px 14px rgba(59, 130, 246, 0.3)",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                transition: "all 0.2s ease",
                textDecoration: "none", // prevents underline
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.backgroundColor = "#2563eb";
                e.currentTarget.style.transform = "translateY(-1px)";
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.backgroundColor = "#3b82f6";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              <Plus style={{ width: "18px", height: "18px" }} />
              Book New Spot
            </a>
          </div>

          {/* Tabs */}
          <div style={{ marginBottom: "40px" }}>
            <div
              style={{
                display: "flex",
                borderBottom: "1px solid #e2e8f0",
                gap: "32px",
              }}
            >
              <button
                onClick={() => setActiveTab("upcoming")}
                style={{
                  all: "unset",
                  cursor: "pointer",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  borderBottom:
                    activeTab === "upcoming"
                      ? "3px solid #3b82f6"
                      : "3px solid transparent",
                  color: activeTab === "upcoming" ? "#3b82f6" : "#64748b",
                  paddingBottom: "13px",
                  paddingTop: "16px",
                  paddingLeft: "8px",
                  paddingRight: "8px",
                }}
              >
                <p style={{ fontSize: "14px", fontWeight: "700", margin: 0 }}>
                  Active / Upcoming ({counts.upcoming})
                </p>
              </button>
              <button
                onClick={() => setActiveTab("history")}
                style={{
                  all: "unset",
                  cursor: "pointer",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  borderBottom:
                    activeTab === "history"
                      ? "3px solid #3b82f6"
                      : "3px solid transparent",
                  color: activeTab === "history" ? "#3b82f6" : "#64748b",
                  paddingBottom: "13px",
                  paddingTop: "16px",
                  paddingLeft: "8px",
                  paddingRight: "8px",
                }}
              >
                <p style={{ fontSize: "14px", fontWeight: "700", margin: 0 }}>
                  History ({counts.history})
                </p>
              </button>
            </div>
          </div>

          {/* Message Display */}
          {message && (
            <div
              style={{
                marginBottom: "24px",
                padding: "16px",
                backgroundColor:
                  message.includes("failed") || message.includes("error")
                    ? "#fef2f2"
                    : "#f0fdf4",
                border: `1px solid ${message.includes("failed") || message.includes("error") ? "#fecaca" : "#bbf7d0"}`,
                borderRadius: "12px",
                display: "flex",
                alignItems: "center",
                gap: "12px",
              }}
            >
              {message.includes("failed") || message.includes("error") ? (
                <AlertCircle
                  style={{
                    width: "20px",
                    height: "20px",
                    color: "#dc2626",
                    flexShrink: 0,
                  }}
                />
              ) : (
                <CheckCircle
                  style={{
                    width: "20px",
                    height: "20px",
                    color: "#16a34a",
                    flexShrink: 0,
                  }}
                />
              )}
              <p
                style={{
                  color:
                    message.includes("failed") || message.includes("error")
                      ? "#dc2626"
                      : "#166534",
                  fontWeight: "500",
                  margin: 0,
                  flex: 1,
                }}
              >
                {message}
              </p>
              <button
                onClick={() => setMessage("")}
                style={{
                  background: "none",
                  border: "none",
                  color:
                    message.includes("failed") || message.includes("error")
                      ? "#dc2626"
                      : "#16a34a",
                  cursor: "pointer",
                  padding: "4px",
                }}
              >
                <X style={{ width: "20px", height: "20px" }} />
              </button>
            </div>
          )}

          {/* Active/Upcoming Tab Content */}
          {activeTab === "upcoming" && (
            <>
              {upcomingReservations.length === 0 && (
                <div
                  style={{
                    textAlign: "center",
                    paddingTop: "64px",
                    paddingBottom: "64px",
                  }}
                >
                  <div
                    style={{
                      backgroundColor: "#ffffff",
                      borderRadius: "16px",
                      boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
                      border: "1px solid #e5e7eb",
                      padding: "48px",
                      maxWidth: "600px",
                      margin: "0 auto",
                    }}
                  >
                    <div
                      style={{
                        width: "64px",
                        height: "64px",
                        backgroundColor: "#f3f4f6",
                        borderRadius: "50%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        margin: "0 auto 16px auto",
                      }}
                    >
                      <Calendar
                        style={{
                          width: "32px",
                          height: "32px",
                          color: "#9ca3af",
                        }}
                      />
                    </div>
                    <h3
                      style={{
                        fontSize: "20px",
                        fontWeight: "600",
                        color: "#111827",
                        margin: "0 0 8px 0",
                      }}
                    >
                      No active or upcoming reservations
                    </h3>
                    <p style={{ color: "#6b7280", margin: "0 0 24px 0" }}>
                      You have no active or scheduled bookings. Book a spot to
                      get started.
                    </p>
                    <a
                      href="#"
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "8px",
                        backgroundColor: "#2563eb",
                        color: "white",
                        padding: "12px 24px",
                        borderRadius: "8px",
                        fontWeight: "500",
                        textDecoration: "none",
                        transition: "background-color 0.2s ease",
                      }}
                      onMouseOver={(e) =>
                        (e.currentTarget.style.backgroundColor = "#1d4ed8")
                      }
                      onMouseOut={(e) =>
                        (e.currentTarget.style.backgroundColor = "#2563eb")
                      }
                    >
                      <MapPin style={{ width: "16px", height: "16px" }} />
                      <span>Find Parking</span>
                    </a>
                  </div>
                </div>
              )}

              {upcomingReservations.length > 0 && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(auto-fill, minmax(400px, 1fr))",
                    gap: "24px",
                  }}
                >
                  {upcomingReservations.map((r) => {
                    const parkingId = r.parking_id || r.spot_id;
                    const isHighlighted = highlightedBookingId === r.booking_id;
                    const actualStatus = r.actual_status || r.status;
                    const {
                      remainingMs,
                      percent,
                      countdown,
                      isWaitingToStart,
                    } = getProgressAndCountdown(r);
                    const startsInText = getStartsInText(
                      r.start_at || r.reserved_at,
                    );
                    const remainingCancelTime = getRemainingCancelTime(r);

                    return (
                      <div
                        key={r.booking_id}
                        id={`booking-${r.booking_id}`}
                        style={{
                          backgroundColor: "#ffffff",
                          borderRadius: "12px",
                          padding: "20px",
                          border: isHighlighted
                            ? "2px solid #fbbf24"
                            : "1px solid #f1f5f9",
                          boxShadow: isHighlighted
                            ? "0 0 0 3px #fbbf24"
                            : "0 1px 2px rgba(0, 0, 0, 0.03)",
                          transition: "all 0.2s ease",
                        }}
                      >
                        {/* Card Header */}
                        <div
                          style={{
                            display: "flex",
                            alignItems: "flex-start",
                            justifyContent: "space-between",
                            gap: "12px",
                            marginBottom: "12px",
                            flexWrap: "wrap",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              gap: "16px",
                              alignItems: "center",
                            }}
                          >
                            <div
                              style={{
                                width: "56px",
                                height: "56px",
                                borderRadius: "12px",
                                backgroundColor: "#f8fafc",
                                flexShrink: 0,
                                border: "1px solid #e5e7eb",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontWeight: "700",
                                color: "#111827",
                              }}
                            >
                              {r.spot_name
                                ? r.spot_name.charAt(0).toUpperCase()
                                : "#"}
                            </div>
                            <div>
                              <h3
                                style={{
                                  fontSize: "18px",
                                  fontWeight: "700",
                                  color: "#1e293b",
                                  margin: "0 0 6px 0",
                                }}
                              >
                                {r.spot_name}
                              </h3>
                              <div
                                style={{
                                  display: "flex",
                                  gap: "8px",
                                  alignItems: "center",
                                  color: "#64748b",
                                  fontSize: "13px",
                                }}
                              >
                                <MapPin
                                  style={{
                                    width: "14px",
                                    height: "14px",
                                    color: "#3b82f6",
                                  }}
                                />
                                <span>{r.address || "-"}</span>
                              </div>

                              {/* ID Badges */}
                              <div
                                style={{
                                  marginTop: "8px",
                                  display: "flex",
                                  gap: "8px",
                                  alignItems: "center",
                                  flexWrap: "wrap",
                                }}
                              >
                                <span
                                  style={{
                                    backgroundColor: "#f3f4f6",
                                    padding: "6px 10px",
                                    borderRadius: "999px",
                                    fontSize: "12px",
                                    fontWeight: "600",
                                    color: "#374151",
                                    border: "1px solid #e5e7eb",
                                  }}
                                >
                                  Booking ID: #{r.booking_id}
                                </span>
                                {r.spot_id && (
                                  <span
                                    style={{
                                      backgroundColor: "#eef2ff",
                                      padding: "6px 10px",
                                      borderRadius: "999px",
                                      fontSize: "12px",
                                      fontWeight: "600",
                                      color: "#3730a3",
                                      border: "1px solid #e0e7ff",
                                    }}
                                  >
                                    Spot ID: {r.spot_id}
                                  </span>
                                )}
                                {r.parking_id && (
                                  <span
                                    style={{
                                      backgroundColor: "#f0fdf4",
                                      padding: "6px 10px",
                                      borderRadius: "999px",
                                      fontSize: "12px",
                                      fontWeight: "600",
                                      color: "#166534",
                                      border: "1px solid #bbf7d0",
                                    }}
                                  >
                                    Parking ID: {r.parking_id}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>

                          <div
                            style={{
                              display: "flex",
                              flexDirection: "column",
                              alignItems: "flex-end",
                              gap: "8px",
                            }}
                          >
                            <div style={getStatusStyle(actualStatus)}>
                              {getStatusIcon(actualStatus)}
                              <span>{actualStatus}</span>
                            </div>
                            <div
                              style={{
                                color: "#6b7280",
                                fontSize: "13px",
                                textAlign: "right",
                              }}
                            >
                              <div>Reserved: {formatFull(r.reserved_at)}</div>
                              {r.start_at && (
                                <div>Start: {formatFull(r.start_at)}</div>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Card Body */}
                        <div
                          style={{
                            borderTop: "1px solid #f1f5f9",
                            paddingTop: "12px",
                            display: "grid",
                            gridTemplateColumns: "1fr auto",
                            gap: "12px",
                            alignItems: "center",
                          }}
                        >
                          <div>
                            <div
                              style={{
                                display: "flex",
                                gap: "12px",
                                alignItems: "center",
                                marginBottom: "8px",
                                flexWrap: "wrap",
                              }}
                            >
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "6px",
                                  color: "#64748b",
                                }}
                              >
                                <Clock
                                  style={{ width: "14px", height: "14px" }}
                                />
                                <span
                                  style={{
                                    fontSize: "13px",
                                    fontWeight: "600",
                                  }}
                                >
                                  {formatDuration(r.duration_minutes)}
                                </span>
                              </div>
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "6px",
                                }}
                              >
                                <span
                                  style={{
                                    fontSize: "13px",
                                    fontWeight: "600",
                                  }}
                                >
                                  Total:{" "}
                                  {new Intl.NumberFormat("en-IN", {
                                    style: "currency",
                                    currency: "INR",
                                  }).format(r.total_price)}
                                </span>
                              </div>
                              {/* Show countdown based on status */}
                              {(actualStatus === "scheduled" ||
                                actualStatus === "active") && (
                                <div
                                  style={{ color: "#64748b", fontSize: "13px" }}
                                >
                                  <span>
                                    {actualStatus === "scheduled"
                                      ? "Starts in: "
                                      : "Ends in: "}
                                    {countdown}
                                  </span>
                                </div>
                              )}
                            </div>

                            {/* Progress Bar for Active and Scheduled Bookings */}
                            {(actualStatus === "active" ||
                              actualStatus === "scheduled") && (
                              <div style={{ marginTop: "8px" }}>
                                <div
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "12px",
                                  }}
                                >
                                  <div style={{ flex: 1 }}>
                                    <div
                                      style={{
                                        height: "10px",
                                        backgroundColor: "#f1f5f9",
                                        borderRadius: "999px",
                                        overflow: "hidden",
                                        border: "1px solid #e2e8f0",
                                      }}
                                    >
                                      <div
                                        style={{
                                          width:
                                            percent > 0 && percent < 5
                                              ? "5%"
                                              : `${percent}%`,
                                          minWidth:
                                            percent > 0 && percent < 5
                                              ? "5%"
                                              : undefined,
                                          height: "100%",
                                          background: isWaitingToStart
                                            ? "linear-gradient(90deg,#fbbf24,#f59e0b)"
                                            : "linear-gradient(90deg,#60a5fa,#3b82f6)",
                                          boxShadow: isWaitingToStart
                                            ? "0 0 10px rgba(251,191,36,0.35)"
                                            : "0 0 10px rgba(59,130,246,0.35)",
                                          transition:
                                            "width 0.5s cubic-bezier(0.4,0,0.2,1)",
                                        }}
                                      />
                                    </div>
                                  </div>
                                  <div
                                    style={{
                                      fontVariantNumeric: "tabular-nums",
                                      fontWeight: "700",
                                      color: "#1e293b",
                                    }}
                                  >
                                    {countdown}
                                  </div>
                                </div>
                                {/* Show percentage based on status */}
                                <div
                                  style={{
                                    marginTop: "8px",
                                    color: "#6b7280",
                                    fontSize: "12px",
                                    textAlign: "right",
                                  }}
                                >
                                  {actualStatus === "scheduled"
                                    ? `${percent}% until start`
                                    : `${percent}% elapsed`}
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Action Buttons */}
                          <div
                            style={{
                              display: "flex",
                              flexDirection: "column",
                              gap: "8px",
                              alignItems: "flex-end",
                            }}
                          >
                            {(r.status === "active" ||
                              r.status === "scheduled") &&
                              (canCancelBooking(r) ? (
                                <div>
                                  <button
                                    onClick={() => cancelBooking(r.booking_id)}
                                    style={{
                                      display: "inline-flex",
                                      alignItems: "center",
                                      gap: "8px",
                                      backgroundColor:
                                        r.status === "active"
                                          ? "#fef2f2"
                                          : "#fff",
                                      color:
                                        r.status === "active"
                                          ? "#dc2626"
                                          : "#111827",
                                      padding: "8px 16px",
                                      borderRadius: "8px",
                                      fontWeight: "600",
                                      fontSize: "13px",
                                      border:
                                        r.status === "active"
                                          ? "1px solid #fecaca"
                                          : "1px solid #e5e7eb",
                                      cursor: "pointer",
                                    }}
                                  >
                                    <X
                                      style={{ width: "14px", height: "14px" }}
                                    />
                                    Cancel
                                  </button>
                                  {remainingCancelTime && (
                                    <div
                                      style={{
                                        fontSize: "11px",
                                        color: "#64748b",
                                        marginTop: "4px",
                                        textAlign: "right",
                                      }}
                                    >
                                      {r.status === "active"
                                        ? `Cancel within: ${remainingCancelTime}`
                                        : `Can cancel for: ${remainingCancelTime}`}
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <div
                                  style={{
                                    display: "inline-flex",
                                    alignItems: "center",
                                    gap: "8px",
                                    backgroundColor: "#f9fafb",
                                    color: "#6b7280",
                                    padding: "8px 16px",
                                    borderRadius: "8px",
                                    fontWeight: "500",
                                    fontSize: "13px",
                                    border: "1px solid #e5e7eb",
                                  }}
                                >
                                  <X
                                    style={{ width: "14px", height: "14px" }}
                                  />
                                  <span>
                                    {r.status === "scheduled"
                                      ? "Cancel window passed"
                                      : "Cannot cancel"}
                                  </span>
                                </div>
                              ))}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}

          {/* History Tab Content */}
          {activeTab === "history" && (
            <>
              {historyReservations.length === 0 && (
                <div
                  style={{
                    textAlign: "center",
                    paddingTop: "64px",
                    paddingBottom: "64px",
                  }}
                >
                  <div
                    style={{
                      backgroundColor: "#ffffff",
                      borderRadius: "16px",
                      boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
                      border: "1px solid #e5e7eb",
                      padding: "48px",
                      maxWidth: "600px",
                      margin: "0 auto",
                    }}
                  >
                    <div
                      style={{
                        width: "64px",
                        height: "64px",
                        backgroundColor: "#f3f4f6",
                        borderRadius: "50%",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        margin: "0 auto 16px auto",
                      }}
                    >
                      <Calendar
                        style={{
                          width: "32px",
                          height: "32px",
                          color: "#9ca3af",
                        }}
                      />
                    </div>
                    <h3
                      style={{
                        fontSize: "20px",
                        fontWeight: "600",
                        color: "#111827",
                        margin: "0 0 8px 0",
                      }}
                    >
                      No history yet
                    </h3>
                    <p style={{ color: "#6b7280", margin: "0 0 24px 0" }}>
                      Past reservations will appear here.
                    </p>
                  </div>
                </div>
              )}

              {historyReservations.length > 0 && (
                <div style={{ display: "grid", gap: "16px" }}>
                  {historyReservations.map((r) => {
                    const parkingId = r.parking_id || r.spot_id;
                    const eligibility = reviewEligibility[parkingId];
                    const isHighlighted = highlightedBookingId === r.booking_id;
                    const actualStatus = r.actual_status || r.status;

                    return (
                      <div
                        key={r.booking_id}
                        id={`booking-${r.booking_id}`}
                        style={{
                          backgroundColor: "#ffffff",
                          borderRadius: "12px",
                          padding: "20px",
                          border: isHighlighted
                            ? "2px solid #fbbf24"
                            : "1px solid #f1f5f9",
                          boxShadow: isHighlighted
                            ? "0 0 0 3px #fbbf24"
                            : "0 1px 2px rgba(0, 0, 0, 0.03)",
                          transition: "all 0.2s ease",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "flex-start",
                            justifyContent: "space-between",
                            marginBottom: "12px",
                            flexWrap: "wrap",
                            gap: "12px",
                          }}
                        >
                          <div style={{ flex: 1 }}>
                            <h3
                              style={{
                                fontSize: "18px",
                                fontWeight: "600",
                                color: "#1e293b",
                                margin: "0 0 12px 0",
                                display: "flex",
                                alignItems: "center",
                                gap: "8px",
                              }}
                            >
                              <MapPin
                                style={{
                                  width: "18px",
                                  height: "18px",
                                  color: "#3b82f6",
                                }}
                              />
                              <span>{r.spot_name}</span>
                            </h3>

                            <div
                              style={{
                                display: "grid",
                                gridTemplateColumns:
                                  "repeat(auto-fit, minmax(180px, 1fr))",
                                gap: "12px",
                                marginBottom: "12px",
                              }}
                            >
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "6px",
                                  color: "#64748b",
                                }}
                              >
                                <Clock
                                  style={{ width: "14px", height: "14px" }}
                                />
                                <span
                                  style={{
                                    fontSize: "13px",
                                    fontWeight: "500",
                                  }}
                                >
                                  Duration: {formatDuration(r.duration_minutes)}
                                </span>
                              </div>

                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "6px",
                                  color: "#64748b",
                                }}
                              >
                                <span
                                  style={{
                                    fontSize: "13px",
                                    fontWeight: "600",
                                  }}
                                >
                                  Total:{" "}
                                  {new Intl.NumberFormat("en-IN", {
                                    style: "currency",
                                    currency: "INR",
                                  }).format(r.total_price)}
                                </span>
                              </div>

                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "8px",
                                }}
                              >
                                <span style={getStatusStyle(actualStatus)}>
                                  {getStatusIcon(actualStatus)}
                                  <span>{actualStatus}</span>
                                </span>
                              </div>
                            </div>

                            <div style={{ color: "#6b7280", fontSize: "13px" }}>
                              <div>Reserved: {formatFull(r.reserved_at)}</div>
                              {r.start_at && (
                                <div>Start: {formatFull(r.start_at)}</div>
                              )}
                              {r.spot_id && <div>Spot ID: {r.spot_id}</div>}
                              {r.parking_id && (
                                <div>Parking ID: {r.parking_id}</div>
                              )}
                            </div>
                          </div>
                        </div>

                        <div
                          style={{
                            paddingTop: "12px",
                            borderTop: "1px solid #f1f5f9",
                            display: "flex",
                            gap: "12px",
                            flexWrap: "wrap",
                          }}
                        >
                          {r.status === "expired" &&
                            parkingId &&
                            (!eligibility ? (
                              <div
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  gap: "8px",
                                  backgroundColor: "#f3f4f6",
                                  color: "#6b7280",
                                  padding: "8px 16px",
                                  borderRadius: "8px",
                                  fontWeight: "500",
                                  fontSize: "13px",
                                  border: "1px solid #e5e7eb",
                                }}
                              >
                                <Star
                                  style={{ width: "14px", height: "14px" }}
                                />
                                <span>Checking...</span>
                              </div>
                            ) : eligibility && eligibility.canReview ? (
                              <button
                                onClick={() => openReviewModal(r)}
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  gap: "8px",
                                  backgroundColor: "#fef3c7",
                                  color: "#92400e",
                                  padding: "8px 16px",
                                  borderRadius: "8px",
                                  fontWeight: "600",
                                  fontSize: "13px",
                                  border: "1px solid #fde68a",
                                  cursor: "pointer",
                                }}
                              >
                                <Star
                                  style={{ width: "14px", height: "14px" }}
                                />
                                <span>Write Review</span>
                              </button>
                            ) : (
                              eligibility && (
                                <div
                                  style={{
                                    display: "inline-flex",
                                    alignItems: "center",
                                    gap: "8px",
                                    backgroundColor: "#f0fdf4",
                                    color: "#166534",
                                    padding: "8px 16px",
                                    borderRadius: "8px",
                                    fontWeight: "500",
                                    fontSize: "13px",
                                    border: "1px solid #bbf7d0",
                                  }}
                                >
                                  <CheckCircle
                                    style={{ width: "14px", height: "14px" }}
                                  />
                                  <span>
                                    {eligibility.reason === "already_reviewed"
                                      ? "Reviewed"
                                      : "Not Available"}
                                  </span>
                                </div>
                              )
                            ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}

          {/* Footer */}
          <div
            style={{
              marginTop: "64px",
              textAlign: "center",
              color: "#94a3b8",
              fontSize: "14px",
              paddingBottom: "40px",
              borderTop: "1px solid #f1f5f9",
              paddingTop: "32px",
            }}
          >
            <p style={{ margin: 0 }}>
              © 2023 NExtSPot Inc. Need help?{" "}
              <a
                href="#"
                style={{
                  color: "#3b82f6",
                  fontWeight: "500",
                  textDecoration: "none",
                }}
                onMouseOver={(e) =>
                  (e.currentTarget.style.textDecoration = "underline")
                }
                onMouseOut={(e) =>
                  (e.currentTarget.style.textDecoration = "none")
                }
              >
                Contact Support
              </a>
            </p>
          </div>
        </div>
      </main>

      {/* Confirmation Modal */}
      {showConfirmModal && (
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
        >
          <div
            style={{
              backgroundColor: "#ffffff",
              borderRadius: "16px",
              padding: "32px",
              maxWidth: "400px",
              width: "90%",
              boxShadow:
                "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
            }}
          >
            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  width: "64px",
                  height: "64px",
                  backgroundColor: "#fef2f2",
                  borderRadius: "50%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  margin: "0 auto 16px auto",
                }}
              >
                <AlertCircle
                  style={{ width: "32px", height: "32px", color: "#dc2626" }}
                />
              </div>

              <h3
                style={{
                  fontSize: "18px",
                  fontWeight: "600",
                  color: "#111827",
                  margin: "0 0 8px 0",
                }}
              >
                Cancel Reservation
              </h3>

              <p
                style={{
                  color: "#6b7280",
                  margin: "0 0 24px 0",
                  lineHeight: "1.5",
                }}
              >
                Are you sure you want to cancel this reservation? This action
                cannot be undone.
              </p>

              <div
                style={{
                  display: "flex",
                  gap: "12px",
                  justifyContent: "center",
                }}
              >
                <button
                  onClick={() => {
                    setShowConfirmModal(false);
                    setBookingToCancel(null);
                  }}
                  style={{
                    padding: "10px 20px",
                    backgroundColor: "#f9fafb",
                    border: "1px solid #d1d5db",
                    borderRadius: "8px",
                    fontWeight: "500",
                    color: "#374151",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.backgroundColor = "#f3f4f6";
                    e.currentTarget.style.borderColor = "#9ca3af";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.backgroundColor = "#f9fafb";
                    e.currentTarget.style.borderColor = "#d1d5db";
                  }}
                >
                  Keep Reservation
                </button>

                <button
                  onClick={confirmCancellation}
                  style={{
                    padding: "10px 20px",
                    backgroundColor: "#dc2626",
                    border: "1px solid #dc2626",
                    borderRadius: "8px",
                    fontWeight: "500",
                    color: "white",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.backgroundColor = "#b91c1c";
                    e.currentTarget.style.borderColor = "#b91c1c";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.backgroundColor = "#dc2626";
                    e.currentTarget.style.borderColor = "#dc2626";
                  }}
                >
                  Yes, Cancel It
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Review Modal */}
      {showReviewModal && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: "rgba(0, 0, 0, 0.6)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 1001,
            backdropFilter: "blur(4px)",
            overflowY: "auto",
            padding: "20px",
          }}
        >
          <div
            style={{
              backgroundColor: "#ffffff",
              borderRadius: "16px",
              maxWidth: "600px",
              width: "100%",
              maxHeight: "90vh",
              boxShadow: "0 20px 40px rgba(0, 0, 0, 0.2)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "24px",
                borderBottom: "1px solid #e5e7eb",
                background: "linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)",
                color: "white",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                }}
              >
                <div style={{ flex: 1 }}>
                  <h3
                    style={{
                      fontSize: "24px",
                      fontWeight: "700",
                      margin: "0 0 8px",
                      display: "flex",
                      alignItems: "center",
                      gap: "10px",
                    }}
                  >
                    <Star style={{ width: "28px", height: "28px" }} />
                    Write a Review
                  </h3>
                  <p style={{ fontSize: "14px", opacity: 0.9, margin: 0 }}>
                    Share your experience at {currentReview.spot_name}
                  </p>
                </div>
                <button
                  onClick={closeReviewModal}
                  style={{
                    background: "rgba(255, 255, 255, 0.2)",
                    border: "none",
                    color: "white",
                    cursor: "pointer",
                    padding: "8px",
                    borderRadius: "8px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.2s ease",
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.background =
                      "rgba(255, 255, 255, 0.3)")
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.background =
                      "rgba(255, 255, 255, 0.2)")
                  }
                >
                  <X style={{ width: "24px", height: "24px" }} />
                </button>
              </div>
            </div>

            <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
              <StarRating
                value={currentReview.rating}
                onChange={(value) =>
                  setCurrentReview({ ...currentReview, rating: value })
                }
                label="Overall Rating"
              />

              <div style={{ marginBottom: "16px" }}>
                <label
                  style={{
                    display: "block",
                    fontSize: "14px",
                    fontWeight: "500",
                    color: "#374151",
                    marginBottom: "8px",
                  }}
                >
                  Your Review (Optional)
                </label>
                <textarea
                  value={currentReview.comment}
                  onChange={(e) =>
                    setCurrentReview({
                      ...currentReview,
                      comment: e.target.value,
                    })
                  }
                  placeholder="Share details about your parking experience..."
                  maxLength={1000}
                  style={{
                    width: "100%",
                    minHeight: "120px",
                    padding: "12px",
                    border: "1px solid #d1d5db",
                    borderRadius: "8px",
                    fontSize: "14px",
                    fontFamily: "inherit",
                    resize: "vertical",
                    outline: "none",
                    transition: "border-color 0.2s ease",
                  }}
                  onFocus={(e) => (e.target.style.borderColor = "#2563eb")}
                  onBlur={(e) => (e.target.style.borderColor = "#d1d5db")}
                />
                <div
                  style={{
                    display: "flex",
                    justifyContent: "flex-end",
                    marginTop: "4px",
                  }}
                >
                  <span style={{ fontSize: "12px", color: "#6b7280" }}>
                    {currentReview.comment.length}/1000 characters
                  </span>
                </div>
              </div>

              <div
                style={{
                  backgroundColor: "#f9fafb",
                  borderRadius: "12px",
                  padding: "20px",
                  border: "1px solid #e5e7eb",
                  marginBottom: "16px",
                }}
              >
                <h4
                  style={{
                    fontSize: "16px",
                    fontWeight: "600",
                    color: "#111827",
                    margin: "0 0 16px",
                  }}
                >
                  Additional Ratings (Optional)
                </h4>

                <StarRating
                  value={currentReview.cleanliness_rating}
                  onChange={(value) =>
                    setCurrentReview({
                      ...currentReview,
                      cleanliness_rating: value,
                    })
                  }
                  label="Cleanliness"
                />
                <StarRating
                  value={currentReview.safety_rating}
                  onChange={(value) =>
                    setCurrentReview({ ...currentReview, safety_rating: value })
                  }
                  label="Safety"
                />
                <StarRating
                  value={currentReview.accessibility_rating}
                  onChange={(value) =>
                    setCurrentReview({
                      ...currentReview,
                      accessibility_rating: value,
                    })
                  }
                  label="Accessibility"
                />
              </div>

              <div
                style={{
                  backgroundColor: "#eff6ff",
                  border: "1px solid #dbeafe",
                  borderRadius: "8px",
                  padding: "12px",
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "8px",
                }}
              >
                <AlertCircle
                  style={{
                    width: "20px",
                    height: "20px",
                    color: "#2563eb",
                    flexShrink: 0,
                    marginTop: "2px",
                  }}
                />
                <p
                  style={{
                    fontSize: "13px",
                    color: "#1e40af",
                    margin: 0,
                    lineHeight: "1.5",
                  }}
                >
                  Your review helps other users make informed decisions and
                  helps parking owners improve their services.
                </p>
              </div>
            </div>

            <div
              style={{
                padding: "24px",
                borderTop: "1px solid #e5e7eb",
                background: "#f9fafb",
                display: "flex",
                gap: "12px",
                justifyContent: "flex-end",
              }}
            >
              <button
                onClick={closeReviewModal}
                disabled={reviewSubmitting}
                style={{
                  padding: "10px 24px",
                  backgroundColor: "#ffffff",
                  border: "1px solid #d1d5db",
                  borderRadius: "8px",
                  fontWeight: "500",
                  color: "#374151",
                  cursor: reviewSubmitting ? "not-allowed" : "pointer",
                  transition: "all 0.2s ease",
                  opacity: reviewSubmitting ? 0.5 : 1,
                }}
                onMouseOver={(e) => {
                  if (!reviewSubmitting) {
                    e.currentTarget.style.backgroundColor = "#f3f4f6";
                    e.currentTarget.style.borderColor = "#9ca3af";
                  }
                }}
                onMouseOut={(e) => {
                  if (!reviewSubmitting) {
                    e.currentTarget.style.backgroundColor = "#ffffff";
                    e.currentTarget.style.borderColor = "#d1d5db";
                  }
                }}
              >
                Cancel
              </button>

              <button
                onClick={submitReview}
                disabled={reviewSubmitting || currentReview.rating === 0}
                style={{
                  padding: "10px 24px",
                  backgroundColor:
                    currentReview.rating === 0 ? "#9ca3af" : "#2563eb",
                  border: "none",
                  borderRadius: "8px",
                  fontWeight: "500",
                  color: "white",
                  cursor:
                    reviewSubmitting || currentReview.rating === 0
                      ? "not-allowed"
                      : "pointer",
                  transition: "all 0.2s ease",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
                onMouseOver={(e) => {
                  if (!reviewSubmitting && currentReview.rating !== 0) {
                    e.currentTarget.style.backgroundColor = "#1d4ed8";
                  }
                }}
                onMouseOut={(e) => {
                  if (!reviewSubmitting && currentReview.rating !== 0) {
                    e.currentTarget.style.backgroundColor = "#2563eb";
                  }
                }}
              >
                {reviewSubmitting ? (
                  <>
                    <div
                      style={{
                        width: "16px",
                        height: "16px",
                        border: "2px solid #ffffff",
                        borderTop: "2px solid transparent",
                        borderRadius: "50%",
                        animation: "spin 1s linear infinite",
                      }}
                    />
                    <span>Submitting...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle style={{ width: "16px", height: "16px" }} />
                    <span>Submit Review</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }

        textarea::-webkit-scrollbar {
          width: 8px;
        }

        textarea::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 4px;
        }

        textarea::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 4px;
        }

        textarea::-webkit-scrollbar-thumb:hover {
          background: #94a3b8;
        }

        * {
          box-sizing: border-box;
        }
      `}</style>
    </div>
  );
}

export default UserReservations;
