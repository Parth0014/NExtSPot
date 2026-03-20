import React, { useState, useEffect } from "react";

function UserReviews() {
  const [myReviews, setMyReviews] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [editingReview, setEditingReview] = useState(null);
  const [editForm, setEditForm] = useState({
    rating: 5,
    comment: "",
    cleanliness_rating: null,
    safety_rating: null,
    accessibility_rating: null,
  });
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [showWriteReview, setShowWriteReview] = useState(false);
  const [selectedParkingSpot, setSelectedParkingSpot] = useState(null);
  const [bookingId, setBookingId] = useState(null);
  const [newReview, setNewReview] = useState({
    rating: 5,
    comment: "",
    cleanliness_rating: null,
    safety_rating: null,
    accessibility_rating: null,
  });
  const [canReviewSpots, setCanReviewSpots] = useState({});
  const [sortBy, setSortBy] = useState("newest");
  const [searchQuery, setSearchQuery] = useState("");

  const getToken = () => localStorage.getItem("token");

  const fetchMyReviews = async () => {
    const token = getToken();
    if (!token) {
      setMessage("Please log in to view your reviews");
      setIsLoading(false);
      return;
    }

    try {
      const res = await fetch("http://localhost:5000/reviews/my-reviews", {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.message || "Failed to fetch reviews");
      }

      const data = await res.json();
      setMyReviews(data);
      setMessage("");
    } catch (err) {
      setMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const checkIfCanReview = async (parkingId) => {
    const token = getToken();
    if (!token) return;

    try {
      const res = await fetch(
        `http://localhost:5000/reviews/can-review/${parkingId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );

      if (res.ok) {
        const data = await res.json();
        setCanReviewSpots((prev) => ({
          ...prev,
          [parkingId]: data,
        }));
        return data;
      }
    } catch (err) {
      console.error("Error checking review eligibility:", err);
    }
    return null;
  };

  const submitNewReview = async () => {
    const token = getToken();
    if (!token) {
      setMessage("Please log in to submit review");
      return;
    }

    if (!selectedParkingSpot || !bookingId) {
      setMessage("Parking spot and booking information required");
      return;
    }

    if (newReview.rating < 1 || newReview.rating > 5) {
      setMessage("Rating must be between 1 and 5");
      return;
    }

    try {
      const res = await fetch("http://localhost:5000/reviews/submit", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          parking_id: selectedParkingSpot.id,
          booking_id: bookingId,
          rating: newReview.rating,
          comment: newReview.comment || null,
          cleanliness_rating: newReview.cleanliness_rating,
          safety_rating: newReview.safety_rating,
          accessibility_rating: newReview.accessibility_rating,
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "Failed to submit review");

      setMessage("Review submitted successfully!");
      closeWriteReview();
      fetchMyReviews();
      setTimeout(() => setMessage(""), 3000);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const closeWriteReview = () => {
    setShowWriteReview(false);
    setSelectedParkingSpot(null);
    setBookingId(null);
    setNewReview({
      rating: 5,
      comment: "",
      cleanliness_rating: null,
      safety_rating: null,
      accessibility_rating: null,
    });
  };

  const openWriteReview = (spot) => {
    if (spot) {
      const existingReview = myReviews.find((r) => r.parking_id === spot.id);
      if (existingReview) {
        setMessage(
          `You already have a review for ${spot.name}. You can edit or delete it instead.`,
        );
        return;
      }
    }

    setSelectedParkingSpot(spot);
    setShowWriteReview(true);

    const params = new URLSearchParams(window.location.search);
    const bookingIdFromUrl = params.get("bookingId");
    if (bookingIdFromUrl) {
      setBookingId(parseInt(bookingIdFromUrl));
    }
  };

  const startEdit = (review) => {
    setEditingReview(review.id);
    setEditForm({
      rating: review.rating,
      comment: review.comment || "",
      cleanliness_rating: review.cleanliness_rating,
      safety_rating: review.safety_rating,
      accessibility_rating: review.accessibility_rating,
    });
  };

  const cancelEdit = () => {
    setEditingReview(null);
    setEditForm({
      rating: 5,
      comment: "",
      cleanliness_rating: null,
      safety_rating: null,
      accessibility_rating: null,
    });
  };

  const handleUpdate = async (reviewId) => {
    const token = getToken();
    if (!token) {
      setMessage("Please log in to update review");
      return;
    }

    if (editForm.rating < 1 || editForm.rating > 5) {
      setMessage("Rating must be between 1 and 5");
      return;
    }

    try {
      const res = await fetch(`http://localhost:5000/reviews/${reviewId}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(editForm),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "Failed to update review");

      setMessage("Review updated successfully!");
      cancelEdit();
      fetchMyReviews();
      setTimeout(() => setMessage(""), 3000);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const handleDelete = async (reviewId) => {
    const token = getToken();
    if (!token) {
      setMessage("Please log in to delete review");
      return;
    }

    try {
      const res = await fetch(`http://localhost:5000/reviews/${reviewId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "Failed to delete review");

      setMessage("Review deleted successfully!");
      setDeleteConfirm(null);
      fetchMyReviews();
      setTimeout(() => setMessage(""), 3000);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  const renderStars = (
    rating,
    interactive = false,
    onRate = null,
    size = "20px",
  ) => {
    return (
      <div style={{ display: "flex", gap: "2px" }}>
        {[1, 2, 3, 4, 5].map((star) => (
          <svg
            key={star}
            onClick={() => interactive && onRate && onRate(star)}
            style={{
              width: size,
              height: size,
              fill: star <= rating ? "#FDB022" : "#E0E0E0",
              cursor: interactive ? "pointer" : "default",
              transition: "all 0.2s ease",
            }}
            viewBox="0 0 20 20"
          >
            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
          </svg>
        ))}
      </div>
    );
  };

  const getSortedReviews = () => {
    let filtered = myReviews.filter((review) =>
      review.spot_name.toLowerCase().includes(searchQuery.toLowerCase()),
    );

    switch (sortBy) {
      case "oldest":
        return filtered.sort(
          (a, b) => new Date(a.created_at) - new Date(b.created_at),
        );
      case "highest":
        return filtered.sort((a, b) => b.rating - a.rating);
      default: // newest
        return filtered.sort(
          (a, b) => new Date(b.created_at) - new Date(a.created_at),
        );
    }
  };

  useEffect(() => {
    fetchMyReviews();
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const spotId = params.get("spotId");
    const bookingIdFromUrl = params.get("bookingId");

    if (spotId && myReviews.length > 0) {
      const existingReview = myReviews.find(
        (r) => r.parking_id === parseInt(spotId),
      );
      if (existingReview) {
        setMessage(
          "You have already reviewed this parking spot. You can edit or delete your review.",
        );
        return;
      }

      checkIfCanReview(parseInt(spotId)).then((canReview) => {
        if (canReview && canReview.canReview) {
          const spotData = { id: parseInt(spotId), name: "Parking Spot" };
          setSelectedParkingSpot(spotData);
          setShowWriteReview(true);
          if (bookingIdFromUrl) {
            setBookingId(parseInt(bookingIdFromUrl));
          }
        } else {
          setMessage(
            "You cannot review this parking spot. You need a completed booking.",
          );
        }
      });
    }
  }, [myReviews]);

  if (isLoading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "100vh",
          background: "#F5F5F5",
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              width: "48px",
              height: "48px",
              border: "4px solid #E0E0E0",
              borderTop: "4px solid #5B6FE4",
              borderRadius: "50%",
              animation: "spin 1s linear infinite",
              margin: "0 auto 16px",
            }}
          ></div>
          <p style={{ color: "#666", fontSize: "14px" }}>
            Loading your reviews...
          </p>
        </div>
      </div>
    );
  }

  const avgRating =
    myReviews.length > 0
      ? (
          myReviews.reduce((sum, r) => sum + r.rating, 0) / myReviews.length
        ).toFixed(1)
      : "0.0";

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#F5F5F5",
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      {/* Main Content */}
      <div
        style={{ maxWidth: "1200px", margin: "0 auto", padding: "32px 48px" }}
      >
        {/* Back to Dashboard Button */}
        <div style={{ marginBottom: "24px" }}>
          <button
            onClick={() => (window.location.href = "/user/dashboard")}
            style={{
              padding: "8px 16px",
              background: "white",
              color: "#6F767E",
              border: "1px solid #E8EBED",
              borderRadius: "8px",
              fontSize: "14px",
              fontWeight: "500",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              transition: "all 0.2s",
            }}
          >
            <svg
              style={{ width: "16px", height: "16px" }}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
            Back to Dashboard
          </button>
        </div>

        {/* Header */}
        <div style={{ marginBottom: "32px" }}>
          <h1
            style={{
              margin: "0 0 8px",
              fontSize: "32px",
              fontWeight: "700",
              color: "#1A1D1F",
            }}
          >
            My Reviews
          </h1>
          <p
            style={{
              margin: 0,
              fontSize: "15px",
              color: "#6F767E",
              lineHeight: "1.5",
            }}
          >
            Manage the feedback you've left for parking spots. Your reviews help
            our community find peace of mind.
          </p>
        </div>

        {/* Message Banner */}
        {message && (
          <div
            style={{
              background: message.includes("success") ? "#F0FDF4" : "#FEF2F2",
              border: `1px solid ${message.includes("success") ? "#BBF7D0" : "#FECACA"}`,
              borderRadius: "8px",
              padding: "12px 16px",
              marginBottom: "24px",
              color: message.includes("success") ? "#15803D" : "#DC2626",
              fontSize: "14px",
            }}
          >
            {message}
          </div>
        )}

        {/* Stats Cards */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "20px",
            marginBottom: "32px",
          }}
        >
          <div
            style={{
              background: "white",
              borderRadius: "12px",
              padding: "24px",
              border: "1px solid #E8EBED",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                marginBottom: "8px",
              }}
            >
              <div
                style={{
                  width: "40px",
                  height: "40px",
                  borderRadius: "8px",
                  background: "#EFF3FE",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <svg
                  style={{ width: "20px", height: "20px", color: "#5B6FE4" }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                  />
                </svg>
              </div>
              <span
                style={{
                  fontSize: "14px",
                  color: "#6F767E",
                  fontWeight: "500",
                }}
              >
                TOTAL REVIEWS
              </span>
            </div>
            <div
              style={{ display: "flex", alignItems: "baseline", gap: "8px" }}
            >
              <span
                style={{
                  fontSize: "36px",
                  fontWeight: "700",
                  color: "#1A1D1F",
                }}
              >
                {myReviews.length}
              </span>
              <span style={{ fontSize: "14px", color: "#9A9FA5" }}>
                contributions
              </span>
            </div>
          </div>

          <div
            style={{
              background: "white",
              borderRadius: "12px",
              padding: "24px",
              border: "1px solid #E8EBED",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                marginBottom: "8px",
              }}
            >
              <div
                style={{
                  width: "40px",
                  height: "40px",
                  borderRadius: "8px",
                  background: "#FEF3E2",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <svg
                  style={{ width: "20px", height: "20px", color: "#FDB022" }}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
              </div>
              <span
                style={{
                  fontSize: "14px",
                  color: "#6F767E",
                  fontWeight: "500",
                }}
              >
                AVG. RATING
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
              <span
                style={{
                  fontSize: "36px",
                  fontWeight: "700",
                  color: "#1A1D1F",
                }}
              >
                {avgRating}
              </span>
              {renderStars(
                Math.round(parseFloat(avgRating)),
                false,
                null,
                "16px",
              )}
            </div>
          </div>
        </div>

        {/* Filters and Search */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "24px",
          }}
        >
          <div style={{ display: "flex", gap: "8px" }}>
            <button
              onClick={() => setSortBy("newest")}
              style={{
                padding: "8px 16px",
                background: sortBy === "newest" ? "#3D4956" : "white",
                color: sortBy === "newest" ? "white" : "#6F767E",
                border: "1px solid #E8EBED",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: "500",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              Newest
            </button>
            <button
              onClick={() => setSortBy("oldest")}
              style={{
                padding: "8px 16px",
                background: sortBy === "oldest" ? "#3D4956" : "white",
                color: sortBy === "oldest" ? "white" : "#6F767E",
                border: "1px solid #E8EBED",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: "500",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              Oldest
            </button>
            <button
              onClick={() => setSortBy("highest")}
              style={{
                padding: "8px 16px",
                background: sortBy === "highest" ? "#3D4956" : "white",
                color: sortBy === "highest" ? "white" : "#6F767E",
                border: "1px solid #E8EBED",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: "500",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              Highest Rated
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div style={{ position: "relative" }}>
              <svg
                style={{
                  position: "absolute",
                  left: "12px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  width: "18px",
                  height: "18px",
                  color: "#9A9FA5",
                }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
              <input
                type="text"
                placeholder="Search your reviews..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  padding: "8px 12px 8px 40px",
                  border: "1px solid #E8EBED",
                  borderRadius: "8px",
                  fontSize: "14px",
                  width: "250px",
                  outline: "none",
                }}
              />
            </div>

            <button
              onClick={() => {
                fetch("http://localhost:5000/api/bookings/my-reservations", {
                  headers: {
                    Authorization: `Bearer ${localStorage.getItem("token")}`,
                  },
                })
                  .then((res) => res.json())
                  .then((data) => {
                    if (Array.isArray(data)) {
                      const history = data.filter(
                        (r) =>
                          r.status !== "active" && r.status !== "scheduled",
                      );
                      if (history.length > 0) {
                        history.sort(
                          (a, b) =>
                            new Date(b.reserved_at) - new Date(a.reserved_at),
                        );
                        const latest = history[0];
                        window.location.href = `/user/reservations?bookingId=${latest.booking_id}`;
                      } else {
                        window.location.href =
                          "/user/reservations?bookingId=history";
                      }
                    } else {
                      window.location.href =
                        "/user/reservations?bookingId=history";
                    }
                  })
                  .catch(() => {
                    window.location.href =
                      "/user/reservations?bookingId=history";
                  });
              }}
              style={{
                padding: "8px 16px",
                background: "#3D4956",
                color: "white",
                border: "none",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: "600",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                transition: "all 0.2s",
              }}
            >
              <svg
                style={{ width: "16px", height: "16px" }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                />
              </svg>
              Write a Review
            </button>
          </div>
        </div>

        {/* Reviews List */}
        {getSortedReviews().length === 0 ? (
          <div
            style={{
              background: "white",
              borderRadius: "12px",
              padding: "80px 40px",
              textAlign: "center",
              border: "1px solid #E8EBED",
            }}
          >
            <div
              style={{
                width: "80px",
                height: "80px",
                background: "#F5F5F5",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 24px",
              }}
            >
              <svg
                style={{ width: "40px", height: "40px", color: "#9A9FA5" }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                />
              </svg>
            </div>
            <h3
              style={{
                fontSize: "20px",
                fontWeight: "600",
                color: "#1A1D1F",
                margin: "0 0 12px",
              }}
            >
              {searchQuery ? "No reviews found" : "No Reviews Yet"}
            </h3>
            <p
              style={{ fontSize: "14px", color: "#6F767E", margin: "0 0 24px" }}
            >
              {searchQuery
                ? "Try adjusting your search"
                : "You haven't written any reviews yet. Complete a booking to leave your first review!"}
            </p>
          </div>
        ) : (
          <div
            style={{ display: "flex", flexDirection: "column", gap: "16px" }}
          >
            {getSortedReviews().map((review) => (
              <div
                key={review.id}
                style={{
                  background: "white",
                  borderRadius: "12px",
                  padding: "24px",
                  border: "1px solid #E8EBED",
                }}
              >
                {editingReview === review.id ? (
                  // Edit Mode
                  <div>
                    <h3
                      style={{
                        fontSize: "18px",
                        fontWeight: "600",
                        color: "#1A1D1F",
                        margin: "0 0 20px",
                        paddingBottom: "16px",
                        borderBottom: "1px solid #E8EBED",
                      }}
                    >
                      Editing Review for {review.spot_name}
                    </h3>

                    <div style={{ marginBottom: "20px" }}>
                      <label
                        style={{
                          display: "block",
                          fontSize: "13px",
                          fontWeight: "600",
                          color: "#1A1D1F",
                          marginBottom: "8px",
                        }}
                      >
                        Overall Rating *
                      </label>
                      {renderStars(editForm.rating, true, (rating) =>
                        setEditForm({ ...editForm, rating }),
                      )}
                    </div>

                    <div style={{ marginBottom: "20px" }}>
                      <label
                        style={{
                          display: "block",
                          fontSize: "13px",
                          fontWeight: "600",
                          color: "#1A1D1F",
                          marginBottom: "8px",
                        }}
                      >
                        Your Review
                      </label>
                      <textarea
                        value={editForm.comment}
                        onChange={(e) =>
                          setEditForm({ ...editForm, comment: e.target.value })
                        }
                        maxLength={1000}
                        placeholder="Share your experience..."
                        style={{
                          width: "100%",
                          minHeight: "120px",
                          padding: "12px",
                          border: "1px solid #E8EBED",
                          borderRadius: "8px",
                          fontSize: "14px",
                          fontFamily: "inherit",
                          resize: "vertical",
                          outline: "none",
                        }}
                      />
                      <p
                        style={{
                          fontSize: "12px",
                          color: "#9A9FA5",
                          margin: "6px 0 0",
                          textAlign: "right",
                        }}
                      >
                        {editForm.comment.length}/1000
                      </p>
                    </div>

                    <div
                      style={{
                        display: "flex",
                        gap: "12px",
                        justifyContent: "flex-end",
                        paddingTop: "16px",
                        borderTop: "1px solid #E8EBED",
                      }}
                    >
                      <button
                        onClick={cancelEdit}
                        style={{
                          padding: "10px 20px",
                          background: "white",
                          color: "#6F767E",
                          border: "1px solid #E8EBED",
                          borderRadius: "8px",
                          fontSize: "14px",
                          fontWeight: "600",
                          cursor: "pointer",
                        }}
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => handleUpdate(review.id)}
                        style={{
                          padding: "10px 20px",
                          background: "#3D4956",
                          color: "white",
                          border: "none",
                          borderRadius: "8px",
                          fontSize: "14px",
                          fontWeight: "600",
                          cursor: "pointer",
                        }}
                      >
                        Save Changes
                      </button>
                    </div>
                  </div>
                ) : (
                  // View Mode
                  <div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "flex-start",
                        justifyContent: "space-between",
                        marginBottom: "16px",
                      }}
                    >
                      <div style={{ display: "flex", gap: "16px", flex: 1 }}>
                        <div
                          style={{
                            width: "72px",
                            height: "72px",
                            borderRadius: "8px",
                            background: "#F5F5F5",
                            flexShrink: 0,
                          }}
                        />
                        <div style={{ flex: 1 }}>
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                              marginBottom: "4px",
                            }}
                          >
                            <h3
                              style={{
                                fontSize: "18px",
                                fontWeight: "600",
                                color: "#1A1D1F",
                                margin: 0,
                              }}
                            >
                              {review.spot_name}
                            </h3>
                            <button
                              onClick={() =>
                                (window.location.href = `/user/spot/${review.parking_id}/reviews`)
                              }
                              style={{
                                padding: "4px 8px",
                                background: "transparent",
                                color: "#5B6FE4",
                                border: "1px solid #5B6FE4",
                                borderRadius: "6px",
                                fontSize: "12px",
                                fontWeight: "600",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: "4px",
                              }}
                            >
                              <svg
                                style={{ width: "12px", height: "12px" }}
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  strokeWidth={2}
                                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                                />
                              </svg>
                              View All
                            </button>
                          </div>
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                              marginBottom: "8px",
                            }}
                          >
                            {renderStars(review.rating, false, null, "16px")}
                            <span
                              style={{ fontSize: "13px", color: "#9A9FA5" }}
                            >
                              • {formatDate(review.created_at)}
                            </span>
                          </div>
                          {review.comment && (
                            <p
                              style={{
                                fontSize: "14px",
                                color: "#6F767E",
                                lineHeight: "1.6",
                                margin: 0,
                              }}
                            >
                              {review.comment}
                            </p>
                          )}
                        </div>
                      </div>

                      <div style={{ display: "flex", gap: "8px" }}>
                        <button
                          onClick={() => startEdit(review)}
                          style={{
                            width: "36px",
                            height: "36px",
                            background: "transparent",
                            border: "1px solid #E8EBED",
                            borderRadius: "8px",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            color: "#6F767E",
                          }}
                        >
                          <svg
                            style={{ width: "16px", height: "16px" }}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"
                            />
                          </svg>
                        </button>
                        <button
                          onClick={() => setDeleteConfirm(review.id)}
                          style={{
                            width: "36px",
                            height: "36px",
                            background: "transparent",
                            border: "1px solid #E8EBED",
                            borderRadius: "8px",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            color: "#6F767E",
                          }}
                        >
                          <svg
                            style={{ width: "16px", height: "16px" }}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Delete Confirmation Modal */}
                {deleteConfirm === review.id && (
                  <div
                    style={{
                      position: "fixed",
                      inset: 0,
                      background: "rgba(0, 0, 0, 0.5)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      zIndex: 10000,
                    }}
                  >
                    <div
                      style={{
                        background: "white",
                        borderRadius: "12px",
                        padding: "32px",
                        maxWidth: "400px",
                        width: "90%",
                      }}
                    >
                      <h3
                        style={{
                          fontSize: "20px",
                          fontWeight: "600",
                          color: "#1A1D1F",
                          margin: "0 0 8px",
                        }}
                      >
                        Delete Review?
                      </h3>
                      <p
                        style={{
                          fontSize: "14px",
                          color: "#6F767E",
                          margin: "0 0 24px",
                          lineHeight: "1.5",
                        }}
                      >
                        Are you sure you want to delete this review? This action
                        cannot be undone.
                      </p>
                      <div
                        style={{
                          display: "flex",
                          gap: "12px",
                          justifyContent: "flex-end",
                        }}
                      >
                        <button
                          onClick={() => setDeleteConfirm(null)}
                          style={{
                            padding: "10px 20px",
                            background: "white",
                            color: "#6F767E",
                            border: "1px solid #E8EBED",
                            borderRadius: "8px",
                            fontSize: "14px",
                            fontWeight: "600",
                            cursor: "pointer",
                          }}
                        >
                          Cancel
                        </button>
                        <button
                          onClick={() => handleDelete(review.id)}
                          style={{
                            padding: "10px 20px",
                            background: "#DC2626",
                            color: "white",
                            border: "none",
                            borderRadius: "8px",
                            fontSize: "14px",
                            fontWeight: "600",
                            cursor: "pointer",
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {getSortedReviews().length > 0 && (
          <div style={{ textAlign: "center", marginTop: "32px" }}>
            <button
              style={{
                padding: "10px 24px",
                background: "white",
                color: "#6F767E",
                border: "1px solid #E8EBED",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: "600",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
              }}
            >
              Load More Reviews
              <svg
                style={{ width: "16px", height: "16px" }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </button>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default UserReviews;
