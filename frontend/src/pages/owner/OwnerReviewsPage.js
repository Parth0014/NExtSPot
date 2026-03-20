import React, { useState, useEffect } from "react";
import {
  Star,
  MessageSquare,
  Send,
  Search,
  MapPin,
  ThumbsUp,
  Calendar,
  Lightbulb,
} from "lucide-react";

function OwnerReviewsPage() {
  const [reviews, setReviews] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [responseText, setResponseText] = useState({});
  const [filterSpot, setFilterSpot] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState("newest");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [spots, setSpots] = useState([]);
  const token = localStorage.getItem("token");

  // Fetch owner's parking spots
  useEffect(() => {
    fetch("http://localhost:5000/reviews/owner/spots", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => setSpots(data))
      .catch((err) => {
        console.error("Error fetching spots:", err);
        setMessage("Could not load parking spots");
      });
  }, [token]);

  // Fetch reviews and summary
  useEffect(() => {
    fetchReviews();
    fetchSummary();
  }, [currentPage, filterSpot, token]);

  const fetchReviews = () => {
    setLoading(true);

    // Build query string with spot filter if not "all"
    let url = `http://localhost:5000/reviews/owner/reviews?page=${currentPage}&limit=10`;
    if (filterSpot && filterSpot !== "all") {
      url += `&spot_id=${filterSpot}`;
    }

    console.log("Fetching reviews from:", url); // Debug log
    console.log("Filter spot:", filterSpot); // Debug log

    fetch(url, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        console.log("Received reviews:", data); // Debug log
        setReviews(data.reviews || []);
        setTotalPages(data.pagination?.totalPages || 1);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching reviews:", err);
        setMessage("Failed to load reviews");
        setLoading(false);
      });
  };

  const fetchSummary = () => {
    // For now, always fetch overall summary
    // If you want spot-specific summary, you'd need a new backend endpoint
    fetch("http://localhost:5000/reviews/owner/summary", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => setSummary(data))
      .catch((err) => console.error("Error fetching summary:", err));
  };

  const handleRespond = (reviewId) => {
    const text = responseText[reviewId];
    if (!text || text.trim().length === 0) {
      setMessage("Please enter a response");
      return;
    }

    fetch(`http://localhost:5000/reviews/${reviewId}/respond`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ response_text: text }),
    })
      .then((res) => res.json())
      .then((data) => {
        setMessage(data.message || "Response submitted successfully");
        setResponseText({ ...responseText, [reviewId]: "" });
        fetchReviews();
        fetchSummary();
      })
      .catch(() => setMessage("Failed to submit response"));
  };

  const renderStars = (rating) => {
    return [...Array(5)].map((_, i) => (
      <Star
        key={i}
        size={18}
        fill={i < rating ? "#fbbf24" : "none"}
        stroke={i < rating ? "#fbbf24" : "#e2e8f0"}
      />
    ));
  };

  const getInitials = (name) => {
    return name
      ?.split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  // Filter reviews by search term
  const filteredReviews = reviews.filter((review) => {
    const matchesSearch =
      review.comment?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      review.user_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      review.spot_name?.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesSearch;
  });

  // Sort reviews
  const sortedReviews = React.useMemo(() => {
    const arr = [...filteredReviews];
    switch (sortBy) {
      case "newest":
        return arr.sort(
          (a, b) => new Date(b.created_at) - new Date(a.created_at),
        );
      case "oldest":
        return arr.sort(
          (a, b) => new Date(a.created_at) - new Date(b.created_at),
        );
      case "highest":
        return arr.sort((a, b) => (b.rating || 0) - (a.rating || 0));
      case "lowest":
        return arr.sort((a, b) => (a.rating || 0) - (b.rating || 0));
      default:
        return arr;
    }
  }, [filteredReviews, sortBy]);

  const getRatingLabel = (rating) => {
    if (!rating) return "N/A";
    if (rating >= 4.5) return "Excellent";
    if (rating >= 3.5) return "Good";
    if (rating >= 2.5) return "Average";
    return "Needs Work";
  };

  const getRatingColor = (rating) => {
    if (!rating) return "#e2e8f0";
    if (rating >= 4.5) return "#86efac";
    if (rating >= 3.5) return "#93c5fd";
    if (rating >= 2.5) return "#fde047";
    return "#fca5a5";
  };

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <aside style={styles.sidebar}>
        <div style={styles.sidebarContent}>
          <div style={styles.logoSection}>
            <div style={styles.logo}>
              <MapPin size={20} color="#ffffff" />
            </div>
            <div>
              <h1 style={styles.logoTitle}>NExtSPot</h1>
              <p style={styles.logoSubtitle}>Owner Dashboard</p>
            </div>
          </div>

          <nav style={styles.nav}>
            <a href="/owner/reviews" style={styles.navItemActive}>
              <Star size={18} />
              <span>Reviews</span>
            </a>
            <a href="/owner/dashboard" style={styles.navItem}>
              <MapPin size={18} />
              <span>My Spots</span>
            </a>
            <a href="/owner/bookings" style={styles.navItem}>
              <Calendar size={18} />
              <span>Bookings</span>
            </a>
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <main style={styles.main}>
        <div style={styles.content}>
          {/* Header */}
          <div style={styles.pageHeader}>
            <div>
              <h1 style={styles.pageTitle}>Spot Review History</h1>
              <p style={styles.pageSubtitle}>
                Granular feedback & sentiment for your parking assets.
              </p>
            </div>
            <div style={styles.spotSelector}>
              <label style={styles.spotLabel}>SELECTED SPOT</label>
              <div style={styles.selectWrapper}>
                <select
                  value={filterSpot}
                  onChange={(e) => {
                    setFilterSpot(e.target.value);
                    setCurrentPage(1);
                  }}
                  style={styles.select}
                >
                  <option value="all">All Spots</option>
                  {spots.map((spot) => (
                    <option key={spot.id} value={spot.id}>
                      {spot.name}
                    </option>
                  ))}
                </select>
                <span style={styles.selectArrow}>▼</span>
              </div>
            </div>

            <div style={{ ...styles.spotSelector, minWidth: 160 }}>
              <label style={styles.spotLabel}>SORT</label>
              <div style={styles.selectWrapper}>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  style={styles.select}
                >
                  <option value="newest">Newest</option>
                  <option value="oldest">Oldest</option>
                  <option value="highest">Highest Rating</option>
                  <option value="lowest">Lowest Rating</option>
                </select>
                <span style={styles.selectArrow}>▼</span>
              </div>
            </div>
          </div>

          {/* Message */}
          {message && (
            <div
              style={{
                ...styles.message,
                backgroundColor: message.toLowerCase().includes("failed")
                  ? "#fef2f2"
                  : "#f0fdf4",
                color: message.toLowerCase().includes("failed")
                  ? "#dc2626"
                  : "#16a34a",
              }}
            >
              {message}
            </div>
          )}

          {/* Summary Cards */}
          {summary && (
            <div style={styles.summaryGrid}>
              <div style={styles.summaryCard}>
                <p style={styles.summaryCardLabel}>OVERALL RATING</p>
                <div style={styles.summaryCardContent}>
                  <span style={styles.summaryCardValue}>
                    {parseFloat(summary.average_rating || 0).toFixed(1)}
                  </span>
                  <div style={styles.starsSmall}>
                    {renderStars(Math.round(summary.average_rating || 0))}
                  </div>
                </div>
              </div>

              <div style={styles.summaryCard}>
                <p style={styles.summaryCardLabel}>CLEANLINESS</p>
                <div style={styles.summaryCardContent}>
                  <span style={styles.summaryCardValue}>
                    {summary.avg_cleanliness !== undefined &&
                    summary.avg_cleanliness !== null
                      ? parseFloat(summary.avg_cleanliness).toFixed(1)
                      : "-"}
                  </span>
                  <div style={styles.ratingBar}>
                    <div
                      style={{
                        ...styles.ratingBarFill,
                        width: `${((summary.avg_cleanliness || 0) / 5) * 100}%`,
                        backgroundColor: getRatingColor(
                          summary.avg_cleanliness,
                        ),
                      }}
                    ></div>
                  </div>
                  <span style={styles.ratingLabel}>
                    {getRatingLabel(summary.avg_cleanliness)}
                  </span>
                </div>
              </div>

              <div style={styles.summaryCard}>
                <p style={styles.summaryCardLabel}>EASE OF ACCESS</p>
                <div style={styles.summaryCardContent}>
                  <span style={styles.summaryCardValue}>
                    {summary.avg_accessibility !== undefined &&
                    summary.avg_accessibility !== null
                      ? parseFloat(summary.avg_accessibility).toFixed(1)
                      : "-"}
                  </span>
                  <div style={styles.ratingBar}>
                    <div
                      style={{
                        ...styles.ratingBarFill,
                        width: `${((summary.avg_accessibility || 0) / 5) * 100}%`,
                        backgroundColor: getRatingColor(
                          summary.avg_accessibility,
                        ),
                      }}
                    ></div>
                  </div>
                  <span style={styles.ratingLabel}>
                    {getRatingLabel(summary.avg_accessibility)}
                  </span>
                </div>
              </div>

              <div style={styles.summaryCard}>
                <p style={styles.summaryCardLabel}>TOTAL REVIEWS</p>
                <div style={styles.summaryCardContent}>
                  <span style={styles.summaryCardValue}>
                    {summary.total_reviews || 0}
                  </span>
                  <span style={styles.reviewsBadge}>
                    {summary.pending_response_count || 0} pending
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Search Bar */}
          <div style={styles.searchContainer}>
            <h2 style={styles.reviewsTitle}>
              Reviews ({sortedReviews.length})
              {filterSpot !== "all" &&
                ` for ${spots.find((s) => s.id == filterSpot)?.name || "Selected Spot"}`}
            </h2>
            <div style={styles.searchBox}>
              <Search size={18} style={styles.searchIcon} />
              <input
                type="text"
                placeholder="Search reviews..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={styles.searchInput}
              />
            </div>
          </div>

          {/* Reviews List */}
          <div style={styles.reviewsList}>
            {loading ? (
              <div style={styles.emptyState}>
                <div style={styles.emptyText}>Loading reviews...</div>
              </div>
            ) : sortedReviews.length === 0 ? (
              <div style={styles.emptyState}>
                <MessageSquare size={48} color="#cbd5e1" />
                <div style={styles.emptyTitle}>No reviews found</div>
                <div style={styles.emptyText}>
                  {searchTerm
                    ? "Try adjusting your search"
                    : "Reviews will appear here"}
                </div>
              </div>
            ) : (
              sortedReviews.map((review) => (
                <div key={review.id} style={styles.reviewCard}>
                  <div style={styles.reviewMain}>
                    {/* Review Header */}
                    <div style={styles.reviewHeader}>
                      <div style={styles.reviewUser}>
                        <div style={styles.reviewAvatar}>
                          {getInitials(review.user_name)}
                        </div>
                        <div>
                          <h4 style={styles.reviewUserName}>
                            {review.user_name}
                          </h4>
                          <p style={styles.reviewDate}>
                            {new Date(review.created_at).toLocaleDateString(
                              "en-US",
                              {
                                month: "short",
                                day: "numeric",
                                year: "numeric",
                              },
                            )}
                            {review.spot_name && (
                              <span style={styles.spotBadge}>
                                {" "}
                                • {review.spot_name}
                              </span>
                            )}
                          </p>
                        </div>
                      </div>
                      <div style={styles.reviewRating}>
                        <div style={styles.starsSmall}>
                          {renderStars(review.rating)}
                        </div>
                        <span style={styles.ratingText}>
                          {review.rating}.0 Overall
                        </span>
                      </div>
                    </div>

                    {/* Rating Details */}
                    {(review.cleanliness_rating ||
                      review.safety_rating ||
                      review.accessibility_rating) && (
                      <div style={styles.ratingBadges}>
                        {review.cleanliness_rating && (
                          <div
                            style={{
                              ...styles.ratingBadge,
                              backgroundColor: "#d1fae5",
                              borderColor: "#86efac",
                            }}
                          >
                            <span
                              style={{ ...styles.badgeLabel, color: "#065f46" }}
                            >
                              Cleanliness
                            </span>
                            <span
                              style={{ ...styles.badgeValue, color: "#059669" }}
                            >
                              {review.cleanliness_rating}/5
                            </span>
                          </div>
                        )}
                        {review.accessibility_rating && (
                          <div
                            style={{
                              ...styles.ratingBadge,
                              backgroundColor: "#dbeafe",
                              borderColor: "#93c5fd",
                            }}
                          >
                            <span
                              style={{ ...styles.badgeLabel, color: "#1e40af" }}
                            >
                              Access
                            </span>
                            <span
                              style={{ ...styles.badgeValue, color: "#1d4ed8" }}
                            >
                              {review.accessibility_rating}/5
                            </span>
                          </div>
                        )}
                        {review.safety_rating && (
                          <div
                            style={{
                              ...styles.ratingBadge,
                              backgroundColor: "#fef3c7",
                              borderColor: "#fde047",
                            }}
                          >
                            <span
                              style={{ ...styles.badgeLabel, color: "#92400e" }}
                            >
                              Safety
                            </span>
                            <span
                              style={{ ...styles.badgeValue, color: "#b45309" }}
                            >
                              {review.safety_rating}/5
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Review Comment */}
                    {review.comment && (
                      <p style={styles.reviewComment}>"{review.comment}"</p>
                    )}

                    {/* Owner Response */}
                    {review.my_response ? (
                      <div style={styles.responseBox}>
                        <div style={styles.responseHeader}>
                          <Send size={14} style={{ color: "#16a34a" }} />
                          <span style={styles.responseTitle}>
                            Your Response
                          </span>
                          <span style={styles.responseDate}>
                            {new Date(
                              review.response_date,
                            ).toLocaleDateString()}
                          </span>
                        </div>
                        <p style={styles.responseText}>{review.my_response}</p>
                      </div>
                    ) : (
                      <div style={styles.responseForm}>
                        <textarea
                          placeholder="Write your response to this review..."
                          value={responseText[review.id] || ""}
                          onChange={(e) =>
                            setResponseText({
                              ...responseText,
                              [review.id]: e.target.value,
                            })
                          }
                          style={styles.responseTextarea}
                        />
                        <button
                          onClick={() => handleRespond(review.id)}
                          style={styles.responseButton}
                        >
                          <Send size={14} />
                          Send Response
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Side Panel */}
                  <div style={styles.reviewSide}>
                    {review.rating >= 4 ? (
                      <div style={styles.sidePanelPositive}>
                        <div style={styles.sidePanelHeader}>
                          <ThumbsUp size={18} />
                          <span style={styles.sidePanelTitle}>
                            POSITIVE TREND
                          </span>
                        </div>
                        <p style={styles.sidePanelText}>
                          High ratings consistently received for this spot.
                        </p>
                      </div>
                    ) : (
                      <div style={styles.sidePanelWarning}>
                        <div style={styles.sidePanelHeader}>
                          <Lightbulb size={18} />
                          <span style={styles.sidePanelTitle}>
                            IMPROVEMENT SUGGESTION
                          </span>
                        </div>
                        <p style={styles.sidePanelText}>
                          Consider reviewing this feedback for potential
                          improvements.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={styles.pagination}>
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                style={{
                  ...styles.paginationButton,
                  ...(currentPage === 1 && styles.paginationButtonDisabled),
                }}
              >
                Previous
              </button>
              <span style={styles.paginationText}>
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={() =>
                  setCurrentPage((p) => Math.min(totalPages, p + 1))
                }
                disabled={currentPage === totalPages}
                style={{
                  ...styles.paginationButton,
                  ...(currentPage === totalPages &&
                    styles.paginationButtonDisabled),
                }}
              >
                Next
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    minHeight: "100vh",
    backgroundColor: "#F8FAFC",
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
  },

  // Sidebar
  sidebar: {
    width: "260px",
    backgroundColor: "#ffffff",
    borderRight: "1px solid #e5e7eb",
    display: "flex",
    flexDirection: "column",
    position: "sticky",
    top: 0,
    height: "100vh",
  },
  sidebarContent: {
    padding: "1.5rem 1.25rem",
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "1.75rem",
  },
  logoSection: {
    display: "flex",
    alignItems: "center",
    gap: "0.875rem",
  },
  logo: {
    width: "42px",
    height: "42px",
    backgroundColor: "#3b82f6",
    borderRadius: "10px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: "0 2px 8px rgba(59, 130, 246, 0.3)",
  },
  logoTitle: {
    fontSize: "1rem",
    fontWeight: "700",
    color: "#111827",
    margin: 0,
    lineHeight: 1.2,
  },
  logoSubtitle: {
    fontSize: "0.8125rem",
    color: "#6b7280",
    margin: "3px 0 0 0",
    fontWeight: "400",
  },
  nav: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  navItem: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    padding: "0.625rem 0.875rem",
    color: "#6b7280",
    textDecoration: "none",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "500",
    transition: "all 0.2s",
    cursor: "pointer",
    backgroundColor: "transparent",
  },
  navItemActive: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    padding: "0.625rem 0.875rem",
    backgroundColor: "#dbeafe",
    color: "#2563eb",
    textDecoration: "none",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "600",
    cursor: "pointer",
  },

  // Main Content
  main: {
    flex: 1,
    overflowY: "auto",
    backgroundColor: "#F8FAFC",
  },
  content: {
    maxWidth: "1100px",
    margin: "0 auto",
    padding: "2rem",
    display: "flex",
    flexDirection: "column",
    gap: "2.5rem",
  },

  // Page Header
  pageHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-end",
    gap: "1.5rem",
    flexWrap: "wrap",
  },
  pageTitle: {
    fontSize: "1.875rem",
    fontWeight: "700",
    color: "#1e293b",
    margin: "0 0 0.5rem 0",
    letterSpacing: "-0.02em",
  },
  pageSubtitle: {
    fontSize: "1rem",
    color: "#64748b",
    margin: 0,
    fontWeight: "400",
  },
  spotSelector: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
    minWidth: "250px",
  },
  spotLabel: {
    fontSize: "0.625rem",
    fontWeight: "700",
    color: "#94a3b8",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    paddingLeft: "0.25rem",
  },
  selectWrapper: {
    position: "relative",
  },
  select: {
    width: "100%",
    backgroundColor: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: "12px",
    padding: "0.625rem 2.5rem 0.625rem 1rem",
    fontSize: "0.875rem",
    fontWeight: "500",
    color: "#334155",
    cursor: "pointer",
    appearance: "none",
    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
    transition: "all 0.2s",
  },
  selectArrow: {
    position: "absolute",
    right: "12px",
    top: "50%",
    transform: "translateY(-50%)",
    pointerEvents: "none",
    color: "#94a3b8",
    fontSize: "0.75rem",
  },

  // Message
  message: {
    padding: "0.75rem 1rem",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "500",
  },

  // Summary Cards
  summaryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
    gap: "1.25rem",
  },
  summaryCard: {
    backgroundColor: "#ffffff",
    border: "1px solid #f1f5f9",
    borderRadius: "20px",
    padding: "1.5rem",
    boxShadow: "0 1px 3px rgba(0,0,0,0.05), 0 10px 15px -5px rgba(0,0,0,0.02)",
    transition: "all 0.3s",
  },
  summaryCardLabel: {
    fontSize: "0.625rem",
    fontWeight: "700",
    color: "#94a3b8",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    margin: "0 0 0.75rem 0",
  },
  summaryCardContent: {
    display: "flex",
    flexDirection: "column",
    gap: "0.75rem",
  },
  summaryCardValue: {
    fontSize: "2.25rem",
    fontWeight: "700",
    color: "#1e293b",
    lineHeight: 1,
  },
  starsSmall: {
    display: "flex",
    gap: "2px",
  },
  ratingBar: {
    width: "100%",
    height: "8px",
    backgroundColor: "#f1f5f9",
    borderRadius: "9999px",
    overflow: "hidden",
  },
  ratingBarFill: {
    height: "100%",
    borderRadius: "9999px",
    transition: "width 0.5s ease",
  },
  ratingLabel: {
    fontSize: "0.75rem",
    color: "#64748b",
    fontWeight: "500",
  },
  reviewsBadge: {
    fontSize: "0.75rem",
    color: "#4f7c7b",
    fontWeight: "600",
  },

  // Search
  searchContainer: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "1rem",
    flexWrap: "wrap",
  },
  reviewsTitle: {
    fontSize: "1.125rem",
    fontWeight: "600",
    color: "#1e293b",
    margin: 0,
  },
  searchBox: {
    position: "relative",
    width: "300px",
    maxWidth: "100%",
  },
  searchIcon: {
    position: "absolute",
    left: "12px",
    top: "50%",
    transform: "translateY(-50%)",
    color: "#94a3b8",
    pointerEvents: "none",
  },
  searchInput: {
    width: "100%",
    paddingLeft: "40px",
    paddingRight: "12px",
    paddingTop: "8px",
    paddingBottom: "8px",
    backgroundColor: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: "10px",
    fontSize: "0.875rem",
    outline: "none",
    boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
  },

  // Reviews List
  reviewsList: {
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },
  reviewCard: {
    backgroundColor: "#ffffff",
    border: "1px solid #f1f5f9",
    borderRadius: "20px",
    overflow: "hidden",
    boxShadow: "0 1px 3px rgba(0,0,0,0.05)",
    display: "flex",
    flexDirection: "row",
    transition: "all 0.3s",
  },
  reviewMain: {
    flex: 1,
    padding: "2rem",
    display: "flex",
    flexDirection: "column",
    gap: "1.25rem",
  },
  reviewHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  reviewUser: {
    display: "flex",
    alignItems: "center",
    gap: "1rem",
  },
  reviewAvatar: {
    width: "48px",
    height: "48px",
    borderRadius: "16px",
    backgroundColor: "#e0e7ff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "0.875rem",
    fontWeight: "700",
    color: "#4338ca",
    border: "1px solid #f1f5f9",
    boxShadow: "inset 0 2px 4px rgba(0,0,0,0.05)",
  },
  reviewUserName: {
    fontSize: "0.875rem",
    fontWeight: "700",
    color: "#1e293b",
    margin: 0,
  },
  reviewDate: {
    fontSize: "0.6875rem",
    color: "#94a3b8",
    margin: "2px 0 0 0",
  },
  spotBadge: {
    color: "#4f7c7b",
    fontWeight: "600",
  },
  reviewRating: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-end",
    gap: "4px",
  },
  ratingText: {
    fontSize: "0.6875rem",
    color: "#94a3b8",
    fontWeight: "500",
  },
  ratingBadges: {
    display: "flex",
    gap: "0.75rem",
    flexWrap: "wrap",
  },
  ratingBadge: {
    padding: "0.375rem 0.75rem",
    borderRadius: "8px",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    border: "1px solid",
  },
  badgeLabel: {
    fontSize: "0.6875rem",
    fontWeight: "600",
  },
  badgeValue: {
    fontSize: "0.75rem",
    fontWeight: "700",
  },
  reviewComment: {
    margin: 0,
    color: "#475569",
    fontSize: "0.875rem",
    lineHeight: "1.6",
    fontWeight: "400",
  },
  responseBox: {
    backgroundColor: "#f0fdf4",
    border: "1px solid #bbf7d0",
    borderRadius: "12px",
    padding: "1rem",
  },
  responseHeader: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    marginBottom: "0.5rem",
  },
  responseTitle: {
    fontSize: "0.75rem",
    fontWeight: "700",
    color: "#16a34a",
    letterSpacing: "0.05em",
    textTransform: "uppercase",
  },
  responseDate: {
    fontSize: "0.75rem",
    color: "#64748b",
    marginLeft: "auto",
  },
  responseText: {
    margin: 0,
    color: "#374151",
    fontSize: "0.875rem",
    lineHeight: "1.5",
  },
  responseForm: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },
  responseTextarea: {
    width: "100%",
    minHeight: "80px",
    padding: "0.75rem",
    border: "1px solid #e2e8f0",
    borderRadius: "10px",
    fontSize: "0.875rem",
    fontFamily: "inherit",
    resize: "vertical",
    outline: "none",
    boxSizing: "border-box",
  },
  responseButton: {
    padding: "0.625rem 1rem",
    backgroundColor: "#1e293b",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "500",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    alignSelf: "flex-start",
    transition: "background-color 0.2s",
  },

  // Side Panel
  reviewSide: {
    width: "288px",
    borderLeft: "1px solid #f1f5f9",
    padding: "2rem",
    display: "flex",
    alignItems: "center",
  },
  sidePanelPositive: {
    backgroundColor: "#d1fae5",
    padding: "1rem",
    borderRadius: "12px",
    border: "1px solid #86efac",
    width: "100%",
  },
  sidePanelWarning: {
    backgroundColor: "#fef3c7",
    padding: "1rem",
    borderRadius: "12px",
    border: "1px solid #fde047",
    width: "100%",
  },
  sidePanelHeader: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    color: "#065f46",
    marginBottom: "0.5rem",
  },
  sidePanelTitle: {
    fontSize: "0.625rem",
    fontWeight: "700",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
  },
  sidePanelText: {
    fontSize: "0.75rem",
    color: "#065f46",
    lineHeight: "1.5",
    margin: 0,
  },

  // Empty State
  emptyState: {
    backgroundColor: "#ffffff",
    border: "1px solid #f1f5f9",
    borderRadius: "20px",
    padding: "4rem",
    textAlign: "center",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "1rem",
  },
  emptyTitle: {
    fontSize: "1rem",
    fontWeight: "500",
    color: "#64748b",
  },
  emptyText: {
    fontSize: "0.875rem",
    color: "#94a3b8",
  },

  // Pagination
  pagination: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: "1rem",
    marginTop: "1rem",
  },
  paginationButton: {
    padding: "0.5rem 1rem",
    backgroundColor: "#ffffff",
    color: "#374151",
    border: "1px solid #e2e8f0",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "500",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  paginationButtonDisabled: {
    backgroundColor: "#f3f4f6",
    color: "#9ca3af",
    cursor: "not-allowed",
  },
  paginationText: {
    padding: "0 1rem",
    color: "#64748b",
    fontSize: "0.875rem",
  },
};

export default OwnerReviewsPage;
