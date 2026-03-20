import React, { useState, useEffect } from "react";
import {
  ArrowLeft,
  Calendar,
  IndianRupee,
  MapPin,
  User,
  Users,
  Clock,
  AlertCircle,
  CheckCircle,
  XCircle,
  Bell,
  HelpCircle,
  Search,
  ChevronDown,
  MoreVertical,
  TrendingUp,
  Car,
  Star,
  Wallet,
} from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";

// Helper: normalize status strings for consistent matching across the file
const normalize = (s) => (s || "").toString().toLowerCase();

function OwnerBookings() {
  const [bookings, setBookings] = useState([]);
  const [stats, setStats] = useState({});
  const [message, setMessage] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [spotFilter, setSpotFilter] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [dateFilter, setDateFilter] = useState(30); // days
  const [showDateDropdown, setShowDateDropdown] = useState(false);
  const [showItemsDropdown, setShowItemsDropdown] = useState(false);
  const [activeTab, setActiveTab] = useState("all");
  const token = localStorage.getItem("token");
  const location = useLocation();
  const navigate = useNavigate();

  // Fetch stats once on mount
  useEffect(() => {
    fetch("http://localhost:5000/api/system/stats", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then(setStats)
      .catch(() => setMessage("Failed to fetch stats"));
  }, [token]);

  // Sync spot filter from URL query param
  useEffect(() => {
    try {
      const params = new URLSearchParams(location.search);
      const s = params.get("spotId");
      setSpotFilter(s);
      setCurrentPage(1);
    } catch (err) {
      // ignore
    }
  }, [location.search]);

  // Fetch bookings from backend with optional search and spot filter
  const fetchBookings = (opts = {}) => {
    const params = new URLSearchParams();
    if (opts.search !== undefined ? opts.search : searchQuery)
      params.set(
        "search",
        opts.search !== undefined ? opts.search : searchQuery,
      );
    if (opts.spotId !== undefined ? opts.spotId : spotFilter)
      params.set(
        "spot_id",
        opts.spotId !== undefined ? opts.spotId : spotFilter,
      );
    if (dateFilter) params.set("days", dateFilter);

    const url = `http://localhost:5000/api/spots/owner/bookings?${params.toString()}`;
    fetch(url, { headers: { Authorization: `Bearer ${token}` } })
      .then((res) => res.json())
      .then((data) =>
        setBookings(Array.isArray(data) ? data : data.bookings || []),
      )
      .catch(() => setMessage("Failed to fetch bookings"));
  };

  // Initial load + when filters change
  useEffect(() => {
    fetchBookings();
  }, [dateFilter, activeTab, spotFilter]);

  // Debounced search
  useEffect(() => {
    const t = setTimeout(() => fetchBookings({ search: searchQuery }), 450);
    return () => clearTimeout(t);
  }, [searchQuery]);

  // Filter bookings based on date range
  const dateFilteredBookings = bookings.filter((booking) => {
    if (!dateFilter) return true;
    const bookingDate = new Date(booking.expires_at);
    const now = new Date();
    const daysAgo = new Date(now.setDate(now.getDate() - dateFilter));
    return bookingDate >= daysAgo;
  });

  // Normalized status groups
  const normalize = (s) => (s || "").toString().toLowerCase();
  const COMPLETED_STATUSES = ["completed", "expired", "finished"];
  const ONGOING_STATUSES = ["active", "ongoing", "in_progress"];
  const CANCELED_STATUSES = [
    "canceled",
    "cancelled",
    "canceled_by_user",
    "cancelled_by_user",
    "user_canceled",
    "cancelled_by_owner",
    "canceled_by_owner",
  ];

  // Filter bookings based on active tab (status)
  const statusFilteredBookings = dateFilteredBookings.filter((booking) => {
    const st = normalize(booking.status);
    if (activeTab === "all") return true;
    if (activeTab === "completed") return COMPLETED_STATUSES.includes(st);
    if (activeTab === "ongoing") return ONGOING_STATUSES.includes(st);
    if (activeTab === "canceled") return CANCELED_STATUSES.includes(st);
    return true;
  });

  // Filter bookings based on search query
  const filteredBookings = statusFilteredBookings.filter((booking) => {
    const searchLower = searchQuery.toLowerCase();
    return (
      booking.booking_id.toString().toLowerCase().includes(searchLower) ||
      booking.user_name.toLowerCase().includes(searchLower) ||
      booking.user_email.toLowerCase().includes(searchLower) ||
      booking.spot_name.toLowerCase().includes(searchLower)
    );
  });

  // Pagination logic
  const totalPages = Math.ceil(filteredBookings.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentBookings = filteredBookings.slice(startIndex, endIndex);

  // Get visible page numbers (max 4)
  const getVisiblePages = () => {
    if (totalPages <= 4) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    if (currentPage <= 2) {
      return [1, 2, 3, 4];
    }

    if (currentPage >= totalPages - 1) {
      return [totalPages - 3, totalPages - 2, totalPages - 1, totalPages];
    }

    return [currentPage - 1, currentPage, currentPage + 1, currentPage + 2];
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
  };

  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
    setCurrentPage(1); // Reset to first page when searching
  };

  const handleItemsPerPageChange = (value) => {
    setItemsPerPage(value);
    setCurrentPage(1);
    setShowItemsDropdown(false);
  };

  const handleDateFilterChange = (days) => {
    setDateFilter(days);
    setCurrentPage(1);
    setShowDateDropdown(false);
  };

  const getDateFilterLabel = () => {
    if (dateFilter === 7) return "Last 7 Days";
    if (dateFilter === 30) return "Last 30 Days";
    if (dateFilter === 90) return "Last 90 Days";
    if (dateFilter === null) return "All Time";
    return "Last 30 Days";
  };

  const getStatusIcon = (status) => {
    const st = normalize(status);
    switch (st) {
      case "active":
      case "ongoing":
        return <CheckCircle size={12} />;
      case "canceled":
      case "cancelled":
        return <XCircle size={12} />;
      case "expired":
        return <AlertCircle size={12} />;
      case "completed":
        return <CheckCircle size={12} />;
      default:
        return <AlertCircle size={12} />;
    }
  };

  const getInitials = (name) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const getAvatarColor = (index) => {
    const colors = [
      { bg: "#dbeafe", text: "#1e40af" },
      { bg: "#e0e7ff", text: "#4338ca" },
      { bg: "#ffe4e6", text: "#be123c" },
      { bg: "#d1fae5", text: "#047857" },
    ];
    return colors[index % colors.length];
  };

  // Compute stats locally from fetched bookings (respecting date filter)
  const computedStats = (() => {
    const arr = dateFilteredBookings || [];
    const total = arr.length;
    const active = arr.filter((b) =>
      ONGOING_STATUSES.includes(normalize(b.status)),
    ).length;
    const canceled = arr.filter(
      (b) =>
        CANCELED_STATUSES.includes(normalize(b.status)) ||
        normalize(b.status) === "expired",
    ).length;
    const revenue = arr.reduce((sum, b) => {
      const st = normalize(b.status);
      // include only completed bookings in revenue
      if (!COMPLETED_STATUSES.includes(st)) return sum;
      const price = Number(b.total_price ?? b.price ?? 0);
      return sum + (isNaN(price) ? 0 : price);
    }, 0);

    return { total, active, canceled, revenue };
  })();

  const formatCurrency = (val) => {
    const num = Number(val) || 0;
    // use Indian number formatting, show up to 2 decimal places
    return num.toLocaleString("en-IN", {
      minimumFractionDigits: num % 1 === 0 ? 0 : 2,
      maximumFractionDigits: 2,
    });
  };

  return (
    <div style={styles.container}>
      {/* Sidebar */}
      <aside style={styles.sidebar}>
        <div style={styles.sidebarContent}>
          <div style={styles.logoSection}>
            <div style={styles.logo}>
              <MapPin size={24} color="#ffffff" />
            </div>
            <div>
              <h1 style={styles.logoTitle}>NExtSPot</h1>
              <p style={styles.logoSubtitle}>Owner Dashboard</p>
            </div>
          </div>

          <nav style={styles.nav}>
            <a href="/owner/bookings" style={styles.navItemActive}>
              <Calendar size={20} />
              <span>Bookings</span>
            </a>
            <a href="/owner/dashboard" style={styles.navItem}>
              <MapPin size={20} />
              <span>My Spots</span>
            </a>
            <a href="/owner/reviews" style={styles.navItem}>
              <Star size={20} />
              <span>Reviews</span>
            </a>
            <div style={styles.navDivider}></div>
          </nav>
        </div>
      </aside>

      {/* Main Content */}
      <main style={styles.main}>
        {/* Top Header */}
        <header style={styles.header}>
          <div style={styles.headerLeft}>
            <h2 style={styles.headerTitle}>Bookings History</h2>
            <div style={styles.searchBox}>
              <Search size={18} style={styles.searchIcon} />
              <input
                type="text"
                placeholder="Search by ID or User..."
                style={styles.searchInput}
                value={searchQuery}
                onChange={handleSearchChange}
              />
            </div>
          </div>
        </header>

        <div style={styles.content}>
          {/* Message Display */}
          {message && (
            <div style={styles.messageBox}>
              <AlertCircle size={20} />
              <span>{message}</span>
            </div>
          )}

          {/* Page Title */}
          <div style={styles.pageHeader}>
            <h1 style={styles.pageTitle}>Bookings Log</h1>
            <p style={styles.pageSubtitle}>
              Manage and track all parking reservations across your locations.
            </p>
          </div>

          {/* Stats from API */}
          {(stats && Object.keys(stats).length > 0) ||
          dateFilteredBookings.length > 0 ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(4,1fr)",
                gap: "12px",
                marginBottom: "16px",
              }}
            >
              <div style={styles.summaryCard}>
                <p style={styles.summaryLabel}>Total Bookings</p>
                <div style={styles.summaryValue}>
                  {stats.total_bookings ??
                    stats.totalBookings ??
                    computedStats.total}
                </div>
                <p style={styles.summarySub}>Last {dateFilter || "All"} days</p>
              </div>
              <div style={styles.summaryCard}>
                <p style={styles.summaryLabel}>Active Bookings</p>
                <div style={styles.summaryValue}>
                  {stats.active_bookings ??
                    stats.activeBookings ??
                    computedStats.active}
                </div>
                <p style={styles.summarySub}>Ongoing reservations</p>
              </div>
              <div style={styles.summaryCard}>
                <p style={styles.summaryLabel}>Canceled</p>
                <div style={styles.summaryValue}>
                  {stats.canceled_bookings ??
                    stats.cancelled_bookings ??
                    computedStats.canceled}
                </div>
                <p style={styles.summarySub}>Canceled / expired</p>
              </div>
              <div style={styles.summaryCard}>
                <p style={styles.summaryLabel}>Revenue</p>
                <div style={styles.summaryValue}>
                  ₹
                  {formatCurrency(
                    stats.total_revenue ??
                      stats.revenue ??
                      computedStats.revenue,
                  )}
                </div>
                <p style={styles.summarySub}>Gross</p>
              </div>
            </div>
          ) : null}

          {/* Bookings Table Section */}
          <div style={styles.tableCard}>
            {/* Tabs and Filters */}
            <div style={styles.tableHeader}>
              <div style={styles.tabs}>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setActiveTab("all");
                    setCurrentPage(1);
                  }}
                  style={activeTab === "all" ? styles.tabActive : styles.tab}
                >
                  All Bookings
                </a>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setActiveTab("completed");
                    setCurrentPage(1);
                  }}
                  style={
                    activeTab === "completed" ? styles.tabActive : styles.tab
                  }
                >
                  Completed
                </a>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setActiveTab("ongoing");
                    setCurrentPage(1);
                  }}
                  style={
                    activeTab === "ongoing" ? styles.tabActive : styles.tab
                  }
                >
                  Ongoing
                </a>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setActiveTab("canceled");
                    setCurrentPage(1);
                  }}
                  style={
                    activeTab === "canceled" ? styles.tabActive : styles.tab
                  }
                >
                  Canceled
                </a>
              </div>
              <div style={styles.filters}>
                {/* Date Filter Dropdown */}
                <div style={styles.dropdownContainer}>
                  <button
                    style={styles.filterButton}
                    onClick={() => {
                      setShowDateDropdown(!showDateDropdown);
                      setShowItemsDropdown(false);
                    }}
                  >
                    <Calendar size={16} />
                    <span>{getDateFilterLabel()}</span>
                    <ChevronDown size={14} />
                  </button>
                  {showDateDropdown && (
                    <div style={styles.dropdown}>
                      <div
                        style={styles.dropdownItem}
                        onClick={() => handleDateFilterChange(7)}
                      >
                        Last 7 Days
                      </div>
                      <div
                        style={styles.dropdownItem}
                        onClick={() => handleDateFilterChange(30)}
                      >
                        Last 30 Days
                      </div>
                      <div
                        style={styles.dropdownItem}
                        onClick={() => handleDateFilterChange(90)}
                      >
                        Last 90 Days
                      </div>
                      <div
                        style={styles.dropdownItem}
                        onClick={() => handleDateFilterChange(null)}
                      >
                        All Time
                      </div>
                    </div>
                  )}
                </div>

                {/* Items Per Page Dropdown */}
                <div style={styles.dropdownContainer}>
                  <button
                    style={styles.filterButton}
                    onClick={() => {
                      setShowItemsDropdown(!showItemsDropdown);
                      setShowDateDropdown(false);
                    }}
                  >
                    <span>{itemsPerPage} per page</span>
                    <ChevronDown size={14} />
                  </button>
                  {showItemsDropdown && (
                    <div style={styles.dropdown}>
                      <div
                        style={styles.dropdownItem}
                        onClick={() => handleItemsPerPageChange(10)}
                      >
                        10 per page
                      </div>
                      <div
                        style={styles.dropdownItem}
                        onClick={() => handleItemsPerPageChange(25)}
                      >
                        25 per page
                      </div>
                      <div
                        style={styles.dropdownItem}
                        onClick={() => handleItemsPerPageChange(50)}
                      >
                        50 per page
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Table */}
            {filteredBookings.length === 0 ? (
              <div style={styles.emptyState}>
                <Calendar size={48} color="#9e9e9e" />
                <h3 style={styles.emptyTitle}>
                  {searchQuery ? "No bookings found" : "No bookings yet"}
                </h3>
                <p style={styles.emptyText}>
                  {searchQuery
                    ? "Try adjusting your search query"
                    : "Your booking history will appear here once customers start reserving your spots."}
                </p>
              </div>
            ) : (
              <>
                <div style={styles.tableWrapper}>
                  <table style={styles.table}>
                    <thead style={styles.thead}>
                      <tr>
                        <th style={styles.th}>BOOKING ID</th>
                        <th style={styles.th}>SPOT ID</th>
                        <th style={styles.th}>USER NAME</th>
                        <th style={styles.th}>VEHICLE</th>
                        <th style={styles.th}>START TIME</th>
                        <th style={styles.th}>STATUS</th>
                        <th style={styles.th}>REVENUE</th>
                        <th style={styles.th}></th>
                      </tr>
                    </thead>
                    <tbody>
                      {currentBookings.map((booking, index) => (
                        <tr key={booking.booking_id} style={styles.tr}>
                          <td style={styles.td}>
                            <span style={styles.bookingId}>
                              #{booking.booking_id}
                            </span>
                          </td>
                          <td style={styles.td}>
                            <span
                              onClick={() =>
                                navigate(
                                  `/owner/bookings?spotId=${booking.spot_id}`,
                                )
                              }
                              style={{ ...styles.spotBadge, cursor: "pointer" }}
                              title="Filter bookings for this spot"
                            >
                              {booking.slot_number || "N/A"}
                            </span>
                          </td>
                          <td style={styles.td}>
                            <div style={styles.userCell}>
                              <div
                                style={{
                                  ...styles.avatar,
                                  backgroundColor: getAvatarColor(index).bg,
                                  color: getAvatarColor(index).text,
                                }}
                              >
                                {getInitials(booking.user_name)}
                              </div>
                              <span style={styles.userName}>
                                {booking.user_name}
                              </span>
                            </div>
                          </td>
                          <td style={styles.td}>
                            <span style={styles.vehicleText}>
                              {booking.spot_name}
                            </span>
                          </td>
                          <td style={styles.td}>
                            <div style={styles.timeCell}>
                              <p style={styles.timeMain}>
                                {new Date(booking.expires_at).toLocaleString(
                                  "en-US",
                                  {
                                    month: "short",
                                    day: "numeric",
                                    hour: "2-digit",
                                    minute: "2-digit",
                                  },
                                )}
                              </p>
                              <p style={styles.timeSub}>
                                {booking.status === "active"
                                  ? "Now Ongoing"
                                  : "Expires"}
                              </p>
                            </div>
                          </td>
                          <td style={styles.td}>
                            <span
                              style={{
                                ...styles.statusBadge,
                                ...getStatusStyleNew(booking.status),
                              }}
                            >
                              <span style={styles.statusDot}></span>
                              {booking.status}
                            </span>
                          </td>
                          <td style={styles.td}>
                            <span style={styles.price}>
                              ₹{booking.total_price}
                            </span>
                          </td>
                          <td style={styles.td}>
                            <button style={styles.moreButton}>
                              <MoreVertical size={18} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div style={styles.pagination}>
                  <p style={styles.paginationText}>
                    Showing{" "}
                    <span style={styles.paginationBold}>
                      {startIndex + 1} to{" "}
                      {Math.min(endIndex, filteredBookings.length)}
                    </span>{" "}
                    of {filteredBookings.length} bookings
                  </p>
                  <div style={styles.paginationButtons}>
                    <button
                      style={
                        currentPage === 1
                          ? styles.paginationButtonDisabled
                          : styles.paginationButton
                      }
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                    >
                      Previous
                    </button>
                    {getVisiblePages().map((page) => (
                      <button
                        key={page}
                        style={
                          currentPage === page
                            ? styles.paginationButtonActive
                            : styles.paginationButton
                        }
                        onClick={() => handlePageChange(page)}
                      >
                        {page}
                      </button>
                    ))}
                    <button
                      style={
                        currentPage === totalPages
                          ? styles.paginationButtonDisabled
                          : styles.paginationButton
                      }
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

const getStatusStyleNew = (status) => {
  const st = normalize(status);
  switch (st) {
    case "active":
    case "ongoing":
      return {
        backgroundColor: "#e0e9fe",
        color: "#336ee6",
      };
    case "canceled":
    case "cancelled":
      return {
        backgroundColor: "#ffebee",
        color: "#c62828",
      };
    case "expired":
      // expired should be shown as neutral/grey
      return {
        backgroundColor: "#f3f4f6",
        color: "#6b7280",
      };
    case "completed":
      return {
        backgroundColor: "#e2ece6",
        color: "#3d664d",
      };
    case "upcoming":
      return {
        backgroundColor: "#fef3c7",
        color: "#92400e",
      };
    default:
      return {
        backgroundColor: "#f3f4f6",
        color: "#6b7280",
      };
  }
};

const styles = {
  container: {
    display: "flex",
    minHeight: "100vh",
    backgroundColor: "#f6f6f8",
    fontFamily:
      'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },

  // Sidebar
  sidebar: {
    width: "256px",
    backgroundColor: "#ffffff",
    borderRight: "1px solid #e5e7eb",
    display: "flex",
    flexDirection: "column",
    position: "sticky",
    top: 0,
    height: "100vh",
  },
  sidebarContent: {
    padding: "1.5rem",
    flex: 1,
  },
  logoSection: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    marginBottom: "2rem",
  },
  logo: {
    width: "40px",
    height: "40px",
    backgroundColor: "#336ee6",
    borderRadius: "12px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: "0 4px 12px rgba(51, 110, 230, 0.2)",
  },
  logoTitle: {
    fontSize: "1rem",
    fontWeight: "700",
    color: "#1f2937",
    margin: 0,
    lineHeight: 1,
  },
  logoSubtitle: {
    fontSize: "0.75rem",
    color: "#6b7280",
    margin: "4px 0 0 0",
  },
  nav: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  navItem: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    padding: "0.625rem 0.75rem",
    color: "#6b7280",
    textDecoration: "none",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "500",
    transition: "all 0.2s",
    cursor: "pointer",
  },
  navItemActive: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    padding: "0.625rem 0.75rem",
    backgroundColor: "rgba(51, 110, 230, 0.1)",
    color: "#336ee6",
    textDecoration: "none",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "600",
    transition: "all 0.2s",
    cursor: "pointer",
  },
  navIcon: {
    fontSize: "1.25rem",
  },
  navDivider: {
    height: "1px",
    backgroundColor: "#e5e7eb",
    margin: "1rem 0",
  },
  userProfile: {
    padding: "1.5rem",
    borderTop: "1px solid #e5e7eb",
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
  },
  userAvatar: {
    width: "40px",
    height: "40px",
    borderRadius: "50%",
    backgroundColor: "#e5e7eb",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "600",
    fontSize: "0.875rem",
    color: "#6b7280",
  },
  userInfo: {
    flex: 1,
    minWidth: 0,
  },
  userName: {
    fontSize: "0.875rem",
    fontWeight: "600",
    color: "#1f2937",
    margin: 0,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  userRole: {
    fontSize: "0.75rem",
    color: "#6b7280",
    margin: "2px 0 0 0",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },

  // Main
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#f6f6f8",
    overflow: "auto",
  },

  // Header
  header: {
    position: "sticky",
    top: 0,
    zIndex: 10,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "1rem 2rem",
    backgroundColor: "rgba(255, 255, 255, 0.8)",
    backdropFilter: "blur(12px)",
    borderBottom: "1px solid #e5e7eb",
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "1.5rem",
  },
  headerTitle: {
    fontSize: "1.25rem",
    fontWeight: "700",
    color: "#1f2937",
    margin: 0,
  },
  searchBox: {
    position: "relative",
    maxWidth: "300px",
  },
  searchIcon: {
    position: "absolute",
    left: "12px",
    top: "50%",
    transform: "translateY(-50%)",
    color: "#9ca3af",
    pointerEvents: "none",
  },
  searchInput: {
    width: "256px",
    paddingLeft: "40px",
    paddingRight: "16px",
    paddingTop: "8px",
    paddingBottom: "8px",
    backgroundColor: "#f3f4f6",
    border: "none",
    borderRadius: "8px",
    fontSize: "0.875rem",
    outline: "none",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
  },
  iconButton: {
    padding: "0.5rem",
    backgroundColor: "#f3f4f6",
    border: "none",
    borderRadius: "8px",
    color: "#6b7280",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.2s",
  },

  // Content
  content: {
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "1.5rem 2rem",
    width: "100%",
  },
  messageBox: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    padding: "1rem",
    marginBottom: "1.5rem",
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "8px",
    color: "#b91c1c",
  },
  pageHeader: {
    marginBottom: "2rem",
  },
  pageTitle: {
    fontSize: "1.875rem",
    fontWeight: "700",
    color: "#1f2937",
    margin: "0 0 0.5rem 0",
  },
  pageSubtitle: {
    fontSize: "0.875rem",
    color: "#6b7280",
    margin: 0,
  },

  // Table Card
  tableCard: {
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    border: "1px solid #e5e7eb",
    overflow: "hidden",
    marginBottom: "1.5rem",
    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
  },
  tableHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 1.5rem",
    borderBottom: "1px solid #f3f4f6",
  },
  tabs: {
    display: "flex",
    gap: "2rem",
  },
  tab: {
    padding: "1rem 0",
    borderBottom: "2px solid transparent",
    color: "#6b7280",
    textDecoration: "none",
    fontSize: "0.875rem",
    fontWeight: "500",
    transition: "all 0.2s",
  },
  tabActive: {
    padding: "1rem 0",
    borderBottom: "2px solid #336ee6",
    color: "#336ee6",
    textDecoration: "none",
    fontSize: "0.875rem",
    fontWeight: "700",
    transition: "all 0.2s",
  },
  filters: {
    display: "flex",
    alignItems: "center",
    gap: "1rem",
    padding: "0.5rem 0",
  },
  dropdownContainer: {
    position: "relative",
  },
  filterButton: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
    padding: "0.375rem 0.75rem",
    backgroundColor: "#f9fafb",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    fontSize: "0.75rem",
    fontWeight: "500",
    color: "#374151",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  dropdown: {
    position: "absolute",
    top: "calc(100% + 4px)",
    right: 0,
    backgroundColor: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
    minWidth: "150px",
    zIndex: 1000,
  },
  dropdownItem: {
    padding: "0.75rem 1rem",
    fontSize: "0.875rem",
    color: "#374151",
    cursor: "pointer",
    transition: "background-color 0.2s",
    borderBottom: "1px solid #f3f4f6",
  },

  // Table
  tableWrapper: {
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
  },
  thead: {
    backgroundColor: "rgba(248, 250, 252, 0.5)",
  },
  th: {
    padding: "1rem 1.5rem",
    textAlign: "left",
    fontSize: "0.75rem",
    fontWeight: "700",
    color: "#6b7280",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  tr: {
    borderBottom: "1px solid #f3f4f6",
    transition: "background-color 0.2s",
    cursor: "pointer",
  },
  td: {
    padding: "1.25rem 1.5rem",
  },
  bookingId: {
    fontFamily: "monospace",
    fontSize: "0.875rem",
    color: "#6b7280",
  },
  spotBadge: {
    display: "inline-block",
    padding: "0.25rem 0.5rem",
    backgroundColor: "#f3f4f6",
    border: "1px solid #e5e7eb",
    borderRadius: "6px",
    fontSize: "0.75rem",
    fontWeight: "600",
    color: "#374151",
  },
  userCell: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
  },
  avatar: {
    width: "32px",
    height: "32px",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "0.625rem",
    fontWeight: "700",
    flexShrink: 0,
  },
  vehicleText: {
    fontSize: "0.875rem",
    color: "#6b7280",
  },
  timeCell: {
    display: "flex",
    flexDirection: "column",
    gap: "2px",
  },
  timeMain: {
    fontSize: "0.875rem",
    fontWeight: "500",
    color: "#1f2937",
    margin: 0,
  },
  timeSub: {
    fontSize: "0.75rem",
    color: "#9ca3af",
    margin: 0,
  },
  statusBadge: {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.375rem",
    padding: "0.25rem 0.625rem",
    borderRadius: "9999px",
    fontSize: "0.75rem",
    fontWeight: "700",
    textTransform: "capitalize",
  },
  statusDot: {
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    backgroundColor: "currentColor",
    opacity: 0.4,
  },
  price: {
    fontSize: "0.875rem",
    fontWeight: "700",
    color: "#1f2937",
    textAlign: "right",
    display: "block",
  },
  moreButton: {
    padding: "0.25rem",
    backgroundColor: "transparent",
    border: "none",
    borderRadius: "4px",
    color: "#9ca3af",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 0.2s",
  },

  // Pagination
  pagination: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "1rem 1.5rem",
    backgroundColor: "rgba(248, 250, 252, 0.5)",
    borderTop: "1px solid #f3f4f6",
  },
  paginationText: {
    fontSize: "0.75rem",
    color: "#6b7280",
    margin: 0,
  },
  paginationBold: {
    fontWeight: "600",
    color: "#374151",
  },
  paginationButtons: {
    display: "flex",
    alignItems: "center",
    gap: "0.5rem",
  },
  paginationButton: {
    padding: "0.375rem 0.75rem",
    backgroundColor: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    fontSize: "0.75rem",
    fontWeight: "500",
    color: "#374151",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  paginationButtonActive: {
    padding: "0.375rem 0.75rem",
    backgroundColor: "#336ee6",
    border: "1px solid #336ee6",
    borderRadius: "8px",
    fontSize: "0.75rem",
    fontWeight: "700",
    color: "#ffffff",
    cursor: "pointer",
  },
  paginationButtonDisabled: {
    padding: "0.375rem 0.75rem",
    backgroundColor: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
    fontSize: "0.75rem",
    fontWeight: "500",
    color: "#9ca3af",
    cursor: "not-allowed",
  },

  // Empty State
  emptyState: {
    textAlign: "center",
    padding: "3rem",
    color: "#6b7280",
  },
  emptyTitle: {
    fontSize: "1.125rem",
    fontWeight: "500",
    color: "#1f2937",
    margin: "1rem 0 0.5rem 0",
  },
  emptyText: {
    margin: 0,
    fontSize: "0.875rem",
  },

  // Summary Cards
  summaryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
    gap: "1.5rem",
  },
  summaryCard: {
    padding: "1rem",
    backgroundColor: "#ffffff",
    border: "1px solid #e5e7eb",
    borderRadius: "12px",
  },
  summaryHeader: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    marginBottom: "0.5rem",
  },
  summaryIcon: {
    padding: "0.5rem",
    borderRadius: "8px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  summaryLabel: {
    fontSize: "0.875rem",
    fontWeight: "600",
    color: "#6b7280",
    margin: 0,
  },
  summaryValue: {
    fontSize: "1.5rem",
    fontWeight: "700",
    color: "#1f2937",
    margin: "0 0 0.25rem 0",
  },
  summaryTrend: {
    fontSize: "0.75rem",
    color: "#10b981",
    margin: 0,
    display: "flex",
    alignItems: "center",
    gap: "0.25rem",
  },
  summarySub: {
    fontSize: "0.75rem",
    color: "#9ca3af",
    margin: 0,
  },
};

export default OwnerBookings;
