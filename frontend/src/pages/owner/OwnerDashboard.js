import React, { useState, useEffect, useRef } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMapEvents,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import {
  Plus,
  MapPin,
  IndianRupee,
  Trash2,
  AlertTriangle,
  LogOut,
  ChevronDown,
  ChevronUp,
  Search,
  Calendar,
  Users,
} from "lucide-react";

// Map click handler component
function MapClickHandler({ onMapClick, isAddingMode, onMapBackgroundClick }) {
  useMapEvents({
    click: (e) => {
      if (isAddingMode) {
        onMapClick(e.latlng);
      } else {
        // If not in adding mode and clicking the map background, deselect
        onMapBackgroundClick();
      }
    },
  });
  return null;
}

// Component to handle map centering when a spot is selected
function MapViewController({ center, zoom }) {
  const map = useMap();

  useEffect(() => {
    if (center) {
      map.flyTo(center, zoom || 16, {
        duration: 1.5,
        easeLinearity: 0.5,
      });
    }
  }, [center, zoom, map]);

  return null;
}

function EnhancedOwnerDashboard() {
  const [spots, setSpots] = useState([]);
  const [newSpot, setNewSpot] = useState({
    name: "",
    latitude: "",
    longitude: "",
    price: "",
    type: "",
    initial_slots: 1,
  });
  const [message, setMessage] = useState("");
  const [isAddingMode, setIsAddingMode] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [spotToDelete, setSpotToDelete] = useState(null);
  const [selectedSpotId, setSelectedSpotId] = useState(null);
  const [isAddFormExpanded, setIsAddFormExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [mapCenter, setMapCenter] = useState(null);
  const [mapZoom, setMapZoom] = useState(15);
  const [expandedSpotId, setExpandedSpotId] = useState(null);
  const markerRefs = useRef({});
  const spotCardRefs = useRef({});
  const token = localStorage.getItem("token");

  // Logout function
  const handleLogout = () => {
    localStorage.removeItem("token");
    window.location.href = "/home";
  };

  // Get user's current location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation([
            position.coords.latitude,
            position.coords.longitude,
          ]);
        },
        (error) => {
          console.log("Location access denied");
        },
      );
    }
  }, []);

  // Fetch owner's spots
  const fetchSpots = () => {
    fetch("http://localhost:5000/api/spots/owner", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        const uniqueSpots = Array.from(
          new Map(data.map((item) => [item.id, item])).values(),
        );
        setSpots(uniqueSpots);
        setLastRefresh(new Date());
      })
      .catch((err) => setMessage(err.message));
  };

  // Auto-refresh every 2 seconds
  useEffect(() => {
    fetchSpots();
    const interval = setInterval(fetchSpots, 2000);
    return () => clearInterval(interval);
  }, []);

  // Handle map click for adding new spot
  const handleMapClick = (latlng) => {
    setNewSpot({
      ...newSpot,
      latitude: latlng.lat.toFixed(6),
      longitude: latlng.lng.toFixed(6),
    });
    setIsAddingMode(false);
    setIsAddFormExpanded(true);
    setMessage("Location selected! Fill in the remaining details.");
  };

  // Add new spot
  const addSpot = () => {
    if (
      !newSpot.name ||
      !newSpot.latitude ||
      !newSpot.longitude ||
      !newSpot.price ||
      !newSpot.type
    ) {
      setMessage("Please fill in all fields");
      return;
    }

    const initialSlots = parseInt(newSpot.initial_slots) || 1;
    if (initialSlots < 1 || initialSlots > 100) {
      setMessage("Initial slots must be between 1 and 100");
      return;
    }

    const payload = {
      ...newSpot,
      latitude: parseFloat(newSpot.latitude),
      longitude: parseFloat(newSpot.longitude),
      price: parseFloat(newSpot.price),
      initial_slots: initialSlots,
    };

    fetch("http://localhost:5000/api/spots/add", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(payload),
    })
      .then((res) => res.json())
      .then((data) => {
        setMessage(data.message || "Spot added successfully!");
        fetchSpots();
        setNewSpot({
          name: "",
          latitude: "",
          longitude: "",
          price: "",
          type: "",
          initial_slots: 1,
        });
        setIsAddFormExpanded(false);
      })
      .catch(() => setMessage("Failed to add spot"));
  };

  // Remove spot
  const removeSpot = (id) => {
    setSpotToDelete(id);
    setShowDeleteDialog(true);
  };

  const confirmDelete = () => {
    if (spotToDelete) {
      fetch(`http://localhost:5000/api/spots/${spotToDelete}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((res) => res.json())
        .then((data) => {
          setMessage(data.message);
          fetchSpots();
          if (selectedSpotId === spotToDelete) {
            setSelectedSpotId(null);
          }
        })
        .catch(() => setMessage("Failed to remove spot"));
    }
    setShowDeleteDialog(false);
    setSpotToDelete(null);
  };

  const cancelDelete = () => {
    setShowDeleteDialog(false);
    setSpotToDelete(null);
  };

  // Handle spot selection from list
  const handleSpotSelect = (spot) => {
    if (selectedSpotId === spot.id) {
      setSelectedSpotId(null);
      setMapCenter(null);
      setMapZoom(15);
      if (markerRefs.current[spot.id]) {
        markerRefs.current[spot.id].closePopup();
      }
    } else {
      Object.keys(markerRefs.current).forEach((id) => {
        if (markerRefs.current[id] && parseInt(id) !== spot.id) {
          markerRefs.current[id].closePopup();
        }
      });

      setSelectedSpotId(spot.id);
      setMapCenter([spot.latitude, spot.longitude]);
      setMapZoom(17);

      // Scroll the selected spot card into view
      setTimeout(() => {
        if (spotCardRefs.current[spot.id]) {
          spotCardRefs.current[spot.id].scrollIntoView({
            behavior: "smooth",
            block: "nearest",
            inline: "nearest",
          });
        }

        if (markerRefs.current[spot.id]) {
          markerRefs.current[spot.id].openPopup();
        }
      }, 100);
    }
  };

  const handleMapBackgroundClick = () => {
    if (!isAddingMode && selectedSpotId !== null) {
      if (markerRefs.current[selectedSpotId]) {
        markerRefs.current[selectedSpotId].closePopup();
      }
      setSelectedSpotId(null);
      setMapCenter(null);
    }
  };

  // Get occupancy level
  const getOccupancyLevel = (spot) => {
    const occupiedSlots = Math.max(0, spot.occupied_slots || 0); // Prevent negative values
    const percentage = (occupiedSlots / spot.total_slots) * 100;
    if (percentage < 50) return { level: "Available", color: "#10B981" };
    if (percentage < 80) return { level: "Busy", color: "#3B82F6" };
    return { level: "High Occupancy", color: "#8B5CF6" };
  };

  const getOccupancyPercentage = (spot) => {
    const occupiedSlots = Math.max(0, spot.occupied_slots || 0); // Prevent negative values
    return Math.round((occupiedSlots / spot.total_slots) * 100);
  };

  const getSafeOccupiedSlots = (spot) => {
    return Math.max(0, spot.occupied_slots || 0); // Always return non-negative value
  };

  const getMarkerColor = (spot) => {
    const percentage = getOccupancyPercentage(spot);
    if (percentage < 50) return { bg: "#10B981", border: "#059669" };
    if (percentage < 80) return { bg: "#3B82F6", border: "#2563EB" };
    return { bg: "#8B5CF6", border: "#7C3AED" };
  };

  // Create custom map icons
  const createCustomIcon = (spot, isSelected) => {
    const colors = getMarkerColor(spot);
    const scale = isSelected ? 1.3 : 1;

    return L.divIcon({
      html: `
        <div style="
          width: ${24 * scale}px;
          height: ${24 * scale}px;
          background-color: ${colors.bg};
          border: ${3 * scale}px solid ${colors.border};
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: ${isSelected ? "0 0 0 4px rgba(139, 92, 246, 0.3)," : ""} 0 2px 4px rgba(0,0,0,0.2);
          position: relative;
          transition: all 0.3s ease;
        ">
          <div style="
            width: ${8 * scale}px;
            height: ${8 * scale}px;
            background-color: #FFFFFF;
            border-radius: 50%;
          "></div>
        </div>
      `,
      className: "custom-spot-icon",
      iconSize: [24 * scale, 24 * scale],
      iconAnchor: [12 * scale, 12 * scale],
      popupAnchor: [0, -12 * scale],
    });
  };

  // Filter spots based on search
  const filteredSpots = spots.filter((spot) =>
    spot.name.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#F8F9FA",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      }}
    >
      {/* Header */}
      <div
        style={{
          background: "#FFFFFF",
          borderBottom: "1px solid #E5E7EB",
          padding: "16px 32px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: 8,
              background: "#336ee6",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <MapPin size={24} color="#ffffff" />
          </div>
          <div>
            <h1
              style={{
                margin: 0,
                fontSize: "20px",
                fontWeight: "600",
                color: "#111827",
              }}
            >
              NExtSPot
            </h1>
            <p style={{ margin: 0, fontSize: "12px", color: "#6B7280" }}>
              Map Overview
            </p>
          </div>
        </div>
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <button
            onClick={() => (window.location.href = "/owner/bookings")}
            style={{
              background: "#FFFFFF",
              color: "#374151",
              border: "1px solid #D1D5DB",
              borderRadius: "8px",
              padding: "8px 16px",
              fontSize: "14px",
              fontWeight: "500",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              transition: "all 0.2s ease",
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "#F9FAFB";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "#FFFFFF";
            }}
          >
            <Calendar size={14} />
            Bookings
          </button>
          <button
            onClick={() => (window.location.href = "/owner/reviews")}
            style={{
              background: "#FFFFFF",
              color: "#374151",
              border: "1px solid #D1D5DB",
              borderRadius: "8px",
              padding: "8px 16px",
              fontSize: "14px",
              fontWeight: "500",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              transition: "all 0.2s ease",
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "#F9FAFB";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "#FFFFFF";
            }}
          >
            <Users size={14} />
            Reviews
          </button>
          <button
            onClick={handleLogout}
            style={{
              background: "#FFFFFF",
              color: "#EF4444",
              border: "1px solid #FCA5A5",
              borderRadius: "8px",
              padding: "8px 16px",
              fontSize: "14px",
              fontWeight: "500",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "6px",
              transition: "all 0.2s ease",
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = "#FEF2F2";
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = "#FFFFFF";
            }}
          >
            <LogOut size={14} />
            Logout
          </button>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          height: "calc(100vh - 88px)",
        }}
      >
        {/* Left Sidebar */}
        <div
          style={{
            width: "380px",
            background: "#FFFFFF",
            borderRight: "1px solid #E5E7EB",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          {/* Add New Spot Collapsible Section */}
          <div
            style={{
              borderBottom: "1px solid #E5E7EB",
            }}
          >
            <button
              onClick={() => setIsAddFormExpanded(!isAddFormExpanded)}
              style={{
                width: "100%",
                padding: "16px 20px",
                background: isAddFormExpanded
                  ? "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)"
                  : "#F8F9FA",
                color: isAddFormExpanded ? "#FFFFFF" : "#374151",
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                fontSize: "14px",
                fontWeight: "600",
                transition: "all 0.3s ease",
              }}
            >
              <div
                style={{ display: "flex", alignItems: "center", gap: "8px" }}
              >
                <Plus size={16} />
                Add New Parking Spot
              </div>
              {isAddFormExpanded ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </button>

            {/* Collapsible Add Form */}
            <div
              style={{
                maxHeight: isAddFormExpanded ? "800px" : "0",
                overflow: "hidden",
                transition: "max-height 0.3s ease",
              }}
            >
              <div style={{ padding: "20px" }}>
                {message && (
                  <div
                    style={{
                      background:
                        message.toLowerCase().includes("cannot") ||
                        message.toLowerCase().includes("failed")
                          ? "#FEF2F2"
                          : "#F0FDF4",
                      color:
                        message.toLowerCase().includes("cannot") ||
                        message.toLowerCase().includes("failed")
                          ? "#DC2626"
                          : "#16A34A",
                      padding: "10px 12px",
                      borderRadius: "6px",
                      fontSize: "13px",
                      fontWeight: "500",
                      marginBottom: "16px",
                      border: `1px solid ${
                        message.toLowerCase().includes("cannot") ||
                        message.toLowerCase().includes("failed")
                          ? "#FECACA"
                          : "#BBF7D0"
                      }`,
                    }}
                  >
                    {message}
                  </div>
                )}

                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "14px",
                  }}
                >
                  <div>
                    <label
                      style={{
                        display: "block",
                        fontSize: "13px",
                        fontWeight: "500",
                        color: "#374151",
                        marginBottom: "6px",
                      }}
                    >
                      Spot Name
                    </label>
                    <input
                      placeholder="Enter spot name"
                      value={newSpot.name}
                      onChange={(e) =>
                        setNewSpot({ ...newSpot, name: e.target.value })
                      }
                      style={{
                        width: "100%",
                        padding: "9px 12px",
                        border: "1px solid #D1D5DB",
                        borderRadius: "6px",
                        fontSize: "13px",
                        transition: "border-color 0.2s ease",
                        outline: "none",
                        background: "#FFFFFF",
                        boxSizing: "border-box",
                      }}
                      onFocus={(e) => (e.target.style.borderColor = "#6366F1")}
                      onBlur={(e) => (e.target.style.borderColor = "#D1D5DB")}
                    />
                  </div>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "10px",
                    }}
                  >
                    <div>
                      <label
                        style={{
                          display: "block",
                          fontSize: "13px",
                          fontWeight: "500",
                          color: "#374151",
                          marginBottom: "6px",
                        }}
                      >
                        Latitude
                      </label>
                      <input
                        placeholder="0.000000"
                        value={newSpot.latitude}
                        onChange={(e) =>
                          setNewSpot({ ...newSpot, latitude: e.target.value })
                        }
                        style={{
                          width: "100%",
                          padding: "9px 12px",
                          border: "1px solid #D1D5DB",
                          borderRadius: "6px",
                          fontSize: "13px",
                          transition: "border-color 0.2s ease",
                          outline: "none",
                          background: "#FFFFFF",
                          boxSizing: "border-box",
                        }}
                        onFocus={(e) =>
                          (e.target.style.borderColor = "#6366F1")
                        }
                        onBlur={(e) => (e.target.style.borderColor = "#D1D5DB")}
                      />
                    </div>
                    <div>
                      <label
                        style={{
                          display: "block",
                          fontSize: "13px",
                          fontWeight: "500",
                          color: "#374151",
                          marginBottom: "6px",
                        }}
                      >
                        Longitude
                      </label>
                      <input
                        placeholder="0.000000"
                        value={newSpot.longitude}
                        onChange={(e) =>
                          setNewSpot({ ...newSpot, longitude: e.target.value })
                        }
                        style={{
                          width: "100%",
                          padding: "9px 12px",
                          border: "1px solid #D1D5DB",
                          borderRadius: "6px",
                          fontSize: "13px",
                          transition: "border-color 0.2s ease",
                          outline: "none",
                          background: "#FFFFFF",
                          boxSizing: "border-box",
                        }}
                        onFocus={(e) =>
                          (e.target.style.borderColor = "#6366F1")
                        }
                        onBlur={(e) => (e.target.style.borderColor = "#D1D5DB")}
                      />
                    </div>
                  </div>

                  <button
                    onClick={() => setIsAddingMode(!isAddingMode)}
                    style={{
                      padding: "9px 14px",
                      background: isAddingMode ? "#6366F1" : "#F8FAFC",
                      color: isAddingMode ? "#FFFFFF" : "#374151",
                      border: `1px solid ${isAddingMode ? "#6366F1" : "#D1D5DB"}`,
                      borderRadius: "6px",
                      fontSize: "13px",
                      fontWeight: "500",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "6px",
                      transition: "all 0.2s ease",
                      width: "100%",
                    }}
                  >
                    <MapPin size={14} />
                    {isAddingMode ? "Cancel Selection" : "Select on Map"}
                  </button>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "10px",
                    }}
                  >
                    <div>
                      <label
                        style={{
                          display: "block",
                          fontSize: "13px",
                          fontWeight: "500",
                          color: "#374151",
                          marginBottom: "6px",
                        }}
                      >
                        Price/Hour
                      </label>
                      <input
                        placeholder="0.00"
                        value={newSpot.price}
                        onChange={(e) =>
                          setNewSpot({ ...newSpot, price: e.target.value })
                        }
                        type="number"
                        step="0.01"
                        style={{
                          width: "100%",
                          padding: "9px 12px",
                          border: "1px solid #D1D5DB",
                          borderRadius: "6px",
                          fontSize: "13px",
                          transition: "border-color 0.2s ease",
                          outline: "none",
                          background: "#FFFFFF",
                          boxSizing: "border-box",
                        }}
                        onFocus={(e) =>
                          (e.target.style.borderColor = "#6366F1")
                        }
                        onBlur={(e) => (e.target.style.borderColor = "#D1D5DB")}
                      />
                    </div>
                    <div>
                      <label
                        style={{
                          display: "block",
                          fontSize: "13px",
                          fontWeight: "500",
                          color: "#374151",
                          marginBottom: "6px",
                        }}
                      >
                        Type
                      </label>
                      <input
                        placeholder="Covered/Open"
                        value={newSpot.type}
                        onChange={(e) =>
                          setNewSpot({ ...newSpot, type: e.target.value })
                        }
                        style={{
                          width: "100%",
                          padding: "9px 12px",
                          border: "1px solid #D1D5DB",
                          borderRadius: "6px",
                          fontSize: "13px",
                          transition: "border-color 0.2s ease",
                          outline: "none",
                          background: "#FFFFFF",
                          boxSizing: "border-box",
                        }}
                        onFocus={(e) =>
                          (e.target.style.borderColor = "#6366F1")
                        }
                        onBlur={(e) => (e.target.style.borderColor = "#D1D5DB")}
                      />
                    </div>
                  </div>

                  <div>
                    <label
                      style={{
                        display: "block",
                        fontSize: "13px",
                        fontWeight: "500",
                        color: "#374151",
                        marginBottom: "6px",
                      }}
                    >
                      Initial Slots
                    </label>
                    <input
                      placeholder="1-100"
                      value={newSpot.initial_slots || 1}
                      onChange={(e) =>
                        setNewSpot({
                          ...newSpot,
                          initial_slots: parseInt(e.target.value) || 1,
                        })
                      }
                      type="number"
                      min="1"
                      max="100"
                      style={{
                        width: "100%",
                        padding: "9px 12px",
                        border: "1px solid #D1D5DB",
                        borderRadius: "6px",
                        fontSize: "13px",
                        transition: "border-color 0.2s ease",
                        outline: "none",
                        background: "#FFFFFF",
                        boxSizing: "border-box",
                      }}
                      onFocus={(e) => (e.target.style.borderColor = "#6366F1")}
                      onBlur={(e) => (e.target.style.borderColor = "#D1D5DB")}
                    />
                  </div>

                  <button
                    onClick={addSpot}
                    style={{
                      padding: "10px 14px",
                      background:
                        "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
                      color: "#FFFFFF",
                      border: "none",
                      borderRadius: "6px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "6px",
                      transition: "transform 0.2s ease",
                      width: "100%",
                    }}
                    onMouseEnter={(e) =>
                      (e.target.style.transform = "scale(1.02)")
                    }
                    onMouseLeave={(e) =>
                      (e.target.style.transform = "scale(1)")
                    }
                  >
                    <Plus size={16} />
                    Add Parking Spot
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Properties List Section */}
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <div style={{ padding: "20px 20px 16px" }}>
              <div style={{ marginBottom: "16px" }}>
                <h2
                  style={{
                    margin: "0 0 4px 0",
                    fontSize: "18px",
                    fontWeight: "600",
                    color: "#111827",
                  }}
                >
                  Your Properties ({spots.length})
                </h2>
                <p style={{ margin: 0, fontSize: "13px", color: "#6B7280" }}>
                  Manage occupancy and status
                </p>
              </div>

              {/* Search Bar */}
              <div style={{ position: "relative" }}>
                <Search
                  size={16}
                  style={{
                    position: "absolute",
                    left: "12px",
                    top: "50%",
                    transform: "translateY(-50%)",
                    color: "#9CA3AF",
                  }}
                />
                <input
                  placeholder="Search for a property..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "10px 12px 10px 38px",
                    border: "1px solid #E5E7EB",
                    borderRadius: "8px",
                    fontSize: "13px",
                    outline: "none",
                    background: "#F9FAFB",
                    boxSizing: "border-box",
                    transition: "all 0.2s ease",
                  }}
                  onFocus={(e) => {
                    e.target.style.borderColor = "#6366F1";
                    e.target.style.background = "#FFFFFF";
                  }}
                  onBlur={(e) => {
                    e.target.style.borderColor = "#E5E7EB";
                    e.target.style.background = "#F9FAFB";
                  }}
                />
              </div>
            </div>

            {/* Properties List */}
            <div
              style={{
                flex: 1,
                overflowY: "auto",
                padding: "0 20px 20px",
              }}
            >
              {filteredSpots.length === 0 ? (
                <div
                  style={{
                    textAlign: "center",
                    padding: "40px 20px",
                    color: "#9CA3AF",
                  }}
                >
                  <MapPin size={48} style={{ margin: "0 auto 12px" }} />
                  <p style={{ margin: 0, fontSize: "14px" }}>
                    {searchQuery ? "No properties found" : "No properties yet"}
                  </p>
                </div>
              ) : (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "12px",
                  }}
                >
                  {filteredSpots.map((spot) => {
                    const isSelected = selectedSpotId === spot.id;
                    const isExpanded = expandedSpotId === spot.id;
                    const occupancy = getOccupancyLevel(spot);
                    const percentage = getOccupancyPercentage(spot);
                    const safeOccupiedSlots = getSafeOccupiedSlots(spot);

                    return (
                      <div
                        key={spot.id}
                        ref={(el) => (spotCardRefs.current[spot.id] = el)}
                        style={{
                          background: isSelected
                            ? "linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%)"
                            : "#FFFFFF",
                          border: isSelected
                            ? "2px solid #6366F1"
                            : "1px solid #E5E7EB",
                          borderRadius: "12px",
                          overflow: "hidden",
                          transition: "all 0.3s ease",
                          boxShadow: isSelected
                            ? "0 4px 12px rgba(99, 102, 241, 0.15)"
                            : "0 1px 2px rgba(0, 0, 0, 0.05)",
                        }}
                      >
                        {/* Card Header - Always Visible */}
                        <div
                          onClick={() => handleSpotSelect(spot)}
                          style={{
                            padding: "14px",
                            cursor: "pointer",
                            transition: "background 0.2s ease",
                          }}
                          onMouseEnter={(e) => {
                            if (!isSelected) {
                              e.currentTarget.style.backgroundColor = "#F9FAFB";
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (!isSelected) {
                              e.currentTarget.style.backgroundColor =
                                "transparent";
                            }
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "flex-start",
                              marginBottom: "10px",
                            }}
                          >
                            <div style={{ flex: 1 }}>
                              <h3
                                style={{
                                  margin: "0 0 4px 0",
                                  fontSize: "15px",
                                  fontWeight: "600",
                                  color: "#111827",
                                }}
                              >
                                {spot.name}
                              </h3>
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "6px",
                                  fontSize: "12px",
                                  color: "#6B7280",
                                }}
                              >
                                <MapPin size={12} />
                                {spot.type}
                              </div>
                            </div>
                            <div
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "4px",
                                padding: "4px 8px",
                                background: occupancy.level.includes("High")
                                  ? "#F3E8FF"
                                  : occupancy.level.includes("Busy")
                                    ? "#DBEAFE"
                                    : "#D1FAE5",
                                color: occupancy.color,
                                borderRadius: "6px",
                                fontSize: "11px",
                                fontWeight: "600",
                              }}
                            >
                              <div
                                style={{
                                  width: "6px",
                                  height: "6px",
                                  borderRadius: "50%",
                                  background: occupancy.color,
                                }}
                              />
                              Live
                            </div>
                          </div>

                          {/* Occupancy Progress */}
                          <div>
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                marginBottom: "6px",
                              }}
                            >
                              <span
                                style={{
                                  fontSize: "11px",
                                  fontWeight: "500",
                                  color: "#6B7280",
                                  textTransform: "uppercase",
                                  letterSpacing: "0.5px",
                                }}
                              >
                                Occupancy
                              </span>
                              <span
                                style={{
                                  fontSize: "13px",
                                  fontWeight: "600",
                                  color: "#111827",
                                }}
                              >
                                {safeOccupiedSlots} / {spot.total_slots} spots
                              </span>
                            </div>
                            <div
                              style={{
                                height: "6px",
                                background: "#E5E7EB",
                                borderRadius: "3px",
                                overflow: "hidden",
                              }}
                            >
                              <div
                                style={{
                                  width: `${Math.max(0, Math.min(100, percentage))}%`,
                                  height: "100%",
                                  background: occupancy.color,
                                  borderRadius: "3px",
                                  transition: "width 0.5s ease",
                                }}
                              />
                            </div>
                          </div>
                        </div>

                        {/* Expandable Details Section */}
                        <div
                          style={{
                            maxHeight: isExpanded ? "200px" : "0",
                            overflow: "hidden",
                            transition: "max-height 0.3s ease",
                            borderTop: isExpanded
                              ? "1px solid #E5E7EB"
                              : "none",
                          }}
                        >
                          <div style={{ padding: "14px" }}>
                            {/* Stats Grid */}
                            <div
                              style={{
                                display: "grid",
                                gridTemplateColumns: "1fr 1fr",
                                gap: "8px",
                                marginBottom: "10px",
                              }}
                            >
                              <div
                                style={{
                                  background: isSelected
                                    ? "#FFFFFF"
                                    : "#F9FAFB",
                                  padding: "10px",
                                  borderRadius: "6px",
                                  border: "1px solid #E5E7EB",
                                }}
                              >
                                <div
                                  style={{
                                    fontSize: "11px",
                                    color: "#6B7280",
                                    marginBottom: "4px",
                                    fontWeight: "500",
                                  }}
                                >
                                  Revenue
                                </div>
                                <div
                                  style={{
                                    fontSize: "16px",
                                    fontWeight: "600",
                                    color: "#6366F1",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "2px",
                                  }}
                                >
                                  <IndianRupee size={14} />
                                  {(safeOccupiedSlots * spot.price).toFixed(0)}
                                </div>
                              </div>
                              <div
                                style={{
                                  background: isSelected
                                    ? "#FFFFFF"
                                    : "#F9FAFB",
                                  padding: "10px",
                                  borderRadius: "6px",
                                  border: "1px solid #E5E7EB",
                                }}
                              >
                                <div
                                  style={{
                                    fontSize: "11px",
                                    color: "#6B7280",
                                    marginBottom: "4px",
                                    fontWeight: "500",
                                  }}
                                >
                                  Active
                                </div>
                                <div
                                  style={{
                                    fontSize: "16px",
                                    fontWeight: "600",
                                    color: "#10B981",
                                  }}
                                >
                                  {safeOccupiedSlots}
                                </div>
                              </div>
                            </div>
                            {/* Action Buttons */}
                            <div
                              style={{
                                display: "flex",
                                flexDirection: "column",
                                gap: "8px",
                              }}
                            >
                              <a
                                href={`/owner/slots/${spot.id}`}
                                style={{
                                  padding: "8px 12px",
                                  background:
                                    "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
                                  color: "#FFFFFF",
                                  border: "none",
                                  borderRadius: "6px",
                                  fontSize: "13px",
                                  fontWeight: "500",
                                  cursor: "pointer",
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                  gap: "6px",
                                  textDecoration: "none",
                                  marginBottom: "10px",
                                  transition: "transform 0.2s ease",
                                }}
                                onMouseEnter={(e) =>
                                  (e.target.style.transform = "scale(0.98)")
                                }
                                onMouseLeave={(e) =>
                                  (e.target.style.transform = "scale(1)")
                                }
                              >
                                Manage Slots
                              </a>
                            </div>

                            {/* Delete Button */}
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                removeSpot(spot.id);
                              }}
                              style={{
                                width: "100%",
                                padding: "10px",
                                background: "#FEF2F2",
                                color: "#EF4444",
                                border: "1px solid #FCA5A5",
                                borderRadius: "6px",
                                fontSize: "13px",
                                fontWeight: "500",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                gap: "6px",
                                transition: "all 0.2s ease",
                              }}
                              onMouseEnter={(e) => {
                                e.target.style.background = "#FEE2E2";
                              }}
                              onMouseLeave={(e) => {
                                e.target.style.background = "#FEF2F2";
                              }}
                            >
                              <Trash2 size={14} />
                              Delete Property
                            </button>
                          </div>
                        </div>

                        {/* Expand/Collapse Toggle Button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            const newExpandedState = isExpanded
                              ? null
                              : spot.id;
                            setExpandedSpotId(newExpandedState);

                            if (
                              newExpandedState === spot.id &&
                              markerRefs.current[spot.id]
                            ) {
                              markerRefs.current[spot.id].closePopup();
                            }

                            if (
                              newExpandedState === null &&
                              selectedSpotId === spot.id &&
                              markerRefs.current[spot.id]
                            ) {
                              setTimeout(() => {
                                markerRefs.current[spot.id].openPopup();
                              }, 100);
                            }
                          }}
                          style={{
                            width: "100%",
                            padding: "8px",
                            background: isExpanded
                              ? isSelected
                                ? "#FFFFFF"
                                : "#F9FAFB"
                              : "transparent",
                            color: "#6B7280",
                            border: "none",
                            borderTop: "1px solid #E5E7EB",
                            cursor: "pointer",
                            fontSize: "12px",
                            fontWeight: "500",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: "4px",
                            transition: "all 0.2s ease",
                          }}
                          onMouseEnter={(e) => {
                            e.target.style.color = "#374151";
                            e.target.style.background = isSelected
                              ? "#F3F4F6"
                              : "#F9FAFB";
                          }}
                          onMouseLeave={(e) => {
                            e.target.style.color = "#6B7280";
                            e.target.style.background = isExpanded
                              ? isSelected
                                ? "#FFFFFF"
                                : "#F9FAFB"
                              : "transparent";
                          }}
                        >
                          {isExpanded ? (
                            <>
                              <ChevronUp size={14} />
                              Show Less
                            </>
                          ) : (
                            <>
                              <ChevronDown size={14} />
                              Show Details
                            </>
                          )}
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Map Container */}
        <div style={{ flex: 1, position: "relative" }}>
          {isAddingMode && (
            <div
              style={{
                position: "absolute",
                top: "24px",
                left: "50%",
                transform: "translateX(-50%)",
                background: "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
                color: "#FFFFFF",
                padding: "10px 20px",
                borderRadius: "8px",
                fontSize: "14px",
                fontWeight: "500",
                zIndex: 1000,
                display: "flex",
                alignItems: "center",
                gap: "8px",
                boxShadow: "0 4px 12px rgba(99, 102, 241, 0.4)",
              }}
            >
              <MapPin size={16} />
              Click on the map to select location
            </div>
          )}

          <MapContainer
            center={userLocation || [20, 77]}
            zoom={userLocation ? 15 : 5}
            minZoom={3}
            maxZoom={18}
            worldCopyJump={true}
            maxBounds={[
              [-85, -180],
              [85, 180],
            ]}
            style={{
              height: "100%",
              width: "100%",
              cursor: isAddingMode ? "crosshair" : "default",
            }}
            zoomControl={true}
            attributionControl={false}
          >
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <MapClickHandler
              onMapClick={handleMapClick}
              isAddingMode={isAddingMode}
              onMapBackgroundClick={handleMapBackgroundClick}
            />
            <MapViewController center={mapCenter} zoom={mapZoom} />

            {spots.map((spot) => (
              <Marker
                key={spot.id}
                position={[spot.latitude, spot.longitude]}
                icon={createCustomIcon(spot, selectedSpotId === spot.id)}
                ref={(ref) => {
                  markerRefs.current[spot.id] = ref;
                }}
                eventHandlers={{
                  click: (e) => {
                    e.originalEvent.stopPropagation();
                    handleSpotSelect(spot);
                  },
                }}
              >
                <Popup>
                  <div style={{ minWidth: "240px", padding: "4px" }}>
                    <h4
                      style={{
                        margin: "0 0 8px 0",
                        fontSize: "15px",
                        fontWeight: "600",
                        color: "#111827",
                      }}
                    >
                      {spot.name}
                    </h4>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        marginBottom: "12px",
                        paddingBottom: "12px",
                        borderBottom: "1px solid #E5E7EB",
                      }}
                    >
                      <IndianRupee size={14} />
                      <span style={{ fontSize: "14px", fontWeight: "600" }}>
                        {spot.price}/hour
                      </span>
                    </div>

                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr 1fr",
                        gap: "6px",
                        marginBottom: "12px",
                      }}
                    >
                      <div
                        style={{
                          background: "#F9FAFB",
                          padding: "6px",
                          borderRadius: "4px",
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "16px",
                            fontWeight: "600",
                            color: "#111827",
                          }}
                        >
                          {spot.total_slots}
                        </div>
                        <div style={{ fontSize: "10px", color: "#6B7280" }}>
                          Total
                        </div>
                      </div>
                      <div
                        style={{
                          background: "#D1FAE5",
                          padding: "6px",
                          borderRadius: "4px",
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "16px",
                            fontWeight: "600",
                            color: "#10B981",
                          }}
                        >
                          {spot.available_slots}
                        </div>
                        <div style={{ fontSize: "10px", color: "#059669" }}>
                          Free
                        </div>
                      </div>
                      <div
                        style={{
                          background: "#FEE2E2",
                          padding: "6px",
                          borderRadius: "4px",
                          textAlign: "center",
                        }}
                      >
                        <div
                          style={{
                            fontSize: "16px",
                            fontWeight: "600",
                            color: "#EF4444",
                          }}
                        >
                          {getSafeOccupiedSlots(spot)}
                        </div>
                        <div style={{ fontSize: "10px", color: "#DC2626" }}>
                          Used
                        </div>
                      </div>
                    </div>
                    {/* Action Buttons */}
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "8px",
                      }}
                    >
                      <a
                        href={`/owner/slots/${spot.id}`}
                        style={{
                          padding: "8px 12px",
                          background:
                            "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
                          color: "#FFFFFF",
                          border: "none",
                          borderRadius: "6px",
                          fontSize: "13px",
                          fontWeight: "500",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          gap: "6px",
                          textDecoration: "none",
                          marginBottom: "10px",
                          transition: "transform 0.2s ease",
                        }}
                        onMouseEnter={(e) =>
                          (e.target.style.transform = "scale(0.98)")
                        }
                        onMouseLeave={(e) =>
                          (e.target.style.transform = "scale(1)")
                        }
                      >
                        Manage Slots
                      </a>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeSpot(spot.id);
                      }}
                      style={{
                        width: "100%",
                        padding: "8px",
                        background: "#FEF2F2",
                        color: "#EF4444",
                        border: "1px solid #FCA5A5",
                        borderRadius: "6px",
                        fontSize: "12px",
                        fontWeight: "500",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: "6px",
                      }}
                    >
                      <Trash2 size={12} />
                      Delete
                    </button>
                  </div>
                </Popup>
              </Marker>
            ))}
          </MapContainer>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      {showDeleteDialog && (
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
            zIndex: 10000,
            backdropFilter: "blur(4px)",
          }}
        >
          <div
            style={{
              background: "#FFFFFF",
              borderRadius: "12px",
              padding: "24px",
              maxWidth: "400px",
              width: "90%",
              boxShadow:
                "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
              border: "1px solid #E2E8F0",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                marginBottom: "16px",
              }}
            >
              <div
                style={{
                  width: "48px",
                  height: "48px",
                  borderRadius: "50%",
                  background: "#FEF2F2",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <AlertTriangle size={24} style={{ color: "#EF4444" }} />
              </div>
              <div>
                <h3
                  style={{
                    margin: 0,
                    fontSize: "18px",
                    fontWeight: "600",
                    color: "#1E293B",
                  }}
                >
                  Delete Parking Spot
                </h3>
                <p
                  style={{
                    margin: "4px 0 0 0",
                    fontSize: "14px",
                    color: "#64748B",
                  }}
                >
                  This action cannot be undone
                </p>
              </div>
            </div>

            <p
              style={{
                margin: "0 0 24px 0",
                fontSize: "14px",
                color: "#374151",
                lineHeight: "1.5",
              }}
            >
              Are you sure you want to permanently delete this parking spot?
              This will remove all associated data.
            </p>

            <div
              style={{
                display: "flex",
                gap: "12px",
                justifyContent: "flex-end",
              }}
            >
              <button
                onClick={cancelDelete}
                style={{
                  padding: "10px 16px",
                  background: "#FFFFFF",
                  color: "#374151",
                  border: "1px solid #D1D5DB",
                  borderRadius: "6px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                style={{
                  padding: "10px 16px",
                  background: "#EF4444",
                  color: "#FFFFFF",
                  border: "none",
                  borderRadius: "6px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: "pointer",
                  transition: "background-color 0.2s ease",
                }}
              >
                Delete Spot
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default EnhancedOwnerDashboard;
