import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Plus,
  Trash2,
  Eye,
  EyeOff,
  ArrowLeft,
  RefreshCw,
  AlertTriangle,
  Search,
  Sliders,
} from "lucide-react";
import { useParams, useNavigate } from "react-router-dom";

const getSlotStatusColor = (slot) => {
  if (slot.status === "disabled")
    return {
      bg: "#F1F5F9",
      text: "#64748B",
      border: "#E2E8F0",
      badge: "#64748B",
      badgeBg: "#F1F5F9",
    };
  if (slot.status === "occupied")
    return {
      bg: "#FEF2F2",
      text: "#DC2626",
      border: "#FEE2E2",
      badge: "#DC2626",
      badgeBg: "#DBEAFE",
    };
  return {
    bg: "#F0F9FF",
    text: "#0284C7",
    border: "#E0F2FE",
    badge: "#0284C7",
    badgeBg: "#DBEAFE",
  };
};

const SlotCard = React.memo(function SlotCard({
  slot,
  onToggle,
  onOpenDelete,
  bulkMode,
  isSelected,
  onSelect,
}) {
  const colors = getSlotStatusColor(slot);

  return (
    <div
      style={{
        background: "#FFFFFF",
        border: `2px solid ${isSelected ? "#3B82F6" : colors.border}`,
        borderRadius: "12px",
        padding: "20px",
        transition: "all 0.2s ease",
        position: "relative",
        cursor: bulkMode ? "pointer" : "default",
        boxShadow: isSelected
          ? "0 4px 12px rgba(59, 130, 246, 0.15)"
          : "0 1px 3px rgba(0, 0, 0, 0.05)",
      }}
      onClick={() => bulkMode && onSelect(slot.slot_id)}
    >
      {/* Status Badge */}
      <div
        style={{
          position: "absolute",
          top: "12px",
          left: "12px",
          display: "inline-flex",
          alignItems: "center",
          gap: "4px",
          padding: "4px 10px",
          borderRadius: "12px",
          fontSize: "11px",
          fontWeight: "600",
          textTransform: "uppercase",
          letterSpacing: "0.5px",
          background: colors.badgeBg,
          color: colors.badge,
        }}
      >
        <div
          style={{
            width: "6px",
            height: "6px",
            borderRadius: "50%",
            background: colors.badge,
          }}
        />
        {slot.status}
      </div>

      {/* Selection Checkbox */}
      {bulkMode && (
        <div
          style={{
            position: "absolute",
            top: "12px",
            right: "12px",
            width: "20px",
            height: "20px",
            borderRadius: "6px",
            border: `2px solid ${isSelected ? "#3B82F6" : "#D1D5DB"}`,
            background: isSelected ? "#3B82F6" : "#FFFFFF",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {isSelected && (
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path
                d="M10 3L4.5 8.5L2 6"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </div>
      )}

      {/* Slot Number */}
      <div
        style={{
          fontSize: "32px",
          fontWeight: "700",
          color: "#1E293B",
          marginTop: "32px",
          marginBottom: "8px",
          letterSpacing: "-0.5px",
        }}
      >
        #{slot.slot_number}
      </div>

      {/* Booking Info */}
      {slot.current_booking_id && (
        <div
          style={{
            background: "#F8FAFC",
            padding: "12px",
            borderRadius: "8px",
            fontSize: "13px",
            color: "#475569",
            marginBottom: "16px",
            border: "1px solid #E2E8F0",
          }}
        >
          <div style={{ marginBottom: "4px" }}>
            <strong style={{ color: "#1E293B" }}>User:</strong>{" "}
            {slot.current_user_name}
          </div>
          <div style={{ fontSize: "12px", color: "#64748B" }}>
            <strong>Until:</strong> {new Date(slot.expires_at).toLocaleString()}
          </div>
        </div>
      )}

      {/* Action Buttons */}
      {!bulkMode && (
        <div style={{ display: "flex", gap: "8px", marginTop: "16px" }}>
          <button
            onClick={() => onToggle(slot.slot_id, slot.is_active)}
            disabled={slot.status === "occupied"}
            style={{
              flex: 1,
              padding: "10px 16px",
              background:
                slot.status === "occupied"
                  ? "#F8FAFC"
                  : slot.is_active
                    ? "#FFFFFF"
                    : "#3B82F6",
              color:
                slot.status === "occupied"
                  ? "#94A3B8"
                  : slot.is_active
                    ? "#DC2626"
                    : "#FFFFFF",
              border:
                slot.status === "occupied"
                  ? "1px solid #E2E8F0"
                  : slot.is_active
                    ? "1px solid #FCA5A5"
                    : "none",
              borderRadius: "8px",
              fontSize: "13px",
              fontWeight: "600",
              cursor: slot.status === "occupied" ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "6px",
              transition: "all 0.2s ease",
            }}
          >
            {slot.is_active ? <EyeOff size={14} /> : <Eye size={14} />}
            {slot.is_active ? "Disable" : "Enable"}
          </button>
          <button
            onClick={() => onOpenDelete(slot.slot_id)}
            disabled={slot.status === "occupied"}
            style={{
              padding: "10px 12px",
              background: slot.status === "occupied" ? "#F8FAFC" : "#FFFFFF",
              color: slot.status === "occupied" ? "#94A3B8" : "#EF4444",
              border: `1px solid ${slot.status === "occupied" ? "#E2E8F0" : "#FCA5A5"}`,
              borderRadius: "8px",
              fontSize: "13px",
              cursor: slot.status === "occupied" ? "not-allowed" : "pointer",
              transition: "all 0.2s ease",
            }}
          >
            <Trash2 size={14} />
          </button>
        </div>
      )}
    </div>
  );
});

function SlotManagement() {
  const { spotId } = useParams();
  const navigate = useNavigate();

  const [spotInfo, setSpotInfo] = useState(null);
  const [slots, setSlots] = useState([]);
  const [message, setMessage] = useState("");
  const [slotsToAdd, setSlotsToAdd] = useState(1);
  const [loading, setLoading] = useState(true);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [isUpdating, setIsUpdating] = useState(false);
  const [bulkMode, setBulkMode] = useState(false);
  const [selectedSlots, setSelectedSlots] = useState([]);
  const [rangeStart, setRangeStart] = useState("");
  const [rangeEnd, setRangeEnd] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(null);
  const [forceResetConfirm, setForceResetConfirm] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [filterStatus, setFilterStatus] = useState("all");
  const slotsPerPage = 10;

  const token = localStorage.getItem("token");
  const isFirstLoadRef = useRef(true);
  const loadingRef = useRef(false);
  const slotsStringRef = useRef("");
  const spotInfoStringRef = useRef("");

  const fetchData = useCallback(() => {
    if (loadingRef.current) return;
    loadingRef.current = true;

    if (isFirstLoadRef.current) {
      setLoading(true);
    } else {
      setIsUpdating(true);
    }

    fetch(`http://localhost:5000/api/spots/${spotId}/details`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        const next = safeStringify(data);
        if (next !== spotInfoStringRef.current) {
          spotInfoStringRef.current = next;
          setSpotInfo(data);
        }
      })
      .catch(() => setMessage("Failed to load spot details"));

    fetch(`http://localhost:5000/api/slots/parking-spot/${spotId}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        const next = safeStringify(data);
        if (next !== slotsStringRef.current) {
          slotsStringRef.current = next;
          setSlots(data);
        }
        if (isFirstLoadRef.current) {
          setLoading(false);
          isFirstLoadRef.current = false;
        }
        setIsUpdating(false);
        loadingRef.current = false;
      })
      .catch(() => {
        setMessage("Failed to load slots");
        if (isFirstLoadRef.current) {
          setLoading(false);
          isFirstLoadRef.current = false;
        }
        setIsUpdating(false);
        loadingRef.current = false;
      });
  }, [spotId, token]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, [fetchData]);

  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => setMessage(""), 5000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  const handleAddSlots = useCallback(() => {
    if (slotsToAdd < 1 || slotsToAdd > 50) {
      setMessage("Can add between 1 and 50 slots at a time");
      return;
    }

    fetch(`http://localhost:5000/api/slots/parking-spot/${spotId}/add`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ count: slotsToAdd }),
    })
      .then((res) => res.json())
      .then((data) => {
        setMessage(data.message);
        setSlotsToAdd(1);
        fetchData();
      })
      .catch(() => setMessage("Failed to add slots"));
  }, [slotsToAdd, spotId, token, fetchData]);

  const handleToggleSlot = useCallback(
    (slotId, currentStatus) => {
      fetch(`http://localhost:5000/api/slots/${slotId}/toggle`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ is_active: !currentStatus }),
      })
        .then((res) => res.json())
        .then((data) => {
          setMessage(data.message);
          fetchData();
        })
        .catch(() => setMessage("Failed to toggle slot"));
    },
    [token, fetchData],
  );

  const handleDeleteSlot = useCallback(
    (slotId) => {
      fetch(`http://localhost:5000/api/slots/${slotId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      })
        .then((res) => res.json())
        .then((data) => {
          setMessage(data.message);
          setDeleteConfirm(null);
          fetchData();
        })
        .catch(() => setMessage("Failed to delete slot"));
    },
    [token, fetchData],
  );

  const handleSelectSlot = useCallback((slotId) => {
    setSelectedSlots((prev) =>
      prev.includes(slotId)
        ? prev.filter((id) => id !== slotId)
        : [...prev, slotId],
    );
  }, []);

  const handleSelectAll = useCallback(() => {
    setSelectedSlots(slots.map((s) => s.slot_id));
  }, [slots]);

  const handleClearSelection = useCallback(() => {
    setSelectedSlots([]);
    setBulkMode(false);
  }, []);

  const handleBulkEnableAll = useCallback(() => {
    const availableSlotIds = slots
      .filter((s) => s.status !== "occupied" && !s.is_active)
      .map((s) => s.slot_id);

    if (availableSlotIds.length === 0) {
      setMessage("No slots available to enable");
      return;
    }

    Promise.all(
      availableSlotIds.map((slotId) =>
        fetch(`http://localhost:5000/api/slots/${slotId}/toggle`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ is_active: true }),
        }),
      ),
    )
      .then(() => {
        setMessage(`Successfully enabled ${availableSlotIds.length} slots`);
        fetchData();
      })
      .catch(() => setMessage("Failed to enable all slots"));
  }, [slots, token, fetchData]);

  const handleBulkDisableAll = useCallback(() => {
    const disableableSlots = slots
      .filter((s) => s.status !== "occupied" && s.is_active)
      .map((s) => s.slot_id);

    if (disableableSlots.length === 0) {
      setMessage("No slots available to disable");
      return;
    }

    Promise.all(
      disableableSlots.map((slotId) =>
        fetch(`http://localhost:5000/api/slots/${slotId}/toggle`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ is_active: false }),
        }),
      ),
    )
      .then(() => {
        setMessage(`Successfully disabled ${disableableSlots.length} slots`);
        fetchData();
      })
      .catch(() => setMessage("Failed to disable all slots"));
  }, [slots, token, fetchData]);

  const handleBulkDelete = useCallback(() => {
    if (selectedSlots.length === 0) {
      setMessage("No slots selected");
      return;
    }

    const deleteableSlots = slots.filter(
      (s) => selectedSlots.includes(s.slot_id) && s.status !== "occupied",
    );

    if (deleteableSlots.length === 0) {
      setMessage("Selected slots cannot be deleted (occupied or invalid)");
      return;
    }

    // Show custom confirmation modal
    setBulkDeleteConfirm(deleteableSlots.length);
  }, [selectedSlots, slots]);

  const confirmBulkDelete = useCallback(() => {
    const deleteableSlots = slots.filter(
      (s) => selectedSlots.includes(s.slot_id) && s.status !== "occupied",
    );

    Promise.all(
      deleteableSlots.map((slot) =>
        fetch(`http://localhost:5000/api/slots/${slot.slot_id}`, {
          method: "DELETE",
          headers: { Authorization: `Bearer ${token}` },
        }),
      ),
    )
      .then(() => {
        setMessage(`Successfully deleted ${deleteableSlots.length} slots`);
        setSelectedSlots([]);
        setBulkMode(false);
        setBulkDeleteConfirm(null);
        fetchData();
      })
      .catch(() => {
        setMessage("Failed to delete selected slots");
        setBulkDeleteConfirm(null);
      });
  }, [selectedSlots, slots, token, fetchData]);

  const handleRangeToggle = useCallback(
    (enable) => {
      const start = parseInt(rangeStart);
      const end = parseInt(rangeEnd);

      if (!start || !end || start > end) {
        setMessage("Invalid range. Start must be less than or equal to end.");
        return;
      }

      const slotsInRange = slots.filter(
        (s) =>
          s.slot_number >= start &&
          s.slot_number <= end &&
          s.status !== "occupied",
      );

      if (slotsInRange.length === 0) {
        setMessage(`No valid slots found in range ${start}-${end}`);
        return;
      }

      Promise.all(
        slotsInRange.map((slot) =>
          fetch(`http://localhost:5000/api/slots/${slot.slot_id}/toggle`, {
            method: "PUT",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ is_active: enable }),
          }),
        ),
      )
        .then(() => {
          setMessage(
            `Successfully ${enable ? "enabled" : "disabled"} ${
              slotsInRange.length
            } slots (${start}-${end})`,
          );
          setRangeStart("");
          setRangeEnd("");
          fetchData();
        })
        .catch(() => setMessage("Failed to update range"));
    },
    [rangeStart, rangeEnd, slots, token, fetchData],
  );

  const handleForceReset = useCallback(() => {
    setForceResetConfirm(true);
  }, []);

  const confirmForceReset = useCallback(() => {
    fetch(
      `http://localhost:5000/api/slots/parking-spot/${spotId}/force-reset`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      },
    )
      .then((res) => res.json())
      .then((data) => {
        if (data.success) {
          setMessage(
            `✅ ${data.message} (${data.cancelled_bookings} bookings cancelled, ${data.reset_slots} slots reset)`,
          );
          fetchData();
        } else {
          setMessage(`❌ ${data.message || "Force reset failed"}`);
        }
        setForceResetConfirm(false);
      })
      .catch(() => {
        setMessage("❌ Failed to force reset slots");
        setForceResetConfirm(false);
      });
  }, [spotId, token, fetchData]);

  const handleOpenDelete = useCallback(
    (slotId) => setDeleteConfirm(slotId),
    [],
  );

  // Filter and paginate slots
  const filteredSlots = slots.filter((slot) => {
    const matchesSearch =
      slot.slot_number.toString().includes(searchTerm) ||
      slot.status.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesFilter =
      filterStatus === "all"
        ? true
        : filterStatus === "available"
          ? slot.status === "available"
          : filterStatus === "occupied"
            ? slot.status === "occupied"
            : filterStatus === "disabled"
              ? slot.status === "disabled"
              : true;

    return matchesSearch && matchesFilter;
  });

  const totalPages = Math.ceil(filteredSlots.length / slotsPerPage);
  const startIndex = (currentPage - 1) * slotsPerPage;
  const endIndex = startIndex + slotsPerPage;
  const paginatedSlots = filteredSlots.slice(startIndex, endIndex);

  // Reset to page 1 when filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filterStatus, searchTerm]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#F8FAFC",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}
    >
      {/* Header */}
      <div
        style={{
          background: "#FFFFFF",
          borderBottom: "1px solid #E2E8F0",
          padding: "16px 32px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            <button
              onClick={() => navigate("/owner/dashboard")}
              style={{
                background: "transparent",
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                padding: "8px",
              }}
            >
              <ArrowLeft size={20} color="#64748B" />
            </button>

            <div>
              <div
                style={{
                  fontSize: "12px",
                  color: "#64748B",
                  marginBottom: "2px",
                }}
              >
                Properties › Downtown Garage
              </div>
              <h1
                style={{
                  margin: 0,
                  fontSize: "20px",
                  fontWeight: "600",
                  color: "#1E293B",
                }}
              >
                Level 1 - Overview
              </h1>
            </div>
          </div>

          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            {isUpdating && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  color: "#64748B",
                  fontSize: "12px",
                }}
              >
                <div
                  style={{
                    width: "6px",
                    height: "6px",
                    borderRadius: "50%",
                    background: "#3B82F6",
                    animation: "pulse 1.5s ease-in-out infinite",
                  }}
                />
              </div>
            )}

            <button
              onClick={fetchData}
              style={{
                background: "#FFFFFF",
                border: "1px solid #E2E8F0",
                borderRadius: "8px",
                padding: "8px 16px",
                fontSize: "14px",
                fontWeight: "500",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "6px",
                color: "#475569",
              }}
            >
              <RefreshCw size={16} />
              Refresh
            </button>

            <button
              onClick={() => navigate(`/owner/ai-setup/${spotId}`)}
              style={{
                padding: "8px 16px",
                background: "#3B82F6",
                color: "#FFFFFF",
                border: "none",
                borderRadius: "8px",
                cursor: "pointer",
                fontSize: "14px",
                fontWeight: "500",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <Sliders size={16} />
              AI Detect
            </button>

            <button
              onClick={handleForceReset}
              style={{
                padding: "8px 16px",
                background: "#FFFFFF",
                color: "#475569",
                border: "1px solid #E2E8F0",
                borderRadius: "8px",
                cursor: "pointer",
                fontSize: "14px",
                fontWeight: "500",
                display: "flex",
                alignItems: "center",
                gap: "6px",
              }}
            >
              <AlertTriangle size={16} />
              Force Reset
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <div style={{ marginTop: "16px" }}>
          <div style={{ position: "relative", maxWidth: "400px" }}>
            <Search
              size={18}
              color="#94A3B8"
              style={{
                position: "absolute",
                left: "12px",
                top: "50%",
                transform: "translateY(-50%)",
              }}
            />
            <input
              type="text"
              placeholder="Search slot ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: "100%",
                padding: "10px 12px 10px 40px",
                border: "1px solid #E2E8F0",
                borderRadius: "8px",
                fontSize: "14px",
                background: "#FFFFFF",
                boxSizing: "border-box",
              }}
            />
          </div>
        </div>
      </div>

      <div style={{ padding: "24px 32px" }}>
        {/* Message */}
        {message && (
          <div
            style={{
              background:
                message.includes("Failed") || message.includes("Cannot")
                  ? "#FEF2F2"
                  : "#F0FDF4",
              color:
                message.includes("Failed") || message.includes("Cannot")
                  ? "#DC2626"
                  : "#16A34A",
              padding: "12px 16px",
              borderRadius: "8px",
              fontSize: "14px",
              marginBottom: "20px",
              border: `1px solid ${
                message.includes("Failed") || message.includes("Cannot")
                  ? "#FCA5A5"
                  : "#BBF7D0"
              }`,
            }}
          >
            {message}
          </div>
        )}

        {/* Stats Cards */}
        {spotInfo && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: "20px",
              marginBottom: "24px",
            }}
          >
            <div
              style={{
                background: "#FFFFFF",
                padding: "24px",
                borderRadius: "12px",
                border: "1px solid #E2E8F0",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748B",
                  marginBottom: "8px",
                  fontWeight: "500",
                }}
              >
                Total Capacity
              </div>
              <div
                style={{
                  fontSize: "36px",
                  fontWeight: "700",
                  color: "#1E293B",
                  letterSpacing: "-1px",
                }}
              >
                {spotInfo.total_slots}
              </div>
              <div
                style={{ fontSize: "12px", color: "#94A3B8", marginTop: "4px" }}
              >
                slots
              </div>
              <div
                style={{
                  marginTop: "12px",
                  height: "4px",
                  background: "#E2E8F0",
                  borderRadius: "2px",
                }}
              >
                <div
                  style={{
                    width: "100%",
                    height: "100%",
                    background: "#3B82F6",
                    borderRadius: "2px",
                  }}
                />
              </div>
            </div>

            <div
              style={{
                background: "#FFFFFF",
                padding: "24px",
                borderRadius: "12px",
                border: "1px solid #E2E8F0",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748B",
                  marginBottom: "8px",
                  fontWeight: "500",
                }}
              >
                Available Now
              </div>
              <div
                style={{
                  fontSize: "36px",
                  fontWeight: "700",
                  color: "#0284C7",
                  letterSpacing: "-1px",
                }}
              >
                {spotInfo.available_slots}
              </div>
              <div
                style={{ fontSize: "12px", color: "#94A3B8", marginTop: "4px" }}
              >
                {Math.round(
                  (spotInfo.available_slots / spotInfo.total_slots) * 100,
                )}
                % Free
              </div>
              <div
                style={{
                  marginTop: "12px",
                  height: "4px",
                  background: "#E0F2FE",
                  borderRadius: "2px",
                }}
              >
                <div
                  style={{
                    width: `${(spotInfo.available_slots / spotInfo.total_slots) * 100}%`,
                    height: "100%",
                    background: "#0284C7",
                    borderRadius: "2px",
                  }}
                />
              </div>
            </div>

            <div
              style={{
                background: "#FFFFFF",
                padding: "24px",
                borderRadius: "12px",
                border: "1px solid #E2E8F0",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748B",
                  marginBottom: "8px",
                  fontWeight: "500",
                }}
              >
                Occupied
              </div>
              <div
                style={{
                  fontSize: "36px",
                  fontWeight: "700",
                  color: "#1E293B",
                  letterSpacing: "-1px",
                }}
              >
                {spotInfo.occupied_slots || 0}
              </div>
              <div
                style={{ fontSize: "12px", color: "#94A3B8", marginTop: "4px" }}
              >
                Active sessions
              </div>
              <div
                style={{
                  marginTop: "12px",
                  height: "4px",
                  background: "#F1F5F9",
                  borderRadius: "2px",
                }}
              >
                <div
                  style={{
                    width: `${((spotInfo.occupied_slots || 0) / spotInfo.total_slots) * 100}%`,
                    height: "100%",
                    background: "#64748B",
                    borderRadius: "2px",
                  }}
                />
              </div>
            </div>

            <div
              style={{
                background: "#FFFFFF",
                padding: "24px",
                borderRadius: "12px",
                border: "1px solid #E2E8F0",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748B",
                  marginBottom: "8px",
                  fontWeight: "500",
                }}
              >
                Maintenance
              </div>
              <div
                style={{
                  fontSize: "36px",
                  fontWeight: "700",
                  color: "#1E293B",
                  letterSpacing: "-1px",
                }}
              >
                {spotInfo.total_slots - spotInfo.active_slots}
              </div>
              <div
                style={{ fontSize: "12px", color: "#94A3B8", marginTop: "4px" }}
              >
                Requires attention
              </div>
            </div>
          </div>
        )}

        {/* Bulk Operations */}
        <div
          style={{
            background: "#FFFFFF",
            padding: "20px 24px",
            borderRadius: "12px",
            border: "1px solid #E2E8F0",
            marginBottom: "24px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  marginBottom: "4px",
                }}
              >
                <Sliders size={18} color="#1E293B" />
                <h3
                  style={{
                    margin: 0,
                    fontSize: "16px",
                    fontWeight: "600",
                    color: "#1E293B",
                  }}
                >
                  Bulk Operations
                </h3>
              </div>
              <p style={{ margin: 0, fontSize: "13px", color: "#64748B" }}>
                Apply settings to multiple selected slots at once.
              </p>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
              <div
                style={{ display: "flex", alignItems: "center", gap: "8px" }}
              >
                <span style={{ fontSize: "13px", color: "#64748B" }}>
                  Selection Mode
                </span>
                <label
                  style={{
                    position: "relative",
                    display: "inline-block",
                    width: "44px",
                    height: "24px",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={bulkMode}
                    onChange={() => {
                      setBulkMode(!bulkMode);
                      if (bulkMode) setSelectedSlots([]);
                    }}
                    style={{ opacity: 0, width: 0, height: 0 }}
                  />
                  <span
                    style={{
                      position: "absolute",
                      cursor: "pointer",
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      background: bulkMode ? "#3B82F6" : "#E2E8F0",
                      borderRadius: "24px",
                      transition: "0.3s",
                    }}
                  >
                    <span
                      style={{
                        position: "absolute",
                        content: "",
                        height: "18px",
                        width: "18px",
                        left: bulkMode ? "23px" : "3px",
                        bottom: "3px",
                        background: "white",
                        borderRadius: "50%",
                        transition: "0.3s",
                      }}
                    />
                  </span>
                </label>
              </div>

              <button
                onClick={handleBulkEnableAll}
                style={{
                  padding: "8px 16px",
                  background: "#ECFDF5",
                  color: "#16A34A",
                  border: "1px solid #BBF7D0",
                  borderRadius: "8px",
                  fontSize: "13px",
                  fontWeight: "500",
                  cursor: "pointer",
                }}
              >
                Enable All
              </button>

              <button
                onClick={handleBulkDisableAll}
                style={{
                  padding: "8px 16px",
                  background: "#FEF2F2",
                  color: "#DC2626",
                  border: "1px solid #FCA5A5",
                  borderRadius: "8px",
                  fontSize: "13px",
                  fontWeight: "500",
                  cursor: "pointer",
                }}
              >
                Disable All
              </button>

              {bulkMode && (
                <>
                  <button
                    onClick={handleSelectAll}
                    style={{
                      padding: "8px 16px",
                      background: "#FFFFFF",
                      color: "#475569",
                      border: "1px solid #E2E8F0",
                      borderRadius: "8px",
                      fontSize: "13px",
                      fontWeight: "500",
                      cursor: "pointer",
                    }}
                  >
                    Select All
                  </button>

                  {selectedSlots.length > 0 && (
                    <button
                      onClick={handleBulkDelete}
                      style={{
                        padding: "8px 16px",
                        background: "#EF4444",
                        color: "#FFFFFF",
                        border: "none",
                        borderRadius: "8px",
                        fontSize: "13px",
                        fontWeight: "600",
                        cursor: "pointer",
                      }}
                    >
                      Delete Selected ({selectedSlots.length})
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        </div>

        {/* Add Slots & Range Operations */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "20px",
            marginBottom: "24px",
          }}
        >
          {/* Add Slots */}
          <div
            style={{
              background: "#FFFFFF",
              padding: "20px",
              borderRadius: "12px",
              border: "1px solid #E2E8F0",
            }}
          >
            <h3
              style={{
                margin: "0 0 12px 0",
                fontSize: "14px",
                fontWeight: "600",
                color: "#1E293B",
              }}
            >
              Add New Slots
            </h3>
            <div style={{ display: "flex", gap: "12px" }}>
              <input
                type="number"
                min="1"
                max="50"
                value={slotsToAdd}
                onChange={(e) => setSlotsToAdd(parseInt(e.target.value) || 1)}
                style={{
                  flex: 1,
                  padding: "10px 12px",
                  border: "1px solid #E2E8F0",
                  borderRadius: "8px",
                  fontSize: "14px",
                }}
              />
              <button
                onClick={handleAddSlots}
                style={{
                  padding: "10px 20px",
                  background: "#1E293B",
                  color: "#FFFFFF",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  whiteSpace: "nowrap",
                }}
              >
                <Plus size={16} />
                Add
              </button>
            </div>
          </div>

          {/* Range Operations */}
          <div
            style={{
              background: "#FFFFFF",
              padding: "20px",
              borderRadius: "12px",
              border: "1px solid #E2E8F0",
            }}
          >
            <h3
              style={{
                margin: "0 0 12px 0",
                fontSize: "14px",
                fontWeight: "600",
                color: "#1E293B",
              }}
            >
              Range Operations
            </h3>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr 2fr",
                gap: "8px",
              }}
            >
              <input
                type="number"
                value={rangeStart}
                onChange={(e) => setRangeStart(e.target.value)}
                placeholder="From"
                style={{
                  padding: "10px 12px",
                  border: "1px solid #E2E8F0",
                  borderRadius: "8px",
                  fontSize: "14px",
                }}
              />
              <input
                type="number"
                value={rangeEnd}
                onChange={(e) => setRangeEnd(e.target.value)}
                placeholder="To"
                style={{
                  padding: "10px 12px",
                  border: "1px solid #E2E8F0",
                  borderRadius: "8px",
                  fontSize: "14px",
                }}
              />
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  onClick={() => handleRangeToggle(true)}
                  disabled={!rangeStart || !rangeEnd}
                  style={{
                    flex: 1,
                    padding: "10px",
                    background:
                      !rangeStart || !rangeEnd ? "#F8FAFC" : "#22C55E",
                    color: !rangeStart || !rangeEnd ? "#94A3B8" : "#FFFFFF",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "13px",
                    fontWeight: "500",
                    cursor:
                      !rangeStart || !rangeEnd ? "not-allowed" : "pointer",
                  }}
                >
                  Enable
                </button>
                <button
                  onClick={() => handleRangeToggle(false)}
                  disabled={!rangeStart || !rangeEnd}
                  style={{
                    flex: 1,
                    padding: "10px",
                    background:
                      !rangeStart || !rangeEnd ? "#F8FAFC" : "#EF4444",
                    color: !rangeStart || !rangeEnd ? "#94A3B8" : "#FFFFFF",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "13px",
                    fontWeight: "500",
                    cursor:
                      !rangeStart || !rangeEnd ? "not-allowed" : "pointer",
                  }}
                >
                  Disable
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Individual Slots */}
        <div style={{ marginBottom: "20px" }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "16px",
            }}
          >
            <h3
              style={{
                margin: 0,
                fontSize: "18px",
                fontWeight: "600",
                color: "#1E293B",
              }}
            >
              Individual Slots
            </h3>

            {/* Filter Tabs */}
            <div
              style={{
                display: "flex",
                gap: "8px",
                background: "#F1F5F9",
                padding: "4px",
                borderRadius: "8px",
              }}
            >
              <button
                onClick={() => setFilterStatus("all")}
                style={{
                  padding: "6px 16px",
                  background:
                    filterStatus === "all" ? "#FFFFFF" : "transparent",
                  color: filterStatus === "all" ? "#1E293B" : "#64748B",
                  border: "none",
                  borderRadius: "6px",
                  fontSize: "13px",
                  fontWeight: "600",
                  cursor: "pointer",
                  boxShadow:
                    filterStatus === "all"
                      ? "0 1px 3px rgba(0, 0, 0, 0.1)"
                      : "none",
                  transition: "all 0.2s ease",
                }}
              >
                All
              </button>
              <button
                onClick={() => setFilterStatus("available")}
                style={{
                  padding: "6px 16px",
                  background:
                    filterStatus === "available" ? "#FFFFFF" : "transparent",
                  color: filterStatus === "available" ? "#0284C7" : "#64748B",
                  border: "none",
                  borderRadius: "6px",
                  fontSize: "13px",
                  fontWeight: "600",
                  cursor: "pointer",
                  boxShadow:
                    filterStatus === "available"
                      ? "0 1px 3px rgba(0, 0, 0, 0.1)"
                      : "none",
                  transition: "all 0.2s ease",
                }}
              >
                Available
              </button>
              <button
                onClick={() => setFilterStatus("occupied")}
                style={{
                  padding: "6px 16px",
                  background:
                    filterStatus === "occupied" ? "#FFFFFF" : "transparent",
                  color: filterStatus === "occupied" ? "#DC2626" : "#64748B",
                  border: "none",
                  borderRadius: "6px",
                  fontSize: "13px",
                  fontWeight: "600",
                  cursor: "pointer",
                  boxShadow:
                    filterStatus === "occupied"
                      ? "0 1px 3px rgba(0, 0, 0, 0.1)"
                      : "none",
                  transition: "all 0.2s ease",
                }}
              >
                Busy
              </button>
              <button
                onClick={() => setFilterStatus("disabled")}
                style={{
                  padding: "6px 16px",
                  background:
                    filterStatus === "disabled" ? "#FFFFFF" : "transparent",
                  color: filterStatus === "disabled" ? "#64748B" : "#64748B",
                  border: "none",
                  borderRadius: "6px",
                  fontSize: "13px",
                  fontWeight: "600",
                  cursor: "pointer",
                  boxShadow:
                    filterStatus === "disabled"
                      ? "0 1px 3px rgba(0, 0, 0, 0.1)"
                      : "none",
                  transition: "all 0.2s ease",
                }}
              >
                Disabled
              </button>
            </div>
          </div>
        </div>

        {loading ? (
          <div
            style={{ textAlign: "center", padding: "40px", color: "#64748B" }}
          >
            Loading slots...
          </div>
        ) : filteredSlots.length === 0 ? (
          <div
            style={{ textAlign: "center", padding: "40px", color: "#64748B" }}
          >
            No slots found
          </div>
        ) : (
          <>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
                gap: "20px",
                marginBottom: "24px",
              }}
            >
              {paginatedSlots.map((slot) => (
                <SlotCard
                  key={slot.slot_id}
                  slot={slot}
                  onToggle={handleToggleSlot}
                  onOpenDelete={handleOpenDelete}
                  bulkMode={bulkMode}
                  isSelected={selectedSlots.includes(slot.slot_id)}
                  onSelect={handleSelectSlot}
                />
              ))}
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  alignItems: "center",
                  gap: "8px",
                  marginTop: "24px",
                }}
              >
                <button
                  onClick={() =>
                    setCurrentPage((prev) => Math.max(1, prev - 1))
                  }
                  disabled={currentPage === 1}
                  style={{
                    padding: "8px 16px",
                    background: currentPage === 1 ? "#F8FAFC" : "#FFFFFF",
                    color: currentPage === 1 ? "#94A3B8" : "#475569",
                    border: "1px solid #E2E8F0",
                    borderRadius: "8px",
                    fontSize: "14px",
                    fontWeight: "500",
                    cursor: currentPage === 1 ? "not-allowed" : "pointer",
                  }}
                >
                  Previous
                </button>

                <div style={{ display: "flex", gap: "4px" }}>
                  {[...Array(totalPages)].map((_, index) => {
                    const page = index + 1;
                    // Show first page, last page, current page, and pages around current
                    if (
                      page === 1 ||
                      page === totalPages ||
                      (page >= currentPage - 1 && page <= currentPage + 1)
                    ) {
                      return (
                        <button
                          key={page}
                          onClick={() => setCurrentPage(page)}
                          style={{
                            padding: "8px 12px",
                            background:
                              currentPage === page ? "#3B82F6" : "#FFFFFF",
                            color: currentPage === page ? "#FFFFFF" : "#475569",
                            border: `1px solid ${currentPage === page ? "#3B82F6" : "#E2E8F0"}`,
                            borderRadius: "8px",
                            fontSize: "14px",
                            fontWeight: "600",
                            cursor: "pointer",
                            minWidth: "40px",
                          }}
                        >
                          {page}
                        </button>
                      );
                    } else if (
                      page === currentPage - 2 ||
                      page === currentPage + 2
                    ) {
                      return (
                        <span
                          key={page}
                          style={{ padding: "8px 4px", color: "#94A3B8" }}
                        >
                          ...
                        </span>
                      );
                    }
                    return null;
                  })}
                </div>

                <button
                  onClick={() =>
                    setCurrentPage((prev) => Math.min(totalPages, prev + 1))
                  }
                  disabled={currentPage === totalPages}
                  style={{
                    padding: "8px 16px",
                    background:
                      currentPage === totalPages ? "#F8FAFC" : "#FFFFFF",
                    color: currentPage === totalPages ? "#94A3B8" : "#475569",
                    border: "1px solid #E2E8F0",
                    borderRadius: "8px",
                    fontSize: "14px",
                    fontWeight: "500",
                    cursor:
                      currentPage === totalPages ? "not-allowed" : "pointer",
                  }}
                >
                  Next
                </button>

                <span
                  style={{
                    marginLeft: "16px",
                    fontSize: "14px",
                    color: "#64748B",
                  }}
                >
                  Showing {startIndex + 1}-
                  {Math.min(endIndex, filteredSlots.length)} of{" "}
                  {filteredSlots.length}
                </span>
              </div>
            )}
          </>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      {deleteConfirm && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10000,
          }}
        >
          <div
            style={{
              background: "#FFFFFF",
              borderRadius: "12px",
              padding: "24px",
              maxWidth: "400px",
              boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
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
                <AlertTriangle size={24} color="#EF4444" />
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
                  Delete Slot
                </h3>
              </div>
            </div>
            <p
              style={{
                margin: "0 0 24px 0",
                fontSize: "14px",
                color: "#374151",
              }}
            >
              Are you sure you want to delete this slot? This action cannot be
              undone.
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
                  padding: "10px 16px",
                  background: "#FFFFFF",
                  border: "1px solid #D1D5DB",
                  borderRadius: "6px",
                  fontSize: "14px",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteSlot(deleteConfirm)}
                style={{
                  padding: "10px 16px",
                  background: "#EF4444",
                  color: "#FFFFFF",
                  border: "none",
                  borderRadius: "6px",
                  fontSize: "14px",
                  cursor: "pointer",
                }}
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Bulk Delete Confirmation Dialog */}
      {bulkDeleteConfirm && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10000,
          }}
        >
          <div
            style={{
              background: "#FFFFFF",
              borderRadius: "12px",
              padding: "24px",
              maxWidth: "440px",
              boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
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
                <AlertTriangle size={24} color="#EF4444" />
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
                  Delete Multiple Slots
                </h3>
              </div>
            </div>
            <p
              style={{
                margin: "0 0 24px 0",
                fontSize: "14px",
                color: "#374151",
                lineHeight: "1.6",
              }}
            >
              Are you sure you want to delete{" "}
              <strong>{bulkDeleteConfirm} slot(s)</strong>? This action cannot
              be undone.
            </p>
            <div
              style={{
                display: "flex",
                gap: "12px",
                justifyContent: "flex-end",
              }}
            >
              <button
                onClick={() => setBulkDeleteConfirm(null)}
                style={{
                  padding: "10px 16px",
                  background: "#FFFFFF",
                  border: "1px solid #D1D5DB",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmBulkDelete}
                style={{
                  padding: "10px 16px",
                  background: "#EF4444",
                  color: "#FFFFFF",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "600",
                  cursor: "pointer",
                }}
              >
                Delete {bulkDeleteConfirm} Slots
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Force Reset Confirmation Dialog */}
      {forceResetConfirm && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 10000,
          }}
        >
          <div
            style={{
              background: "#FFFFFF",
              borderRadius: "12px",
              padding: "24px",
              maxWidth: "480px",
              boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
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
                <AlertTriangle size={24} color="#EF4444" />
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
                  Force Reset All Slots
                </h3>
              </div>
            </div>
            <div
              style={{
                margin: "0 0 24px 0",
                fontSize: "14px",
                color: "#374151",
                lineHeight: "1.6",
              }}
            >
              {slots.filter((s) => s.status === "occupied").length > 0 ? (
                <>
                  <p
                    style={{
                      margin: "0 0 12px 0",
                      fontWeight: "600",
                      color: "#DC2626",
                    }}
                  >
                    ⚠️ WARNING: This will:
                  </p>
                  <ul style={{ margin: "0 0 0 20px", padding: 0 }}>
                    <li style={{ marginBottom: "6px" }}>
                      Cancel{" "}
                      <strong>
                        {slots.filter((s) => s.status === "occupied").length}{" "}
                        active booking(s)
                      </strong>
                    </li>
                    <li style={{ marginBottom: "6px" }}>
                      Reset ALL slots to available
                    </li>
                    <li>Allow switching to AI mode</li>
                  </ul>
                  <p style={{ margin: "12px 0 0 0", fontWeight: "600" }}>
                    This action cannot be undone. Continue?
                  </p>
                </>
              ) : (
                <p style={{ margin: 0 }}>
                  This will reset all slots to available. Continue?
                </p>
              )}
            </div>
            <div
              style={{
                display: "flex",
                gap: "12px",
                justifyContent: "flex-end",
              }}
            >
              <button
                onClick={() => setForceResetConfirm(false)}
                style={{
                  padding: "10px 16px",
                  background: "#FFFFFF",
                  border: "1px solid #D1D5DB",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "500",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={confirmForceReset}
                style={{
                  padding: "10px 16px",
                  background: "#EF4444",
                  color: "#FFFFFF",
                  border: "none",
                  borderRadius: "8px",
                  fontSize: "14px",
                  fontWeight: "600",
                  cursor: "pointer",
                }}
              >
                Force Reset
              </button>
            </div>
          </div>
        </div>
      )}

      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
          }
        `}
      </style>
    </div>
  );
}

function safeStringify(obj) {
  try {
    return JSON.stringify(obj);
  } catch {
    return "";
  }
}

export default SlotManagement;
