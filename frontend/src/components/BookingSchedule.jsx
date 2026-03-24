import React, { useState, useEffect } from "react";
import axios from "axios";

const BookingSchedule = ({ parkingSpotId, onClose }) => {
  const [scheduleData, setScheduleData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [daysToShow, setDaysToShow] = useState(7);

  const fetchSchedule = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(
        `http://localhost:5000/api/bookings/spot/${parkingSpotId}/schedule?days=${daysToShow}`,
      );
      setScheduleData(response.data);
    } catch (err) {
      setError(err.response?.data?.message || "Failed to load schedule");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (parkingSpotId) {
      fetchSchedule();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parkingSpotId, daysToShow]);

  const toDateKey = (value) => {
    if (!value) return "";
    if (typeof value === "string") {
      return value.includes("T") ? value.split("T")[0] : value;
    }
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return "";
    return d.toISOString().split("T")[0];
  };

  const toStartDateTime = (bookingDate, bookingHour) => {
    const dateKey = toDateKey(bookingDate);
    const hour = Number(bookingHour);
    if (!dateKey || Number.isNaN(hour)) return null;
    const safeHour = String(hour).padStart(2, "0");
    return new Date(`${dateKey}T${safeHour}:00:00`);
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    if (date.toDateString() === today.toDateString()) {
      return "Today";
    } else if (date.toDateString() === tomorrow.toDateString()) {
      return "Tomorrow";
    } else {
      return date.toLocaleDateString("en-US", {
        weekday: "short",
        month: "short",
        day: "numeric",
      });
    }
  };

  const getAvailabilityColor = (slotsBooked, totalSlots) => {
    const percentage = (slotsBooked / totalSlots) * 100;
    if (percentage >= 100) return "#ef4444"; // Red - Full
    if (percentage >= 75) return "#f59e0b"; // Orange - Almost full
    if (percentage >= 50) return "#eab308"; // Yellow - Half full
    return "#10b981"; // Green - Available
  };

  if (loading) {
    return (
      <div className="booking-schedule-overlay">
        <div className="booking-schedule-modal">
          <div className="schedule-loading">Loading schedule...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="booking-schedule-overlay">
        <div className="booking-schedule-modal">
          <div className="schedule-error">{error}</div>
          <button onClick={onClose} className="close-btn">
            Close
          </button>
        </div>
      </div>
    );
  }

  const spot = scheduleData?.spot;
  const now = new Date();

  const rows = (scheduleData?.schedule || []).map((item) => {
    const startAt = toStartDateTime(item.booking_date, item.booking_hour);
    const dateKey = toDateKey(item.booking_date);
    const hour = Number(item.booking_hour) || 0;
    const endHour = (hour + 1) % 24;
    const availableSlots = Math.max(
      0,
      (spot?.active_slots || 0) - Number(item.slots_booked || 0),
    );
    return {
      ...item,
      dateKey,
      hour,
      endHour,
      startAt,
      availableSlots,
    };
  });

  const currentRows = rows.filter((row) => {
    if (!row.startAt) return false;
    const endAt = new Date(row.startAt.getTime() + 60 * 60 * 1000);
    return row.startAt <= now && now < endAt;
  });

  const futureRows = rows.filter((row) => row.startAt && row.startAt > now);

  const renderTable = (tableRows) => (
    <div className="schedule-table-wrap">
      <table className="schedule-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Time</th>
            <th>Booked</th>
            <th>Available</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {tableRows.map((row, idx) => {
            const availabilityColor = getAvailabilityColor(
              Number(row.slots_booked || 0),
              spot?.active_slots || 1,
            );
            return (
              <tr key={`${row.dateKey}-${row.hour}-${idx}`}>
                <td>{formatDate(row.dateKey)}</td>
                <td>
                  {String(row.hour).padStart(2, "0")}:00 -{" "}
                  {String(row.endHour).padStart(2, "0")}:00
                </td>
                <td>
                  {row.slots_booked}/{spot?.active_slots || 0}
                </td>
                <td>
                  <span
                    className="availability-badge"
                    style={{ backgroundColor: availabilityColor }}
                  >
                    {row.availableSlots}
                  </span>
                </td>
                <td>
                  {row.booking_details ? (
                    <details className="booking-details-expand">
                      <summary>View</summary>
                      <div className="booking-details-list">
                        {row.booking_details.split("; ").map((detail, i) => (
                          <div key={i} className="booking-detail-item">
                            • {detail}
                          </div>
                        ))}
                      </div>
                    </details>
                  ) : (
                    <span>-</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className="booking-schedule-overlay" onClick={onClose}>
      <div
        className="booking-schedule-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="schedule-header">
          <h2>📅 Booking Schedule</h2>
          <button onClick={onClose} className="close-btn-x">
            ✕
          </button>
        </div>

        {spot && (
          <div className="spot-info-header">
            <h3>{spot.name}</h3>
            <div className="spot-stats">
              <span className="stat">
                <strong>Total Slots:</strong> {spot.total_slots}
              </span>
              <span className="stat">
                <strong>Currently Available:</strong> {spot.available_slots}
              </span>
              <span className="stat">
                <strong>Price:</strong> ₹{spot.price}/hr
              </span>
            </div>
          </div>
        )}

        <div className="days-filter">
          <label>Show bookings for:</label>
          <select
            value={daysToShow}
            onChange={(e) => setDaysToShow(Number(e.target.value))}
          >
            <option value={3}>Next 3 days</option>
            <option value={7}>Next 7 days</option>
            <option value={14}>Next 14 days</option>
            <option value={30}>Next 30 days</option>
          </select>
        </div>

        <div className="schedule-content">
          {rows.length === 0 ? (
            <div className="no-bookings">
              <p>🎉 No bookings scheduled for the selected period!</p>
              <p>All slots are available for booking.</p>
            </div>
          ) : (
            <>
              <div className="table-section">
                <div className="date-header">
                  <h4>Current Bookings</h4>
                  <span className="date-full">Active now</span>
                </div>
                {currentRows.length > 0 ? (
                  renderTable(currentRows)
                ) : (
                  <p className="table-empty">No current bookings right now.</p>
                )}
              </div>

              <div className="table-section">
                <div className="date-header">
                  <h4>Future Bookings</h4>
                  <span className="date-full">Upcoming schedule</span>
                </div>
                {futureRows.length > 0 ? (
                  renderTable(futureRows)
                ) : (
                  <p className="table-empty">
                    No future bookings in this range.
                  </p>
                )}
              </div>
            </>
          )}
        </div>

        <div className="schedule-legend">
          <h5>Legend:</h5>
          <div className="legend-items">
            <span className="legend-item">
              <span
                className="legend-color"
                style={{ background: "#10b981" }}
              ></span>
              Available (&lt;50% booked)
            </span>
            <span className="legend-item">
              <span
                className="legend-color"
                style={{ background: "#eab308" }}
              ></span>
              Moderate (50-75% booked)
            </span>
            <span className="legend-item">
              <span
                className="legend-color"
                style={{ background: "#f59e0b" }}
              ></span>
              Almost Full (75-99% booked)
            </span>
            <span className="legend-item">
              <span
                className="legend-color"
                style={{ background: "#ef4444" }}
              ></span>
              Fully Booked
            </span>
          </div>
        </div>

        <div className="schedule-footer">
          <button onClick={onClose} className="close-btn">
            Close
          </button>
        </div>
      </div>

      <style jsx>{`
        .booking-schedule-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 20px;
        }

        .booking-schedule-modal {
          background: white;
          border-radius: 12px;
          max-width: 800px;
          width: 100%;
          max-height: 90vh;
          overflow-y: auto;
          box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }

        .schedule-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 20px;
          border-bottom: 2px solid #e5e7eb;
          position: sticky;
          top: 0;
          background: white;
          z-index: 10;
        }

        .schedule-header h2 {
          margin: 0;
          color: #1f2937;
        }

        .close-btn-x {
          background: none;
          border: none;
          font-size: 24px;
          cursor: pointer;
          color: #6b7280;
          padding: 0;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 6px;
        }

        .close-btn-x:hover {
          background: #f3f4f6;
          color: #1f2937;
        }

        .spot-info-header {
          padding: 20px;
          background: #f9fafb;
          border-bottom: 1px solid #e5e7eb;
        }

        .spot-info-header h3 {
          margin: 0 0 10px 0;
          color: #1f2937;
        }

        .spot-stats {
          display: flex;
          gap: 20px;
          flex-wrap: wrap;
        }

        .spot-stats .stat {
          font-size: 14px;
          color: #6b7280;
        }

        .days-filter {
          padding: 15px 20px;
          background: #fffbeb;
          border-bottom: 1px solid #fef3c7;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .days-filter label {
          font-weight: 500;
          color: #78350f;
        }

        .days-filter select {
          padding: 6px 12px;
          border: 1px solid #fbbf24;
          border-radius: 6px;
          font-size: 14px;
          cursor: pointer;
        }

        .schedule-content {
          padding: 20px;
        }

        .no-bookings {
          text-align: center;
          padding: 40px 20px;
          color: #6b7280;
        }

        .no-bookings p:first-child {
          font-size: 18px;
          font-weight: 500;
          margin-bottom: 10px;
        }

        .date-group {
          margin-bottom: 30px;
        }

        .date-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 15px;
          padding-bottom: 10px;
          border-bottom: 2px solid #e5e7eb;
        }

        .date-header h4 {
          margin: 0;
          color: #1f2937;
          font-size: 18px;
        }

        .date-full {
          font-size: 14px;
          color: #6b7280;
        }

        .time-slots {
          display: grid;
          gap: 12px;
        }

        .table-section {
          margin-bottom: 22px;
        }

        .table-empty {
          margin: 8px 0 0;
          color: #6b7280;
          font-size: 14px;
        }

        .schedule-table-wrap {
          overflow-x: auto;
          border: 1px solid #e5e7eb;
          border-radius: 10px;
          background: #fff;
        }

        .schedule-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 13px;
        }

        .schedule-table th,
        .schedule-table td {
          padding: 10px 12px;
          text-align: left;
          border-bottom: 1px solid #f1f5f9;
          vertical-align: top;
        }

        .schedule-table thead th {
          background: #f8fafc;
          color: #334155;
          font-weight: 700;
          white-space: nowrap;
        }

        .schedule-table tbody tr:last-child td {
          border-bottom: none;
        }

        .time-slot-card {
          border: 1px solid #e5e7eb;
          border-radius: 8px;
          padding: 15px;
          background: #fafafa;
        }

        .time-slot-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .time-range {
          font-weight: 600;
          color: #1f2937;
          font-size: 16px;
        }

        .availability-badge {
          padding: 4px 12px;
          border-radius: 20px;
          color: white;
          font-size: 13px;
          font-weight: 500;
        }

        .slot-details {
          margin-top: 10px;
        }

        .available-count {
          margin: 0 0 8px 0;
          font-size: 14px;
          color: #374151;
        }

        .booking-details-expand {
          margin-top: 8px;
        }

        .booking-details-expand summary {
          cursor: pointer;
          color: #2563eb;
          font-size: 13px;
          user-select: none;
        }

        .booking-details-expand summary:hover {
          text-decoration: underline;
        }

        .booking-details-list {
          margin-top: 8px;
          padding: 10px;
          background: white;
          border-radius: 6px;
          border: 1px solid #e5e7eb;
        }

        .booking-detail-item {
          font-size: 13px;
          color: #6b7280;
          padding: 4px 0;
        }

        .schedule-legend {
          padding: 15px 20px;
          background: #f9fafb;
          border-top: 1px solid #e5e7eb;
        }

        .schedule-legend h5 {
          margin: 0 0 10px 0;
          font-size: 14px;
          color: #6b7280;
        }

        .legend-items {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          color: #6b7280;
        }

        .legend-color {
          width: 16px;
          height: 16px;
          border-radius: 3px;
          display: inline-block;
        }

        .schedule-footer {
          padding: 15px 20px;
          border-top: 1px solid #e5e7eb;
          text-align: right;
        }

        .close-btn {
          padding: 10px 24px;
          background: #3b82f6;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
        }

        .close-btn:hover {
          background: #2563eb;
        }

        .schedule-loading,
        .schedule-error {
          padding: 40px;
          text-align: center;
          color: #6b7280;
        }

        .schedule-error {
          color: #ef4444;
        }
      `}</style>
    </div>
  );
};

export default BookingSchedule;
