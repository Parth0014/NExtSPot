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

  const groupByDate = () => {
    if (!scheduleData?.schedule) return {};

    const grouped = {};
    scheduleData.schedule.forEach((item) => {
      const date = item.booking_date.split("T")[0]; // Get YYYY-MM-DD
      if (!grouped[date]) {
        grouped[date] = [];
      }
      grouped[date].push(item);
    });
    return grouped;
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

  const groupedSchedule = groupByDate();
  const spot = scheduleData?.spot;

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
          {Object.keys(groupedSchedule).length === 0 ? (
            <div className="no-bookings">
              <p>🎉 No bookings scheduled for the selected period!</p>
              <p>All slots are available for booking.</p>
            </div>
          ) : (
            Object.entries(groupedSchedule).map(([date, bookings]) => (
              <div key={date} className="date-group">
                <div className="date-header">
                  <h4>{formatDate(date)}</h4>
                  <span className="date-full">{date}</span>
                </div>

                <div className="time-slots">
                  {bookings.map((booking, idx) => {
                    const availableSlots =
                      spot.active_slots - booking.slots_booked;
                    const availabilityColor = getAvailabilityColor(
                      booking.slots_booked,
                      spot.active_slots,
                    );

                    return (
                      <div key={idx} className="time-slot-card">
                        <div className="time-slot-header">
                          <span className="time-range">
                            {booking.booking_hour}:00 -{" "}
                            {booking.booking_hour + 1}:00
                          </span>
                          <span
                            className="availability-badge"
                            style={{ backgroundColor: availabilityColor }}
                          >
                            {booking.slots_booked}/{spot.active_slots} booked
                          </span>
                        </div>

                        <div className="slot-details">
                          <p className="available-count">
                            {availableSlots > 0 ? (
                              <>
                                ✅{" "}
                                <strong>
                                  {availableSlots} slots available
                                </strong>
                              </>
                            ) : (
                              <>
                                ❌ <strong>Fully booked</strong>
                              </>
                            )}
                          </p>
                          {booking.booking_details && (
                            <details className="booking-details-expand">
                              <summary>View booking details</summary>
                              <div className="booking-details-list">
                                {booking.booking_details
                                  .split("; ")
                                  .map((detail, i) => (
                                    <div
                                      key={i}
                                      className="booking-detail-item"
                                    >
                                      • {detail}
                                    </div>
                                  ))}
                              </div>
                            </details>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
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
