# Scheduled Bookings System - How It Works

## ✅ Fixed Issues

### Problem Solved:

Your concern was: **"If a spot has 5 slots and 5 users book for tomorrow at the same time, will the 6th user be blocked or will the system break?"**

### Solution Implemented:

## 1. Smart Slot Selection (bookings.js)

When a user tries to book a slot, the system now:

- ✅ Checks if the slot is currently available (`is_available = 1`)
- ✅ **NEW:** Checks if the slot has any conflicting future bookings
- ✅ Only assigns slots that are free for the requested time period

```sql
-- This query finds slots with NO overlapping bookings
SELECT ps.id, ps.slot_number
FROM parking_slots ps
WHERE ps.parking_spot_id = ?
  AND ps.is_active = 1
  AND ps.is_available = 1
  AND NOT EXISTS (
    SELECT 1 FROM bookings b
    WHERE b.slot_id = ps.id
      AND b.status = 'active'
      AND (
        -- Checks for any time overlap
        (b.start_at < ? AND b.expires_at > ?)
        OR (b.start_at >= ? AND b.start_at < ?)
      )
  )
```

## 2. Background Job (scheduledBookingsJob.js)

A background job runs **every 60 seconds** to:

### A) Activate Scheduled Bookings

When a future booking reaches its start time:

- Marks the slot as unavailable (`is_available = 0`)
- Updates parking spot statistics (occupied +1, available -1)
- Updates the availability table

### B) Expire Finished Bookings

When a booking passes its expiration time:

- Changes booking status to 'expired'
- Marks the slot as available again (`is_available = 1`)
- Updates parking spot statistics (occupied -1, available +1)
- Frees up the slot for new bookings

## How It Works in Practice

### Scenario: Spot has 5 slots, 6 users want tomorrow 10 AM - 11 AM

**Timeline:**

| User   | Action                  | Result                                                                    |
| ------ | ----------------------- | ------------------------------------------------------------------------- |
| User 1 | Books tomorrow 10-11 AM | ✅ Gets Slot 1 (Slot 1 still shows available NOW)                         |
| User 2 | Books tomorrow 10-11 AM | ✅ Gets Slot 2 (Slot 2 still shows available NOW)                         |
| User 3 | Books tomorrow 10-11 AM | ✅ Gets Slot 3                                                            |
| User 4 | Books tomorrow 10-11 AM | ✅ Gets Slot 4                                                            |
| User 5 | Books tomorrow 10-11 AM | ✅ Gets Slot 5                                                            |
| User 6 | Books tomorrow 10-11 AM | ❌ **BLOCKED** - "No available slots found for the requested time period" |

**Tomorrow at 10:00 AM:**

- Background job runs
- Detects 5 bookings have reached their start time
- Marks Slots 1-5 as unavailable
- Users can now see these slots are occupied

**Tomorrow at 11:00 AM:**

- Background job runs
- Detects 5 bookings have expired
- Marks Slots 1-5 as available again
- New users can now book these slots

## Key Features

### For Scheduled (Future) Bookings:

- ✅ Slot remains visible as "available" until the booking start time
- ✅ Slot cannot be double-booked for overlapping time periods
- ✅ Automatically becomes "occupied" when the scheduled time arrives
- ✅ Automatically becomes "available" again when the booking expires

### For Immediate Bookings:

- ✅ Slot is marked unavailable immediately upon booking
- ✅ Statistics updated instantly
- ✅ Automatically freed when booking expires

## Starting the System

The background job starts automatically when you run:

```bash
cd backend
node server.js
```

You'll see:

```
⏰ Starting scheduled bookings maintenance job...
✅ Background job started - checking every 60 seconds
```

## Testing the System

1. **Create a future booking:**

   ```bash
   POST /api/bookings/reserve/:parking_id
   Body: {
     "start_at": "2026-02-02T10:00:00",
     "duration_minutes": 60
   }
   ```

2. **Try to book the same slot for overlapping time:**

   ```bash
   POST /api/bookings/reserve/:parking_id
   Body: {
     "start_at": "2026-02-02T10:30:00",
     "duration_minutes": 60
   }
   ```

   Result: ❌ Different slot assigned or error if all slots occupied

3. **Check slot status before booking starts:**
   - Slot shows `is_available = 1` (available)
   - But has a future booking in the database

4. **Check slot status after booking starts:**
   - Background job automatically marks it `is_available = 0`
   - Slot shows as occupied

## Summary

✅ **System will NOT break**
✅ 6th user will be blocked with clear error message
✅ Slots intelligently managed for present and future
✅ Automatic activation/expiration every minute
✅ No manual intervention required
