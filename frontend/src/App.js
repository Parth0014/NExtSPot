import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";

// Existing auth pages
import Home from "./pages/Home";
import Login from "./pages/Login";
import OwnerLogin from "./pages/OwnerLogin";
import Register from "./pages/Register";

// New parking management pages
import UserMap from "./pages/user/UserMap";
import UserReservations from "./pages/user/UserReservations";
import UserReviews from "./pages/user/UserReviews";
import OwnerDashboard from "./pages/owner/OwnerDashboard";
import OwnerBookings from "./pages/owner/OwnerBookings";
import OwnerReviewsPage from "./pages/owner/OwnerReviewsPage";
import SlotManagement from "./components/SlotManagement";
import AICameraSetup from "./components/AICameraSetup";

function App() {
  return (
    <Routes>
      {/* Default/Auth pages */}
      <Route path="/" element={<Home />} />
      <Route path="/home" element={<Home />} />
      <Route path="/login" element={<Login />} />
      <Route path="/owner-login" element={<OwnerLogin />} />
      <Route path="/register" element={<Register />} />

      {/* User pages - accessible after user login */}
      <Route path="/user/dashboard" element={<UserMap />} />
      <Route path="/user/reservations" element={<UserReservations />} />
      <Route path="/user/reviews" element={<UserReviews />} />

      {/* Owner pages - accessible after owner login */}
      <Route path="/owner/dashboard" element={<OwnerDashboard />} />
      <Route path="/owner/bookings" element={<OwnerBookings />} />
      <Route path="/owner/reviews" element={<OwnerReviewsPage />} />
      <Route path="/owner/slots/:spotId" element={<SlotManagement />} />
      <Route path="/owner/ai-setup/:spotId" element={<AICameraSetup />} />

      {/* Redirect unknown routes to home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
