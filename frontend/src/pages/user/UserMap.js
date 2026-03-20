import React, { useState, useEffect, useRef } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMap,
  ZoomControl,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import BookingSchedule from "../../components/BookingSchedule";

// Component to handle routing
function RoutingController({
  userLocation,
  destination,
  onRouteFound,
  onRouteClear,
}) {
  const map = useMap();
  const routingControlRef = useRef(null);
  const isActiveRef = useRef(false);
  const currentDestinationRef = useRef(null);

  const clearRoute = () => {
    if (routingControlRef.current && map) {
      try {
        if (routingControlRef.current.getContainer) {
          const container = routingControlRef.current.getContainer();
          if (container && container.style) {
            container.style.display = "none";
          }
        }
        if (routingControlRef.current.setWaypoints) {
          routingControlRef.current.setWaypoints([]);
        }
      } catch (error) {
        // Silently handle errors
      }
    }
    isActiveRef.current = false;
    currentDestinationRef.current = null;
  };

  useEffect(() => {
    if (!destination || !userLocation || !map) {
      clearRoute();
      onRouteClear();
      return;
    }

    const destinationKey = `${destination.lat}-${destination.lng}`;
    if (
      currentDestinationRef.current === destinationKey &&
      isActiveRef.current
    ) {
      return;
    }

    currentDestinationRef.current = destinationKey;

    if (!routingControlRef.current) {
      if (!window.L || !window.L.Routing) {
        const script = document.createElement("script");
        script.src =
          "https://cdnjs.cloudflare.com/ajax/libs/leaflet-routing-machine/3.2.12/leaflet-routing-machine.min.js";

        script.onload = () => {
          const link = document.createElement("link");
          link.rel = "stylesheet";
          link.href =
            "https://cdnjs.cloudflare.com/ajax/libs/leaflet-routing-machine/3.2.12/leaflet-routing-machine.css";
          document.head.appendChild(link);
          setTimeout(createRoutingControl, 500);
        };

        script.onerror = () => onRouteFound("Routing unavailable");
        document.head.appendChild(script);
      } else {
        setTimeout(createRoutingControl, 200);
      }
    } else {
      updateWaypoints();
    }

    function createRoutingControl() {
      if (!window.L || !window.L.Routing || !map || routingControlRef.current)
        return;

      try {
        routingControlRef.current = window.L.Routing.control({
          waypoints: [
            L.latLng(userLocation[0], userLocation[1]),
            L.latLng(destination.lat, destination.lng),
          ],
          routeWhileDragging: false,
          addWaypoints: false,
          createMarker: () => null,
          lineOptions: {
            styles: [{ color: "#4F86C6", weight: 4, opacity: 0.8 }],
          },
          show: false,
          collapsible: true,
          router: window.L.Routing.osrmv1({
            serviceUrl: "https://router.project-osrm.org/route/v1",
            profile: "driving",
          }),
        });

        routingControlRef.current.on("routesfound", handleRouteFound);
        routingControlRef.current.on("routingerror", handleRouteError);

        if (map && map.addControl) {
          map.addControl(routingControlRef.current);
          isActiveRef.current = true;

          setTimeout(() => {
            if (
              routingControlRef.current &&
              routingControlRef.current.getContainer
            ) {
              const container = routingControlRef.current.getContainer();
              if (container && container.style) {
                container.style.display = "block";
              }
            }
          }, 100);
        }
      } catch (error) {
        console.error("Failed to create routing control:", error);
        onRouteFound("Routing failed");
      }
    }

    function updateWaypoints() {
      if (routingControlRef.current && routingControlRef.current.setWaypoints) {
        try {
          routingControlRef.current.setWaypoints([
            L.latLng(userLocation[0], userLocation[1]),
            L.latLng(destination.lat, destination.lng),
          ]);

          if (routingControlRef.current.getContainer) {
            const container = routingControlRef.current.getContainer();
            if (container && container.style) {
              container.style.display = "block";
            }
          }

          isActiveRef.current = true;
        } catch (error) {
          console.warn("Failed to update waypoints:", error);
          onRouteFound("Route update failed");
        }
      }
    }

    function handleRouteFound(e) {
      try {
        const routes = e.routes;
        if (routes && routes[0] && routes[0].summary) {
          const summary = routes[0].summary;
          const distance = (summary.totalDistance / 1000).toFixed(1);
          const time = Math.round(summary.totalTime / 60);
          onRouteFound(`${distance} km • ${time} min`);
        } else {
          onRouteFound("Route calculated");
        }
      } catch (err) {
        onRouteFound("Route found");
      }
    }
    // (MapResizeHandler removed from here and defined at top-level below)

    function handleRouteError(e) {
      console.warn("Routing error:", e);
      onRouteFound("Route unavailable");
    }

    return () => {
      if (!destination) {
        clearRoute();
      }
    };
  }, [userLocation, destination, map, onRouteFound, onRouteClear]);

  useEffect(() => {
    if (!destination && isActiveRef.current) {
      clearRoute();
      onRouteClear();
    }
  }, [destination, onRouteClear]);

  return null;
}

// Component to force map resize when sidebar toggles
function MapResizeHandler({ showSpotsList }) {
  const map = useMap();

  useEffect(() => {
    if (!map || !map.invalidateSize) return;

    // Immediate resize
    const resize = () => {
      try {
        map.invalidateSize({ animate: true });
      } catch (err) {
        // ignore
      }
    };

    resize();

    // Use requestAnimationFrame for smoother animation
    let frameCount = 0;
    const maxFrames = 30; // ~500ms at 60fps

    const animate = () => {
      if (frameCount < maxFrames) {
        resize();
        frameCount++;
        requestAnimationFrame(animate);
      }
    };

    const animationFrame = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, [showSpotsList, map]);

  return null;
}

// Component to handle map updates
function MapController({
  userLocation,
  initialSpots,
  hasInitialized,
  hasActiveRoute,
  shouldClosePopups,
  onPopupsClosed,
}) {
  const map = useMap();
  const initialViewSetRef = useRef(false);

  useEffect(() => {
    if (!hasInitialized || hasActiveRoute || initialViewSetRef.current || !map)
      return;

    setTimeout(() => {
      try {
        if (userLocation && initialSpots.length > 0) {
          const bounds = L.latLngBounds([
            userLocation,
            ...initialSpots.map((spot) => [spot.latitude, spot.longitude]),
          ]);
          map.fitBounds(bounds, { padding: [40, 40] });
          initialViewSetRef.current = true;
        } else if (userLocation) {
          map.setView(userLocation, 15);
          initialViewSetRef.current = true;
        }
      } catch (error) {
        console.warn("Error setting initial map view:", error);
      }
    }, 100);
  }, [map, hasInitialized, hasActiveRoute, userLocation, initialSpots]);

  useEffect(() => {
    if (!hasActiveRoute) {
      initialViewSetRef.current = false;
    }
  }, [hasActiveRoute]);

  useEffect(() => {
    if (shouldClosePopups && onPopupsClosed && map) {
      try {
        map.closePopup();
        onPopupsClosed();
      } catch (error) {
        console.warn("Error closing popups:", error);
        onPopupsClosed();
      }
    }
  }, [map, shouldClosePopups, onPopupsClosed]);

  return null;
}

// Component to handle map view centering when spot is selected
function MapViewController({ center, zoom, onAnimationComplete }) {
  const map = useMap();
  const lastCenterRef = useRef(null);

  useEffect(() => {
    if (!map || !center) return;

    // Only animate if the center has actually changed
    const centerKey = `${center[0]}-${center[1]}`;
    if (lastCenterRef.current === centerKey) return;

    lastCenterRef.current = centerKey;

    try {
      map.flyTo(center, zoom || 16, {
        duration: 1.5,
        easeLinearity: 0.5,
      });

      // Clear the center after animation to prevent re-triggering
      setTimeout(() => {
        if (onAnimationComplete) {
          onAnimationComplete();
        }
      }, 1600); // Slightly longer than animation duration
    } catch (err) {
      console.warn("Error setting map view:", err);
    }
  }, [center, zoom, map, onAnimationComplete]);

  return null;
}

function UserMap() {
  const [spots, setSpots] = useState([]);
  const [message, setMessage] = useState("");
  const [userLocation, setUserLocation] = useState(null);
  const [locationError, setLocationError] = useState("");
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [routeDestination, setRouteDestination] = useState(null);
  const [routeDistance, setRouteDistance] = useState("");
  const [initialSpots, setInitialSpots] = useState([]);
  const [mapInitialized, setMapInitialized] = useState(false);
  const [selectedSpot, setSelectedSpot] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showSpotsList, setShowSpotsList] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [showBookingDialog, setShowBookingDialog] = useState(false);
  const [bookingSpotId, setBookingSpotId] = useState(null);
  // duration in minutes
  const [bookingDurationMinutes, setBookingDurationMinutes] = useState(60);
  const [bookingStartAt, setBookingStartAt] = useState("");
  const [showCustomDuration, setShowCustomDuration] = useState(false);
  const [shouldClosePopups, setShouldClosePopups] = useState(false);
  const [filters, setFilters] = useState({
    minPrice: 0,
    maxPrice: null,
    minDistance: 0,
    maxDistance: null,
    status: "all",
    sortBy: "distance",
  });
  const [showFilters, setShowFilters] = useState(false);
  const [reviewSummaries, setReviewSummaries] = useState({});
  const [canReviewSpots, setCanReviewSpots] = useState({});
  const [showReviewsModal, setShowReviewsModal] = useState(false);
  const [selectedSpotReviews, setSelectedSpotReviews] = useState([]);
  const [reviewsLoading, setReviewsLoading] = useState(false);
  const [reviewsPage, setReviewsPage] = useState(1);
  const [reviewsPagination, setReviewsPagination] = useState(null);
  const [selectedSpotForReviews, setSelectedSpotForReviews] = useState(null);
  const [expandedSpotId, setExpandedSpotId] = useState(null);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [scheduleSpotId, setScheduleSpotId] = useState(null);
  const [mapCenter, setMapCenter] = useState(null);
  const [mapZoom, setMapZoom] = useState(15);
  const refreshIntervalRef = useRef(null);
  const isRefreshingRef = useRef(false);
  const markerRefs = useRef({});
  const spotCardRefs = useRef({});
  const lastCenteredSpotId = useRef(null);

  // Helper function to safely format numbers
  const safeToFixed = (value, decimals = 2) => {
    const num = parseFloat(value);
    return isNaN(num) ? "0.00" : num.toFixed(decimals);
  };

  const getToken = () => localStorage.getItem("token");

  const handleLogout = () => {
    localStorage.removeItem("token");
    window.location.href = "/home";
  };

  const fetchSpots = async (isInitialLoad = false) => {
    const token = getToken();
    if (!token) {
      setMessage("Please log in to continue");
      return;
    }

    if (!isInitialLoad && (selectedRoute || routeDestination)) {
      return;
    }

    if (isInitialLoad) setIsLoading(true);
    isRefreshingRef.current = true;

    try {
      const res = await fetch("http://localhost:5000/api/spots/available", {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.message || "Failed to fetch spots");
      }

      const data = await res.json();
      const spotsData = Array.isArray(data) ? data : [];

      setSpots(spotsData);

      if (isInitialLoad) {
        setInitialSpots(spotsData);
        setTimeout(() => setMapInitialized(true), 500);
        spotsData.forEach((spot) => {
          fetchReviewSummary(spot.id);
          checkCanReview(spot.id);
        });
      }
      setMessage("");
    } catch (err) {
      if (isInitialLoad) {
        setMessage(err.message);
      }
    } finally {
      if (isInitialLoad) setIsLoading(false);
      isRefreshingRef.current = false;
    }
  };

  const getUserLocation = () => {
    if (!navigator.geolocation) {
      setLocationError("Location services not available");
      return;
    }

    setMessage("Locating...");

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        const newLocation = [latitude, longitude];

        if (
          !userLocation ||
          calculateDistance(
            userLocation[0],
            userLocation[1],
            latitude,
            longitude,
          ) > 0.01
        ) {
          setUserLocation(newLocation);
        }

        setMessage("");
        setLocationError("");

        // Pan map to user location
        setMapCenter(newLocation);
        setMapZoom(15);
      },
      (error) => {
        let errorMessage = "";
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage = "Location access denied";
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage = "Location unavailable";
            break;
          case error.TIMEOUT:
            errorMessage = "Location request timed out";
            break;
          default:
            errorMessage = "Location error occurred";
            break;
        }
        setLocationError(errorMessage);
        setMessage("");
      },
      {
        enableHighAccuracy: true,
        timeout: 8000,
        maximumAge: 300000,
      },
    );
  };

  const calculateDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371;
    const dLat = ((lat2 - lat1) * Math.PI) / 180;
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) *
        Math.cos((lat2 * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  const fetchReviewSummary = async (spotId) => {
    const token = getToken();
    if (!token || reviewSummaries[spotId]) return;

    try {
      const res = await fetch(
        `http://localhost:5000/reviews/spot/${spotId}/summary`,
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );

      if (res.ok) {
        const data = await res.json();
        const sanitizedData = {
          ...data,
          average_rating: parseFloat(data.average_rating) || 0,
          total_reviews: parseInt(data.total_reviews) || 0,
        };
        setReviewSummaries((prev) => ({
          ...prev,
          [spotId]: sanitizedData,
        }));
      }
    } catch (err) {
      console.error("Error fetching review summary:", err);
    }
  };

  const checkCanReview = async (spotId) => {
    const token = getToken();
    if (!token) return;

    try {
      const res = await fetch(
        `http://localhost:5000/reviews/can-review/${spotId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );

      if (res.ok) {
        const data = await res.json();
        setCanReviewSpots((prev) => ({
          ...prev,
          [spotId]: data,
        }));
      }
    } catch (err) {
      console.error("Error checking review eligibility:", err);
    }
  };

  const fetchSpotReviews = async (spotId, page = 1) => {
    const token = getToken();
    if (!token) return;

    setReviewsLoading(true);
    try {
      const res = await fetch(
        `http://localhost:5000/reviews/spot/${spotId}?page=${page}&limit=5`,
        { headers: { Authorization: `Bearer ${token}` } },
      );

      if (res.ok) {
        const data = await res.json();
        setSelectedSpotReviews(data.reviews || []);
        setReviewsPagination(data.pagination);
      }
    } catch (err) {
      console.error("Error fetching reviews:", err);
      setSelectedSpotReviews([]);
    } finally {
      setReviewsLoading(false);
    }
  };

  const openReviewsModal = (spot) => {
    setSelectedSpotForReviews(spot);
    setShowReviewsModal(true);
    setReviewsPage(1);
    fetchSpotReviews(spot.id, 1);
  };

  const closeReviewsModal = () => {
    setShowReviewsModal(false);
    setSelectedSpotReviews([]);
    setReviewsPagination(null);
    setSelectedSpotForReviews(null);
  };

  const handleReviewsPageChange = (newPage) => {
    if (selectedSpotForReviews) {
      setReviewsPage(newPage);
      fetchSpotReviews(selectedSpotForReviews.id, newPage);
    }
  };

  const voteOnReview = async (reviewId, isHelpful) => {
    const token = getToken();
    if (!token) return;

    try {
      const res = await fetch(
        `http://localhost:5000/reviews/${reviewId}/vote`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ is_helpful: isHelpful }),
        },
      );

      if (res.ok) {
        if (selectedSpotForReviews) {
          fetchSpotReviews(selectedSpotForReviews.id, reviewsPage);
        }
      }
    } catch (err) {
      console.error("Error voting on review:", err);
    }
  };

  const showRoute = (spotLat, spotLng, spotName) => {
    if (!userLocation) {
      setMessage("Location required for directions");
      return;
    }

    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current);
    }

    setRouteDestination({ lat: spotLat, lng: spotLng });
    setSelectedRoute(spotName);
    setMessage(`Getting directions to ${spotName}...`);
  };

  const handleRouteFound = (distanceTime) => {
    setRouteDistance(distanceTime);
    if (
      !distanceTime.includes("unavailable") &&
      !distanceTime.includes("error") &&
      !distanceTime.includes("failed")
    ) {
      setMessage("");
    }
  };

  const clearRoute = () => {
    setRouteDestination(null);
    setSelectedRoute(null);
    setRouteDistance("");
    setSelectedSpot(null);
    setMessage("");
    setShouldClosePopups(true);
    lastCenteredSpotId.current = null;

    if (!refreshIntervalRef.current) {
      refreshIntervalRef.current = setInterval(() => fetchSpots(false), 2000);
    }
  };

  const handlePopupsClosed = () => {
    setShouldClosePopups(false);
  };

  const handleRouteClear = () => {
    // Route cleared by RoutingController
  };

  const openBookingDialog = (id) => {
    setBookingSpotId(id);
    setBookingDurationMinutes(60);
    setBookingStartAt("");
    setShowCustomDuration(false);
    setShowBookingDialog(true);
  };

  const closeBookingDialog = () => {
    setShowBookingDialog(false);
    setBookingSpotId(null);
    setBookingDurationMinutes(60);
    setBookingStartAt("");
    setShowCustomDuration(false);
  };

  const confirmBooking = async () => {
    const MIN_MINUTES = 30;
    const MAX_MINUTES = 7 * 24 * 60; // 7 days
    if (
      !bookingSpotId ||
      bookingDurationMinutes < MIN_MINUTES ||
      bookingDurationMinutes > MAX_MINUTES
    ) {
      setMessage(
        `Please enter a valid duration (${MIN_MINUTES} - ${MAX_MINUTES} minutes)`,
      );
      return;
    }

    // Prepare start_at: if user provided a datetime use it, otherwise start now
    let startAtPayload = null;
    if (bookingStartAt && bookingStartAt.trim() !== "") {
      // bookingStartAt is in local datetime-local format: yyyy-MM-ddTHH:mm
      // Convert to ISO by appending seconds and treating as local
      const dt = new Date(bookingStartAt);
      if (isNaN(dt.getTime())) {
        setMessage("Invalid start time");
        return;
      }
      // Do not allow in past
      if (dt < new Date()) {
        setMessage("Start time cannot be in the past");
        return;
      }
      startAtPayload = dt.toISOString();
    }

    const token = getToken();
    try {
      const res = await fetch(
        `http://localhost:5000/api/bookings/reserve/${bookingSpotId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            duration_minutes: bookingDurationMinutes,
            ...(startAtPayload ? { start_at: startAtPayload } : {}),
          }),
        },
      );

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "Booking failed");

      setMessage(
        `Booking successful! Slot #${data.slot_number} reserved. ${data.slots_remaining} slot(s) remaining.`,
      );
      closeBookingDialog();

      setSpots((prevSpots) =>
        prevSpots.map((spot) =>
          spot.id === bookingSpotId
            ? {
                ...spot,
                available_slots: data.slots_remaining,
                status: data.slots_remaining > 0 ? "available" : "occupied",
              }
            : spot,
        ),
      );

      if (selectedSpot && selectedSpot.id === bookingSpotId) {
        setSelectedSpot((prev) => ({
          ...prev,
          available_slots: data.slots_remaining,
          status: data.slots_remaining > 0 ? "available" : "occupied",
        }));
      }

      setTimeout(() => fetchSpots(false), 1000);

      setTimeout(() => {
        setSelectedSpot(null);
        clearRoute();
      }, 5000);
    } catch (err) {
      setMessage(err.message);
    }
  };

  const getSpotStatusInfo = (status) => {
    switch (status) {
      case "available":
        return {
          text: "Available",
          color: "#10b981",
          bg: "#dcfce7",
          icon: "check_circle",
        };
      case "reserved":
      case "occupied":
        return {
          text: "Occupied",
          color: "#ef4444",
          bg: "#fee2e2",
          icon: "cancel",
        };
      case "unavailable":
        return {
          text: "Unavailable",
          color: "#f59e0b",
          bg: "#fed7aa",
          icon: "remove_circle",
        };
      default:
        return {
          text: "Unknown",
          color: "#6b7280",
          bg: "#f3f4f6",
          icon: "help",
        };
    }
  };

  const selectSpot = (spot) => {
    if (selectedSpot && selectedSpot.id === spot.id) {
      // Deselect if clicking the same spot
      setSelectedSpot(null);
      setMapCenter(null);
      setMapZoom(15);
      lastCenteredSpotId.current = null;
      if (markerRefs.current[spot.id]) {
        markerRefs.current[spot.id].closePopup();
      }
    } else {
      // Close all other popups
      Object.keys(markerRefs.current).forEach((id) => {
        if (markerRefs.current[id] && parseInt(id) !== spot.id) {
          markerRefs.current[id].closePopup();
        }
      });

      setSelectedSpot(spot);

      // Only center map if this is a different spot than the last centered one
      if (lastCenteredSpotId.current !== spot.id) {
        setMapCenter([spot.latitude, spot.longitude]);
        setMapZoom(17);
        lastCenteredSpotId.current = spot.id;
      }

      // Scroll the selected spot card into view and open popup
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

  const toggleSpotExpanded = (spotId) => {
    const newExpandedState = expandedSpotId === spotId ? null : spotId;
    setExpandedSpotId(newExpandedState);

    // Close popup when expanding details
    if (newExpandedState === spotId && markerRefs.current[spotId]) {
      markerRefs.current[spotId].closePopup();
    }

    // Reopen popup when collapsing details
    if (
      newExpandedState === null &&
      selectedSpot &&
      selectedSpot.id === spotId &&
      markerRefs.current[spotId]
    ) {
      setTimeout(() => {
        markerRefs.current[spotId].openPopup();
      }, 100);
    }
  };

  const filterAndSortSpots = (spotsToFilter) => {
    let filtered = spotsToFilter.filter((spot) => {
      // Search filter (by name or address)
      if (searchQuery && searchQuery.trim() !== "") {
        const q = searchQuery.trim().toLowerCase();
        const name = (spot.name || "").toLowerCase();
        const address = (spot.address || "").toLowerCase();
        if (!name.includes(q) && !address.includes(q)) return false;
      }
      if (spot.price < filters.minPrice) return false;
      if (filters.maxPrice !== null && spot.price > filters.maxPrice)
        return false;

      if (userLocation) {
        const distance = calculateDistance(
          userLocation[0],
          userLocation[1],
          spot.latitude,
          spot.longitude,
        );
        if (distance < filters.minDistance) return false;
        if (filters.maxDistance !== null && distance > filters.maxDistance)
          return false;
      }

      if (filters.status !== "all") {
        if (filters.status === "unavailable") {
          if (spot.status !== "unavailable" && spot.status !== "occupied")
            return false;
        } else {
          if (spot.status !== filters.status) return false;
        }
      }

      return true;
    });

    filtered.sort((a, b) => {
      switch (filters.sortBy) {
        case "distance":
          if (!userLocation) return 0;
          const distA = calculateDistance(
            userLocation[0],
            userLocation[1],
            a.latitude,
            a.longitude,
          );
          const distB = calculateDistance(
            userLocation[0],
            userLocation[1],
            b.latitude,
            b.longitude,
          );
          return distA - distB;
        case "price-low":
          return a.price - b.price;
        case "price-high":
          return b.price - a.price;
        case "name":
          return a.name.localeCompare(b.name);
        default:
          return 0;
      }
    });

    return filtered;
  };

  const resetFilters = () => {
    setFilters({
      minPrice: 0,
      maxPrice: null,
      minDistance: 0,
      maxDistance: null,
      status: "all",
      sortBy: "distance",
    });
  };

  const filteredSpots = filterAndSortSpots(spots);

  useEffect(() => {
    getUserLocation();
    fetchSpots(true);

    refreshIntervalRef.current = setInterval(() => fetchSpots(false), 2000);

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, []);

  const createUserIcon = () => {
    return L.divIcon({
      html: `<div class="flex items-center justify-center">
        <div class="w-4 h-4 rounded-full bg-[#4F86C6] border-2 border-white shadow-lg"></div>
      </div>`,
      className: "user-location-marker",
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });
  };

  const createSpotIcon = (spot, isSelected = false) => {
    let color;
    const status = spot.status;

    switch (status) {
      case "available":
        color = "#10b981";
        break;
      case "occupied":
      case "reserved":
        color = "#ef4444";
        break;
      case "unavailable":
        color = "#f59e0b";
        break;
      default:
        color = "#6b7280";
    }

    const size = isSelected ? 34 : 28;
    const pulseAnimation = isSelected
      ? `
      @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
      }
      animation: pulse 2s ease-in-out infinite;
    `
      : "";

    return L.divIcon({
      html: `<div style="
        width: ${size}px;
        height: ${size}px;
        border-radius: 50%;
        background: ${color};
        border: 3px solid white;
        box-shadow: ${isSelected ? "0 0 0 4px rgba(139, 92, 246, 0.3)," : ""} 0 ${isSelected ? "6" : "4"}px ${isSelected ? "16" : "12"}px rgba(0, 0, 0, ${isSelected ? "0.25" : "0.15"});
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: white;
        font-size: ${isSelected ? "16" : "14"}px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        ${pulseAnimation}
      ">
        P
      </div>`,
      className: "parking-marker",
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="text-center bg-white p-10 rounded-2xl shadow-soft border border-slate-100">
          <div className="w-16 h-16 border-4 border-slate-200 border-t-[#4F86C6] rounded-full animate-spin mx-auto mb-6"></div>
          <h2 className="text-slate-800 text-xl font-bold mb-2">
            Loading NExtSPot
          </h2>
          <p className="text-slate-500 text-sm">
            Finding nearby parking spots...
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
        rel="stylesheet"
      />
      <link
        href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
        rel="stylesheet"
      />

      <div className="h-screen w-full bg-[#F8FAFC] font-['Inter'] text-[#334155] flex flex-col overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-[#E2E8F0] bg-white px-6 py-3 h-16 shrink-0 z-20 shadow-sm">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 flex items-center justify-center rounded-lg bg-[#4F86C6]/10 text-[#4F86C6]">
              <span className="material-symbols-outlined">local_parking</span>
            </div>
            <h2 className="text-lg font-bold tracking-tight">NExtSPot</h2>
          </div>

          <div className="flex flex-1 justify-end gap-6 items-center">
            <nav className="hidden md:flex gap-6 text-sm font-medium text-[#64748B]">
              <a
                className="hover:text-[#4F86C6] transition-colors"
                href="/user/reservations"
              >
                My Bookings
              </a>
              <a
                className="hover:text-[#4F86C6] transition-colors"
                href="/user/reviews"
              >
                My Reviews
              </a>
            </nav>

            <div className="h-6 w-px bg-[#E2E8F0] mx-2"></div>

            <div className="flex items-center gap-3">
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white text-sm font-semibold rounded-lg transition-colors"
              >
                Logout
              </button>
            </div>
          </div>
        </header>

        {/* Status Bar */}
        {(message || locationError) && (
          <div
            className={`px-6 py-3 border-b ${
              locationError ||
              message.includes("error") ||
              message.includes("failed")
                ? "bg-red-50 border-red-200 text-red-700"
                : message.includes("successful")
                  ? "bg-green-50 border-green-200 text-green-700"
                  : "bg-blue-50 border-blue-200 text-blue-700"
            }`}
          >
            <div className="flex items-center gap-2 text-sm font-medium">
              <span className="material-symbols-outlined text-lg">
                {locationError || message.includes("error")
                  ? "error"
                  : message.includes("successful")
                    ? "check_circle"
                    : "info"}
              </span>
              {locationError || message}
            </div>
          </div>
        )}

        {/* Main Content */}
        <main className="flex flex-1 overflow-hidden relative">
          {/* Sidebar */}
          <aside
            className={`sidebar ${showSpotsList ? "open" : "closed"} w-[420px] flex flex-col bg-white border-r border-[#E2E8F0] z-10 shadow-[0_4px_20px_-2px_rgba(0,0,0,0.05)] shrink-0`}
          >
            {/* Spots Header - moved to top */}
            <div className="px-5 py-3 border-b border-[#E2E8F0] bg-white">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[#4F86C6]">
                    location_on
                  </span>
                  <h3 className="font-bold text-base">
                    Found {filteredSpots.length} spots nearby
                  </h3>
                </div>
                <button
                  onClick={() => setShowSpotsList(false)}
                  className="text-slate-400 hover:text-slate-600 transition-colors"
                >
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
            </div>

            {/* Search Section */}
            <div className="flex flex-col border-b border-[#E2E8F0]">
              <div className="px-5 pt-5 pb-3">
                <label className="flex flex-col w-full">
                  <div className="flex w-full items-center rounded-xl h-12 bg-white border border-[#E2E8F0] shadow-sm focus-within:border-[#4F86C6] focus-within:ring-2 focus-within:ring-[#4F86C6]/20 transition-all">
                    <div className="text-[#4F86C6] flex items-center justify-center pl-4 pr-2">
                      <span className="material-symbols-outlined">search</span>
                    </div>
                    <input
                      className="w-full bg-transparent border-none focus:outline-none focus:ring-0 placeholder:text-slate-400 px-2 text-base font-normal"
                      placeholder="Where do you want to park?"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          const first = filteredSpots && filteredSpots[0];
                          if (first) {
                            selectSpot(first);
                            setShowSpotsList(false);
                            setMessage(`Selected ${first.name}`);
                          }
                        }
                      }}
                    />
                    <button className="text-slate-400 hover:text-[#334155] flex items-center justify-center px-4">
                      <span
                        className="material-symbols-outlined"
                        onClick={() => setSearchQuery("")}
                        style={{ cursor: "pointer" }}
                      >
                        close
                      </span>
                    </button>
                  </div>
                </label>
              </div>

              {/* Filter Tags */}
              <div className="px-5 pb-5 overflow-x-auto">
                <div className="flex gap-2">
                  <button
                    onClick={() => setShowFilters(!showFilters)}
                    className={`flex h-8 items-center gap-1.5 rounded-full px-3 transition-colors whitespace-nowrap border text-xs font-bold ${
                      showFilters
                        ? "bg-[#4F86C6] text-white border-[#4F86C6]"
                        : "bg-white text-[#64748B] border-[#E2E8F0] hover:bg-slate-50 hover:text-[#4F86C6]"
                    }`}
                  >
                    <span className="material-symbols-outlined text-[18px]">
                      tune
                    </span>
                    <span>Filters</span>
                  </button>
                </div>
              </div>

              {/* Filters Panel */}
              {showFilters && (
                <div className="px-5 pb-5 bg-slate-50 border-t border-[#E2E8F0]">
                  <div className="pt-4 space-y-4">
                    {/* Price Range */}
                    <div>
                      <label className="text-xs font-semibold text-[#64748B] uppercase tracking-wide mb-2 block">
                        Price Range
                      </label>
                      <div className="flex gap-2">
                        <input
                          type="number"
                          placeholder="Min ₹"
                          min="0"
                          step="10"
                          value={filters.minPrice || ""}
                          onChange={(e) =>
                            setFilters({
                              ...filters,
                              minPrice: e.target.value
                                ? Math.max(0, Number(e.target.value))
                                : 0,
                            })
                          }
                          className="flex-1 min-w-0 px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:border-[#4F86C6] focus:ring-2 focus:ring-[#4F86C6]/20"
                        />
                        <input
                          type="number"
                          placeholder="Max ₹"
                          min="0"
                          step="10"
                          value={filters.maxPrice || ""}
                          onChange={(e) =>
                            setFilters({
                              ...filters,
                              maxPrice: e.target.value
                                ? Number(e.target.value)
                                : null,
                            })
                          }
                          className="flex-1 min-w-0 px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:border-[#4F86C6] focus:ring-2 focus:ring-[#4F86C6]/20"
                        />
                      </div>
                    </div>

                    {/* Distance Range */}
                    <div className={userLocation ? "" : "opacity-50"}>
                      <label className="text-xs font-semibold text-[#64748B] uppercase tracking-wide mb-2 block">
                        Distance Range (km)
                      </label>
                      <div className="flex gap-2">
                        <input
                          type="number"
                          placeholder="Min"
                          min="0"
                          step="0.5"
                          value={filters.minDistance || ""}
                          onChange={(e) =>
                            setFilters({
                              ...filters,
                              minDistance: e.target.value
                                ? Math.max(0, Number(e.target.value))
                                : 0,
                            })
                          }
                          disabled={!userLocation}
                          className="flex-1 min-w-0 px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:border-[#4F86C6] focus:ring-2 focus:ring-[#4F86C6]/20 disabled:cursor-not-allowed"
                        />
                        <input
                          type="number"
                          placeholder="Max"
                          min="0"
                          step="0.5"
                          value={filters.maxDistance || ""}
                          onChange={(e) =>
                            setFilters({
                              ...filters,
                              maxDistance: e.target.value
                                ? Number(e.target.value)
                                : null,
                            })
                          }
                          disabled={!userLocation}
                          className="flex-1 min-w-0 px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm focus:outline-none focus:border-[#4F86C6] focus:ring-2 focus:ring-[#4F86C6]/20 disabled:cursor-not-allowed"
                        />
                      </div>
                      {!userLocation && (
                        <p className="text-[10px] text-red-600 mt-1 font-medium">
                          ⚠️ Enable location to filter by distance
                        </p>
                      )}
                    </div>

                    {/* Status Filter */}
                    <div>
                      <label className="text-xs font-semibold text-[#64748B] uppercase tracking-wide mb-2 block">
                        Status
                      </label>
                      <select
                        value={filters.status}
                        onChange={(e) =>
                          setFilters({ ...filters, status: e.target.value })
                        }
                        className="w-full px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm bg-white focus:outline-none focus:border-[#4F86C6] focus:ring-2 focus:ring-[#4F86C6]/20"
                      >
                        <option value="all">All Spots</option>
                        <option value="available">Available</option>
                        <option value="reserved">Reserved</option>
                        <option value="unavailable">Unavailable</option>
                      </select>
                    </div>

                    {/* Sort By */}
                    <div>
                      <label className="text-xs font-semibold text-[#64748B] uppercase tracking-wide mb-2 block">
                        Sort By
                      </label>
                      <select
                        value={filters.sortBy}
                        onChange={(e) =>
                          setFilters({ ...filters, sortBy: e.target.value })
                        }
                        className="w-full px-3 py-2 border border-[#E2E8F0] rounded-lg text-sm bg-white focus:outline-none focus:border-[#4F86C6] focus:ring-2 focus:ring-[#4F86C6]/20"
                      >
                        <option value="distance">Distance (Nearest)</option>
                        <option value="price-low">Price (Low to High)</option>
                        <option value="price-high">Price (High to Low)</option>
                        <option value="name">Name (A-Z)</option>
                      </select>
                    </div>

                    {/* Reset Button */}
                    <button
                      onClick={resetFilters}
                      className="w-full py-2 bg-white border border-red-300 hover:bg-red-50 text-red-600 text-sm font-semibold rounded-lg transition-colors"
                    >
                      Reset Filters
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Spots List */}
            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3">
              {filteredSpots.map((spot) => {
                const statusInfo = getSpotStatusInfo(spot.status);
                const spotReview = reviewSummaries[spot.id];
                const isExpanded = expandedSpotId === spot.id;
                const isSelected = selectedSpot?.id === spot.id;

                return (
                  <div
                    key={spot.id}
                    ref={(el) => (spotCardRefs.current[spot.id] = el)}
                    className={`bg-white rounded-xl border transition-all duration-300 shadow-[0_2px_10px_-1px_rgba(0,0,0,0.03)] ${
                      isSelected
                        ? "border-[#4F86C6] shadow-[0_4px_20px_-2px_rgba(79,134,198,0.2)] scale-[1.01]"
                        : "border-[#E2E8F0] hover:border-[#4F86C6]/30 hover:shadow-[0_4px_20px_-2px_rgba(0,0,0,0.05)]"
                    }`}
                  >
                    {/* Collapsed View - Always Visible */}
                    <div
                      onClick={() => selectSpot(spot)}
                      className="p-4 cursor-pointer transition-colors hover:bg-slate-50/50"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-bold text-base text-[#334155] hover:text-[#4F86C6] transition-colors">
                              {spot.name}
                            </h3>
                            {spotReview && spotReview.average_rating > 0 && (
                              <div className="flex items-center gap-1 bg-amber-50 px-2 py-0.5 rounded-md">
                                <span
                                  className="material-symbols-outlined text-amber-500"
                                  style={{ fontSize: "14px" }}
                                >
                                  star
                                </span>
                                <span className="text-xs font-bold text-amber-700">
                                  {spotReview.average_rating.toFixed(1)}
                                </span>
                              </div>
                            )}
                          </div>

                          <div className="flex items-center gap-2 mb-2">
                            <span
                              className="px-2 py-0.5 rounded-md text-[10px] font-semibold border"
                              style={{
                                backgroundColor: statusInfo.bg,
                                color: statusInfo.color,
                                borderColor: statusInfo.color + "30",
                              }}
                            >
                              {statusInfo.text}
                            </span>
                            <span className="px-2 py-0.5 rounded-md bg-slate-50 text-[10px] text-slate-500 font-medium border border-slate-200">
                              {spot.available_slots}/{spot.total_slots} slots
                            </span>
                            {userLocation && (
                              <span className="text-xs text-[#64748B]">
                                •{" "}
                                {safeToFixed(
                                  calculateDistance(
                                    userLocation[0],
                                    userLocation[1],
                                    spot.latitude,
                                    spot.longitude,
                                  ),
                                  1,
                                )}{" "}
                                km
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-2">
                            <span className="text-sm font-bold text-[#4F86C6]">
                              ₹{spot.price}/hr
                            </span>
                          </div>
                        </div>

                        <button
                          className="ml-2 p-1 hover:bg-slate-100 rounded-lg transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleSpotExpanded(spot.id);
                          }}
                        >
                          <span
                            className={`material-symbols-outlined text-[#64748B] transition-transform ${
                              isExpanded ? "rotate-180" : ""
                            }`}
                          >
                            expand_more
                          </span>
                        </button>
                      </div>
                    </div>

                    {/* Expanded View - Accordion Content */}
                    <div
                      className={`overflow-hidden transition-all duration-300 ease-in-out ${
                        isExpanded
                          ? "max-h-[800px] opacity-100"
                          : "max-h-0 opacity-0"
                      }`}
                    >
                      <div className="px-4 pb-4 border-t border-slate-100 pt-4">
                        {/* Address */}
                        <p className="text-sm text-[#64748B] mb-4">
                          {spot.address ||
                            `Lat: ${safeToFixed(spot.latitude, 4)}, Lng: ${safeToFixed(spot.longitude, 4)}`}
                        </p>

                        {/* Action Buttons */}
                        {spot.status === "available" && (
                          <div className="flex gap-2 mb-2">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                showRoute(
                                  spot.latitude,
                                  spot.longitude,
                                  spot.name,
                                );
                              }}
                              className="flex-1 py-2.5 bg-white border border-[#E2E8F0] hover:bg-slate-50 hover:border-[#4F86C6] text-[#334155] text-sm font-semibold rounded-lg transition-all duration-200 flex items-center justify-center gap-2 hover:shadow-md"
                            >
                              <span className="material-symbols-outlined text-[18px]">
                                directions
                              </span>
                              Route
                            </button>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                openBookingDialog(spot.id);
                              }}
                              className="flex-1 py-2.5 bg-[#4F86C6] hover:bg-[#3B6FA6] text-white text-sm font-semibold rounded-lg transition-all duration-200 shadow-sm shadow-[#4F86C6]/30 hover:shadow-lg hover:shadow-[#4F86C6]/40 hover:-translate-y-0.5"
                            >
                              Book Now
                            </button>
                          </div>
                        )}

                        {/* Reviews Button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            openReviewsModal(spot);
                          }}
                          className="w-full py-2 bg-amber-50 border border-amber-200 hover:bg-amber-100 text-amber-800 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center justify-center gap-2 hover:shadow-md"
                        >
                          <span className="material-symbols-outlined text-[16px]">
                            star
                          </span>
                          View Reviews
                          {canReviewSpots[spot.id]?.canReview && (
                            <span className="bg-purple-600 text-white px-2 py-0.5 rounded text-[10px] font-bold">
                              Can Review
                            </span>
                          )}
                        </button>

                        {/* Booking Schedule Button */}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setScheduleSpotId(spot.id);
                            setShowScheduleModal(true);
                          }}
                          className="w-full py-2 bg-blue-50 border border-blue-200 hover:bg-blue-100 text-blue-800 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center justify-center gap-2 hover:shadow-md mt-2"
                        >
                          <span className="material-symbols-outlined text-[16px]">
                            calendar_month
                          </span>
                          View Booking Schedule
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}

              {filteredSpots.length === 0 && (
                <div className="text-center py-16 text-[#64748B]">
                  <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="material-symbols-outlined text-4xl text-slate-300">
                      search_off
                    </span>
                  </div>
                  <p className="font-semibold text-base mb-2">No spots found</p>
                  <p className="text-sm mb-4">Try adjusting your filters</p>
                  <button
                    onClick={resetFilters}
                    className="px-6 py-2 bg-[#4F86C6] hover:bg-[#3B6FA6] text-white text-sm font-semibold rounded-lg transition-colors"
                  >
                    Reset Filters
                  </button>
                </div>
              )}
            </div>
          </aside>

          {/* Map Section */}
          <div className="flex-1 relative bg-slate-200">
            {!showSpotsList && (
              <button
                onClick={() => setShowSpotsList(true)}
                className="absolute top-4 left-4 z-[1000] bg-white border border-[#E2E8F0] px-4 py-2.5 rounded-lg shadow-lg flex items-center gap-2 hover:bg-slate-50 transition-colors"
              >
                <span className="material-symbols-outlined text-[#4F86C6]">
                  list
                </span>
                <span className="text-sm font-semibold">Show List</span>
              </button>
            )}

            {/* Map Controls - shift down when route control is present to avoid overlap */}
            <div
              className={`absolute ${
                selectedRoute || routeDestination ? "top-20" : "top-4"
              } right-4 z-[1000] flex flex-col gap-2`}
            >
              {(selectedRoute || selectedSpot) && (
                <button
                  onClick={clearRoute}
                  className="w-10 h-10 bg-white border border-red-300 text-red-600 rounded-lg shadow-lg flex items-center justify-center hover:bg-red-50 transition-colors"
                  title="Clear Route"
                >
                  <span className="material-symbols-outlined">close</span>
                </button>
              )}

              <button
                onClick={getUserLocation}
                className="w-10 h-10 bg-white border border-gray-200 text-[#4F86C6] rounded-lg shadow-lg flex items-center justify-center hover:bg-gray-50 transition-colors"
                title="Locate Me"
              >
                <span className="material-symbols-outlined">my_location</span>
              </button>
            </div>

            {mapInitialized ? (
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
                maxBoundsOptions={{
                  animate: true,
                  duration: 0.25,
                  padding: [20, 20],
                }}
                style={{ height: "100%", width: "100%" }}
                zoomControl={false}
                attributionControl={false}
              >
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution=""
                />

                <ZoomControl position="bottomright" />

                <MapController
                  userLocation={userLocation}
                  initialSpots={initialSpots}
                  hasInitialized={mapInitialized}
                  hasActiveRoute={!!selectedRoute}
                  shouldClosePopups={shouldClosePopups}
                  onPopupsClosed={handlePopupsClosed}
                />

                <RoutingController
                  userLocation={userLocation}
                  destination={routeDestination}
                  onRouteFound={handleRouteFound}
                  onRouteClear={handleRouteClear}
                />
                <MapResizeHandler showSpotsList={showSpotsList} />
                <MapViewController
                  center={mapCenter}
                  zoom={mapZoom}
                  onAnimationComplete={() => setMapCenter(null)}
                />

                {userLocation && (
                  <Marker position={userLocation} icon={createUserIcon()}>
                    <Popup>
                      <div className="p-3 min-w-[180px]">
                        <h4 className="text-base font-semibold mb-2 text-[#334155] flex items-center gap-2">
                          <span className="material-symbols-outlined text-[#4F86C6]">
                            person_pin_circle
                          </span>
                          Your Location
                        </h4>
                        <p className="text-sm text-[#64748B]">
                          Current position
                        </p>
                      </div>
                    </Popup>
                  </Marker>
                )}

                {filteredSpots
                  .filter(
                    (spot) =>
                      spot && spot.latitude != null && spot.longitude != null,
                  )
                  .map((spot) => (
                    <Marker
                      key={`${spot.id}-${spot.status}-${selectedSpot?.id === spot.id ? "selected" : "unselected"}`}
                      position={[
                        parseFloat(spot.latitude),
                        parseFloat(spot.longitude),
                      ]}
                      icon={createSpotIcon(spot, selectedSpot?.id === spot.id)}
                      ref={(ref) => {
                        markerRefs.current[spot.id] = ref;
                      }}
                      eventHandlers={{
                        click: (e) => {
                          e.originalEvent?.stopPropagation?.();
                          selectSpot(spot);
                        },
                      }}
                    >
                      <Popup>
                        <div className="p-4 min-w-[280px]">
                          <h4 className="font-bold text-lg text-[#334155] mb-3 flex items-center gap-2">
                            <span className="material-symbols-outlined text-[#4F86C6]">
                              local_parking
                            </span>
                            {spot.name}
                          </h4>

                          <div className="space-y-2 mb-4">
                            <div className="flex justify-between text-sm">
                              <span className="text-[#64748B]">Available:</span>
                              <span className="font-semibold text-[#334155]">
                                {spot.available_slots}/{spot.total_slots} slots
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span className="text-[#64748B]">Price:</span>
                              <span className="font-semibold text-[#334155]">
                                ₹{spot.price}/hr
                              </span>
                            </div>
                            {userLocation && (
                              <div className="flex justify-between text-sm">
                                <span className="text-[#64748B]">
                                  Distance:
                                </span>
                                <span className="font-semibold text-[#334155]">
                                  {safeToFixed(
                                    calculateDistance(
                                      userLocation[0],
                                      userLocation[1],
                                      spot.latitude,
                                      spot.longitude,
                                    ),
                                    1,
                                  )}{" "}
                                  km
                                </span>
                              </div>
                            )}
                          </div>

                          {spot.status === "available" && (
                            <div className="flex gap-2 mb-2">
                              <button
                                onClick={() =>
                                  showRoute(
                                    spot.latitude,
                                    spot.longitude,
                                    spot.name,
                                  )
                                }
                                className="flex-1 py-2 bg-blue-50 border border-blue-200 text-blue-700 text-sm font-semibold rounded-lg hover:bg-blue-100 transition-colors"
                              >
                                Directions
                              </button>
                              <button
                                onClick={() => openBookingDialog(spot.id)}
                                className="flex-1 py-2 bg-[#4F86C6] text-white text-sm font-semibold rounded-lg hover:bg-[#3B6FA6] transition-colors"
                              >
                                Reserve
                              </button>
                            </div>
                          )}

                          {/* Reviews Button */}
                          <button
                            onClick={() => openReviewsModal(spot)}
                            className="w-full py-2 bg-amber-50 border border-amber-200 hover:bg-amber-100 text-amber-800 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center justify-center gap-2 hover:shadow-md mb-2"
                          >
                            <span className="material-symbols-outlined text-[16px]">
                              star
                            </span>
                            View Reviews
                            {canReviewSpots[spot.id]?.canReview && (
                              <span className="bg-purple-600 text-white px-2 py-0.5 rounded text-[10px] font-bold">
                                Can Review
                              </span>
                            )}
                          </button>

                          {/* Booking Schedule Button */}
                          <button
                            onClick={() => {
                              setScheduleSpotId(spot.id);
                              setShowScheduleModal(true);
                            }}
                            className="w-full py-2 bg-blue-50 border border-blue-200 hover:bg-blue-100 text-blue-800 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center justify-center gap-2 hover:shadow-md"
                          >
                            <span className="material-symbols-outlined text-[16px]">
                              calendar_month
                            </span>
                            View Booking Schedule
                          </button>
                        </div>
                      </Popup>
                    </Marker>
                  ))}
              </MapContainer>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center bg-slate-100">
                <div className="text-center">
                  <div className="w-12 h-12 border-4 border-slate-200 border-t-[#4F86C6] rounded-full animate-spin mx-auto mb-4"></div>
                  <p className="text-[#64748B] text-sm">Initializing map...</p>
                </div>
              </div>
            )}
          </div>
        </main>

        {/* Selected Spot Info Bar removed - using popup only */}

        {/* Booking Dialog */}
        {showBookingDialog && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-[10000] p-4">
            <div className="bg-white rounded-2xl p-8 max-w-md w-full shadow-[0_20px_40px_rgba(0,0,0,0.2)] animate-[slideIn_0.3s_ease-out]">
              <div className="text-center mb-6">
                <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <span className="material-symbols-outlined text-[#4F86C6] text-2xl">
                    event_available
                  </span>
                </div>
                <h3 className="text-xl font-bold text-[#334155] mb-2">
                  Reserve Parking Spot
                </h3>
                <p className="text-sm text-[#64748B]">
                  How long would you like to reserve this spot?
                </p>
              </div>

              <div className="mb-6">
                <label className="block text-xs font-semibold text-[#64748B] uppercase tracking-wide mb-3">
                  Duration
                </label>

                <div className="grid grid-cols-4 gap-2 mb-4">
                  {[
                    { m: 30, label: "30m" },
                    { m: 60, label: "1h" },
                    { m: 120, label: "2h" },
                    { m: 240, label: "4h" },
                    { m: 1440, label: "1d" },
                  ].map((p) => (
                    <button
                      key={p.m}
                      onClick={() => {
                        setBookingDurationMinutes(p.m);
                        setShowCustomDuration(false);
                      }}
                      className={`py-3 rounded-lg text-sm font-semibold transition-all ${
                        bookingDurationMinutes === p.m
                          ? "bg-[#4F86C6] text-white border-2 border-[#4F86C6]"
                          : "bg-white text-[#334155] border-2 border-[#E2E8F0] hover:border-[#4F86C6]/30"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}

                  <button
                    onClick={() => setShowCustomDuration((s) => !s)}
                    className={`py-3 rounded-lg text-sm font-semibold transition-all col-span-4 ${
                      showCustomDuration
                        ? "bg-[#4F86C6] text-white border-2 border-[#4F86C6]"
                        : "bg-white text-[#334155] border-2 border-[#E2E8F0] hover:border-[#4F86C6]/30"
                    }`}
                  >
                    {showCustomDuration
                      ? "Custom: set duration/start"
                      : "Custom"}
                  </button>
                </div>

                {showCustomDuration && (
                  <div className="space-y-3 mb-4">
                    <div>
                      <label className="text-xs text-[#64748B]">
                        Start time (optional)
                      </label>
                      <input
                        type="datetime-local"
                        value={bookingStartAt}
                        onChange={(e) => setBookingStartAt(e.target.value)}
                        className="mt-1 w-full border rounded p-2 text-sm"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-xs text-[#64748B]">Hours</label>
                        <input
                          type="number"
                          min={0}
                          className="mt-1 w-full border rounded p-2 text-sm"
                          value={Math.floor(bookingDurationMinutes / 60)}
                          onChange={(e) => {
                            const h = parseInt(e.target.value || 0, 10);
                            const m = bookingDurationMinutes % 60;
                            setBookingDurationMinutes(h * 60 + m);
                          }}
                        />
                      </div>
                      <div>
                        <label className="text-xs text-[#64748B]">
                          Minutes
                        </label>
                        <input
                          type="number"
                          min={0}
                          max={59}
                          className="mt-1 w-full border rounded p-2 text-sm"
                          value={bookingDurationMinutes % 60}
                          onChange={(e) => {
                            const mins = parseInt(e.target.value || 0, 10);
                            const h = Math.floor(bookingDurationMinutes / 60);
                            setBookingDurationMinutes(h * 60 + mins);
                          }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 text-sm">
                  <div className="flex justify-between mb-2">
                    <span className="text-[#64748B]">Duration:</span>
                    <span className="font-semibold text-[#334155]">
                      {Math.floor(bookingDurationMinutes / 60) > 0
                        ? `${Math.floor(bookingDurationMinutes / 60)}h${bookingDurationMinutes % 60 ? ` ${bookingDurationMinutes % 60}m` : ""}`
                        : `${bookingDurationMinutes}m`}
                    </span>
                  </div>
                  <div className="flex justify-between mb-2">
                    <span className="text-[#64748B]">Rate:</span>
                    <span className="font-semibold text-[#334155]">
                      ₹{spots.find((s) => s.id === bookingSpotId)?.price || 0}
                      /hour
                    </span>
                  </div>
                  <div className="flex justify-between pt-2 border-t border-slate-300">
                    <span className="font-bold text-[#334155]">Total:</span>
                    <span className="font-bold text-[#4F86C6] text-lg">
                      ₹
                      {(
                        (spots.find((s) => s.id === bookingSpotId)?.price ||
                          0) *
                        (bookingDurationMinutes / 60)
                      ).toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={closeBookingDialog}
                  className="flex-1 py-3 border-2 border-[#E2E8F0] hover:bg-slate-50 text-[#64748B] text-sm font-semibold rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmBooking}
                  className="flex-1 py-3 bg-[#4F86C6] hover:bg-[#3B6FA6] text-white text-sm font-semibold rounded-lg transition-colors shadow-sm shadow-[#4F86C6]/30"
                >
                  Confirm Booking
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Reviews Modal */}
        {showReviewsModal && selectedSpotForReviews && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[10001] p-4 overflow-y-auto">
            <div className="bg-white rounded-2xl max-w-3xl w-full max-h-[90vh] shadow-[0_20px_40px_rgba(0,0,0,0.2)] flex flex-col">
              {/* Modal Header */}
              <div className="px-6 py-5 border-b border-[#E2E8F0] bg-gradient-to-r from-[#4F86C6] to-[#74B9BE] text-white rounded-t-2xl">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold mb-2 flex items-center gap-2">
                      <span className="material-symbols-outlined text-3xl">
                        reviews
                      </span>
                      {selectedSpotForReviews.name}
                    </h3>
                    <p className="text-sm opacity-90">
                      See what others are saying about this spot
                    </p>

                    {reviewSummaries[selectedSpotForReviews.id] && (
                      <div className="flex items-center gap-3 mt-3 bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 w-fit">
                        <div className="flex items-center gap-1">
                          {[...Array(5)].map((_, i) => (
                            <span
                              key={i}
                              className="material-symbols-outlined text-lg"
                              style={{
                                color:
                                  i <
                                  Math.round(
                                    reviewSummaries[selectedSpotForReviews.id]
                                      .average_rating,
                                  )
                                    ? "#fbbf24"
                                    : "rgba(255,255,255,0.3)",
                              }}
                            >
                              star
                            </span>
                          ))}
                        </div>
                        <span className="text-lg font-bold">
                          {reviewSummaries[
                            selectedSpotForReviews.id
                          ].average_rating.toFixed(1)}
                        </span>
                        <span className="text-sm opacity-90">
                          (
                          {
                            reviewSummaries[selectedSpotForReviews.id]
                              .total_reviews
                          }{" "}
                          reviews)
                        </span>
                      </div>
                    )}
                  </div>

                  <button
                    onClick={closeReviewsModal}
                    className="bg-white/20 backdrop-blur-sm p-2 rounded-lg hover:bg-white/30 transition-colors"
                  >
                    <span className="material-symbols-outlined text-2xl">
                      close
                    </span>
                  </button>
                </div>
              </div>

              {/* Modal Content */}
              <div className="flex-1 overflow-y-auto px-6 py-4">
                {reviewsLoading ? (
                  <div className="text-center py-16">
                    <div className="w-12 h-12 border-4 border-slate-200 border-t-[#4F86C6] rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-[#64748B] text-sm">Loading reviews...</p>
                  </div>
                ) : selectedSpotReviews.length === 0 ? (
                  <div className="text-center py-16 text-[#64748B]">
                    <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                      <span className="material-symbols-outlined text-4xl text-slate-300">
                        rate_review
                      </span>
                    </div>
                    <p className="font-semibold text-base mb-2">
                      No reviews yet
                    </p>
                    <p className="text-sm">
                      Be the first to review this parking spot!
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {selectedSpotReviews.map((review) => (
                      <div
                        key={review.id}
                        className="bg-slate-50 border border-slate-200 rounded-xl p-5 hover:shadow-sm transition-shadow"
                      >
                        <div className="flex justify-between items-start mb-3">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[#4F86C6] to-[#74B9BE] flex items-center justify-center text-white font-bold text-sm">
                              {review.user_name?.charAt(0).toUpperCase() || "U"}
                            </div>
                            <div>
                              <h4 className="font-semibold text-[#334155]">
                                {review.user_name || "Anonymous User"}
                              </h4>
                              <p className="text-xs text-[#64748B]">
                                {new Date(review.created_at).toLocaleDateString(
                                  "en-US",
                                  {
                                    year: "numeric",
                                    month: "short",
                                    day: "numeric",
                                  },
                                )}
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-1">
                            {[...Array(5)].map((_, i) => (
                              <span
                                key={i}
                                className="material-symbols-outlined text-base"
                                style={{
                                  color:
                                    i < review.rating ? "#fbbf24" : "#e5e7eb",
                                }}
                              >
                                star
                              </span>
                            ))}
                          </div>
                        </div>

                        {review.comment && (
                          <p className="text-sm text-[#334155] mb-3 bg-white p-3 rounded-lg border border-slate-100">
                            {review.comment}
                          </p>
                        )}

                        {(review.cleanliness_rating ||
                          review.safety_rating ||
                          review.accessibility_rating) && (
                          <div className="grid grid-cols-3 gap-2 mb-3">
                            {review.cleanliness_rating && (
                              <div className="bg-white p-2 rounded-lg border border-slate-100 text-center">
                                <p className="text-[10px] text-[#64748B] mb-1">
                                  Cleanliness
                                </p>
                                <p className="text-sm font-bold text-[#4F86C6]">
                                  {review.cleanliness_rating}/5
                                </p>
                              </div>
                            )}
                            {review.safety_rating && (
                              <div className="bg-white p-2 rounded-lg border border-slate-100 text-center">
                                <p className="text-[10px] text-[#64748B] mb-1">
                                  Safety
                                </p>
                                <p className="text-sm font-bold text-[#4F86C6]">
                                  {review.safety_rating}/5
                                </p>
                              </div>
                            )}
                            {review.accessibility_rating && (
                              <div className="bg-white p-2 rounded-lg border border-slate-100 text-center">
                                <p className="text-[10px] text-[#64748B] mb-1">
                                  Access
                                </p>
                                <p className="text-sm font-bold text-[#4F86C6]">
                                  {review.accessibility_rating}/5
                                </p>
                              </div>
                            )}
                          </div>
                        )}

                        <div className="flex items-center justify-between pt-3 border-t border-slate-200">
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => voteOnReview(review.id, true)}
                              className="flex items-center gap-1 bg-white border border-slate-200 px-3 py-1.5 rounded-lg text-xs font-medium text-[#334155] hover:bg-green-50 hover:border-green-200 hover:text-green-700 transition-colors"
                            >
                              <span className="material-symbols-outlined text-sm">
                                thumb_up
                              </span>
                              {review.helpful_count || 0}
                            </button>
                            <button
                              onClick={() => voteOnReview(review.id, false)}
                              className="flex items-center gap-1 bg-white border border-slate-200 px-3 py-1.5 rounded-lg text-xs font-medium text-[#334155] hover:bg-red-50 hover:border-red-200 hover:text-red-700 transition-colors"
                            >
                              <span className="material-symbols-outlined text-sm">
                                thumb_down
                              </span>
                              {review.not_helpful_count || 0}
                            </button>
                          </div>
                          {review.updated_at !== review.created_at && (
                            <span className="text-[10px] text-slate-400 italic">
                              Edited
                            </span>
                          )}
                        </div>

                        {review.owner_response && (
                          <div className="mt-3 bg-blue-50 border border-blue-100 rounded-lg p-3">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="material-symbols-outlined text-[#4F86C6] text-sm">
                                support_agent
                              </span>
                              <span className="text-xs font-semibold text-blue-900">
                                Response from {review.owner_name || "Owner"}
                              </span>
                              <span className="text-[10px] text-blue-600">
                                •{" "}
                                {new Date(
                                  review.response_created_at,
                                ).toLocaleDateString("en-US", {
                                  month: "short",
                                  day: "numeric",
                                })}
                              </span>
                            </div>
                            <p className="text-xs text-blue-900">
                              {review.owner_response}
                            </p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Pagination */}
              {reviewsPagination && reviewsPagination.totalPages > 1 && (
                <div className="px-6 py-4 border-t border-[#E2E8F0] bg-slate-50 rounded-b-2xl flex justify-between items-center">
                  <span className="text-sm text-[#64748B]">
                    Page {reviewsPagination.page} of{" "}
                    {reviewsPagination.totalPages}
                  </span>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleReviewsPageChange(reviewsPage - 1)}
                      disabled={reviewsPage === 1}
                      className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${
                        reviewsPage === 1
                          ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                          : "bg-white border border-[#E2E8F0] text-[#334155] hover:bg-slate-50"
                      }`}
                    >
                      Previous
                    </button>
                    <button
                      onClick={() => handleReviewsPageChange(reviewsPage + 1)}
                      disabled={reviewsPage === reviewsPagination.totalPages}
                      className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${
                        reviewsPage === reviewsPagination.totalPages
                          ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                          : "bg-[#4F86C6] text-white hover:bg-[#3B6FA6]"
                      }`}
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}

              {/* Can Review Notice */}
              {canReviewSpots[selectedSpotForReviews.id]?.canReview && (
                <div className="px-6 py-4 bg-green-50 border-t border-green-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="material-symbols-outlined text-green-600">
                      check_circle
                    </span>
                    <span className="text-sm font-semibold text-green-800">
                      You can review this parking spot!
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      closeReviewsModal();
                      // Use window.location to redirect, passing both spotId and bookingId
                      window.location.href = `/user/reservations?spotId=${selectedSpotForReviews.id}&bookingId=${canReviewSpots[selectedSpotForReviews.id].booking_id}`;
                    }}
                    className="px-5 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-semibold rounded-lg transition-colors"
                  >
                    Write Review
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        @keyframes slideIn {
          0% {
            opacity: 0;
            transform: translateY(-20px) scale(0.95);
          }
          100% {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }

        ::-webkit-scrollbar {
          width: 8px;
        }

        ::-webkit-scrollbar-track {
          background: #F8FAFC;
        }

        ::-webkit-scrollbar-thumb {
          background: #CBD5E1;
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: #94A3B8;
        }

        .leaflet-popup-content-wrapper {
          border-radius: 12px !important;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15) !important;
          border: none !important;
        }

        .leaflet-popup-tip {
          background: white !important;
        }

        .leaflet-control-zoom {
          border: none !important;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }

        .leaflet-control-zoom a {
          background: white !important;
          color: #334155 !important;
          border: none !important;
          border-radius: 8px !important;
          margin: 2px !important;
          font-weight: 600 !important;
        }

        .leaflet-control-zoom a:hover {
          background: #F1F5F9 !important;
        }

        .parking-marker:hover div {
          transform: scale(1.15) !important;
          box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25) !important;
        }

        .material-symbols-outlined {
          font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        }

        /* Smooth sidebar collapse/expand for nicer map resize */
        .sidebar {
          flex: 0 0 420px;
          transition: flex-basis 420ms cubic-bezier(0.22, 1, 0.36, 1),
                      opacity 320ms ease, transform 420ms cubic-bezier(0.22, 1, 0.36, 1);
          overflow: hidden;
          will-change: flex-basis, opacity, transform;
        }

        .sidebar.closed {
          flex-basis: 0 !important;
          opacity: 0;
          pointer-events: none;
          transform: translateX(-8px);
        }

        .sidebar.open {
          flex-basis: 420px !important;
          opacity: 1;
          pointer-events: auto;
          transform: translateX(0);
        }
      `}</style>

      {/* Booking Schedule Modal */}
      {showScheduleModal && scheduleSpotId && (
        <BookingSchedule
          parkingSpotId={scheduleSpotId}
          onClose={() => {
            setShowScheduleModal(false);
            setScheduleSpotId(null);
          }}
        />
      )}
    </>
  );
}

export default UserMap;
