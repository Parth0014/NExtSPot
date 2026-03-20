import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function Login() {
  const [form, setForm] = useState({ email: "", password: "" });
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage("Loading...");

    try {
      const res = await fetch("http://localhost:5000/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, role: "user" }),
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.message || "Login failed");

      // Save token and user info
      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));

      setMessage(`Welcome ${data.user.name}! Redirecting...`);
      setTimeout(() => navigate("/user/dashboard"), 1500);
    } catch (err) {
      setMessage(err.message || "Account not found");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.body}>
      {/* Subtle Background Illustrations */}
      <div style={styles.illustrationTopLeft}>
        <span style={styles.materialIcon}>local_parking</span>
      </div>
      <div style={styles.illustrationBottomRight}>
        <span style={styles.materialIcon}>directions_car</span>
      </div>
      <div style={styles.illustrationMiddleLeft}>
        <span style={styles.materialIcon}>electric_bolt</span>
      </div>

      {/* Navigation Bar */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logoContainer}>
            <svg
              fill="none"
              viewBox="0 0 48 48"
              xmlns="http://www.w3.org/2000/svg"
              style={styles.logo}
            >
              <path
                d="M4 4H17.3334V17.3334H30.6666V30.6666H44V44H4V4Z"
                fill="currentColor"
              ></path>
            </svg>
          </div>
          <h2 style={styles.headerTitle}>Get Your Parking</h2>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={styles.main}>
        <div style={styles.card}>
          {/* Headline & Body Text */}
          <div style={styles.cardHeader}>
            <h1 style={styles.title}>Welcome Back!</h1>
            <p style={styles.subtitle}>
              Enter your credentials to book your parking spaces.
            </p>
          </div>

          {/* Message Display */}
          {message && (
            <div
              style={
                message.includes("Welcome") ? styles.success : styles.error
              }
            >
              {message}
            </div>
          )}

          {/* Login Form */}
          <form style={styles.form} onSubmit={handleSubmit}>
            {/* Email Field */}
            <div style={styles.inputGroup}>
              <label style={styles.label} htmlFor="email">
                Email Address
              </label>
              <input
                id="email"
                type="email"
                name="email"
                placeholder="name@company.com"
                value={form.email}
                onChange={handleChange}
                style={styles.input}
                required
              />
            </div>

            {/* Password Field */}
            <div style={styles.inputGroup}>
              <div style={styles.passwordLabelRow}>
                <label style={styles.label} htmlFor="password">
                  Password
                </label>
                <a href="#" style={styles.forgotPassword}>
                  Forgot password?
                </a>
              </div>
              <div style={styles.passwordWrapper}>
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  name="password"
                  placeholder="Enter your password"
                  value={form.password}
                  onChange={handleChange}
                  style={styles.input}
                  required
                />
                <button
                  type="button"
                  style={styles.visibilityButton}
                  onClick={() => setShowPassword(!showPassword)}
                >
                  <span style={styles.visibilityIcon}>
                    {showPassword ? "visibility_off" : "visibility"}
                  </span>
                </button>
              </div>
            </div>

            {/* Primary CTA */}
            <button
              type="submit"
              disabled={isLoading}
              style={{
                ...styles.submitButton,
                ...(isLoading ? styles.submitButtonDisabled : {}),
              }}
            >
              {isLoading ? "Signing in..." : "Sign In"}
            </button>
          </form>

          {/* Create Account Link */}
          <div style={styles.footer}>
            <p style={styles.footerText}>
              Don't have an account?
              <button
                onClick={() => navigate("/register")}
                style={styles.createAccountLink}
              >
                Create an account
              </button>
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={styles.pageFooter}>
        © 2024 Access Your Parking. All rights reserved.
        <div style={styles.footerLinks}>
          <a href="#" style={styles.footerLink}>
            Privacy Policy
          </a>
          <a href="#" style={styles.footerLink}>
            Terms of Service
          </a>
        </div>
      </footer>
    </div>
  );
}

const styles = {
  body: {
    fontFamily:
      "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    backgroundColor: "#f6f8f8",
    minHeight: "100vh",
    position: "relative",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },

  // Background Illustrations
  illustrationTopLeft: {
    position: "absolute",
    top: "40px",
    left: "40px",
    opacity: 0.15,
    pointerEvents: "none",
    color: "#0e1b19",
    display: window.innerWidth >= 1024 ? "block" : "none",
  },
  illustrationBottomRight: {
    position: "absolute",
    bottom: "40px",
    right: "40px",
    opacity: 0.15,
    pointerEvents: "none",
    color: "#0e1b19",
    display: window.innerWidth >= 1024 ? "block" : "none",
  },
  illustrationMiddleLeft: {
    position: "absolute",
    top: "50%",
    left: "-50px",
    opacity: 0.15,
    pointerEvents: "none",
    color: "#0e1b19",
    display: window.innerWidth >= 1280 ? "block" : "none",
  },
  materialIcon: {
    fontFamily: "'Material Symbols Outlined'",
    fontSize: "120px",
    fontWeight: "normal",
    fontStyle: "normal",
    display: "inline-block",
    lineHeight: 1,
    textTransform: "none",
    letterSpacing: "normal",
    wordWrap: "normal",
    whiteSpace: "nowrap",
    direction: "ltr",
  },

  // Header
  header: {
    width: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "20px 40px",
    zIndex: 10,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "16px",
    color: "#0e1b19",
  },
  logoContainer: {
    width: "32px",
    height: "32px",
    color: "#19e6c4",
  },
  logo: {
    width: "100%",
    height: "100%",
  },
  headerTitle: {
    fontSize: "18px",
    fontWeight: "700",
    margin: 0,
    letterSpacing: "-0.015em",
  },

  // Main Content
  main: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px",
    zIndex: 10,
  },
  card: {
    width: "100%",
    maxWidth: "480px",
    backgroundColor: "#ffffff",
    boxShadow:
      "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
    borderRadius: "12px",
    padding: "48px",
    border: "1px solid #e7f3f1",
  },
  cardHeader: {
    textAlign: "center",
    marginBottom: "32px",
  },
  title: {
    color: "#0e1b19",
    fontSize: "30px",
    fontWeight: "700",
    margin: "0 0 12px 0",
    lineHeight: 1.2,
  },
  subtitle: {
    color: "#4e978b",
    fontSize: "16px",
    fontWeight: "400",
    margin: 0,
  },

  // Form
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  label: {
    color: "#0e1b19",
    fontSize: "14px",
    fontWeight: "500",
  },
  passwordLabelRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  forgotPassword: {
    color: "#19e6c4",
    fontSize: "12px",
    fontWeight: "600",
    textDecoration: "none",
  },
  input: {
    width: "100%",
    height: "56px",
    padding: "0 16px",
    borderRadius: "12px",
    border: "1px solid #d0e7e3",
    backgroundColor: "#f8fcfb",
    fontSize: "16px",
    color: "#0e1b19",
    outline: "none",
    transition: "all 0.2s",
    boxSizing: "border-box",
  },
  passwordWrapper: {
    position: "relative",
  },
  visibilityButton: {
    position: "absolute",
    right: "16px",
    top: "50%",
    transform: "translateY(-50%)",
    background: "none",
    border: "none",
    color: "#4e978b",
    cursor: "pointer",
    padding: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  visibilityIcon: {
    fontFamily: "'Material Symbols Outlined'",
    fontSize: "20px",
  },
  submitButton: {
    width: "100%",
    height: "56px",
    backgroundColor: "#19e6c4",
    color: "#0e1b19",
    borderRadius: "12px",
    fontSize: "16px",
    fontWeight: "700",
    border: "none",
    cursor: "pointer",
    transition: "all 0.2s",
    boxShadow: "0 4px 14px rgba(25, 230, 196, 0.2)",
  },
  submitButtonDisabled: {
    backgroundColor: "#9ca3af",
    cursor: "not-allowed",
  },

  // Messages
  error: {
    padding: "12px 16px",
    backgroundColor: "#fef2f2",
    color: "#dc2626",
    borderRadius: "12px",
    border: "1px solid #fecaca",
    fontSize: "14px",
    marginBottom: "16px",
  },
  success: {
    padding: "12px 16px",
    backgroundColor: "#f0fdf4",
    color: "#166534",
    borderRadius: "12px",
    border: "1px solid #bbf7d0",
    fontSize: "14px",
    marginBottom: "16px",
  },

  // Footer in Card
  footer: {
    marginTop: "32px",
    textAlign: "center",
  },
  footerText: {
    fontSize: "14px",
    color: "#4e978b",
    margin: 0,
  },
  createAccountLink: {
    background: "none",
    border: "none",
    color: "#19e6c4",
    fontSize: "14px",
    fontWeight: "700",
    cursor: "pointer",
    textDecoration: "none",
    marginLeft: "4px",
  },

  // Page Footer
  pageFooter: {
    width: "100%",
    textAlign: "center",
    padding: "24px",
    fontSize: "12px",
    color: "#4e978b",
    zIndex: 10,
  },
  footerLinks: {
    marginTop: "8px",
    display: "flex",
    gap: "16px",
    justifyContent: "center",
  },
  footerLink: {
    color: "#4e978b",
    textDecoration: "none",
    transition: "color 0.2s",
  },
};

// Add Google Fonts
const link1 = document.createElement("link");
link1.href =
  "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap";
link1.rel = "stylesheet";
document.head.appendChild(link1);

const link2 = document.createElement("link");
link2.href =
  "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght@100..700,0..1&display=swap";
link2.rel = "stylesheet";
document.head.appendChild(link2);

// Add CSS for focus states and hover effects
const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = `
  input:focus {
    border-color: #19e6c4 !important;
    box-shadow: 0 0 0 3px rgba(25, 230, 196, 0.1) !important;
  }
  
  button:not(:disabled):hover {
    transform: scale(0.98);
  }
  
  a:hover {
    text-decoration: underline;
  }
`;
document.head.appendChild(styleSheet);

export default Login;