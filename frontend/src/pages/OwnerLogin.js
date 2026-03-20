import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function OwnerLogin() {
  const [form, setForm] = useState({
    email: "",
    password: "",
    rememberMe: false,
  });
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm({ ...form, [name]: type === "checkbox" ? checked : value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage("Loading...");

    try {
      const res = await fetch("http://localhost:5000/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: form.email,
          password: form.password,
          role: "owner",
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "Login failed");

      localStorage.setItem("token", data.token);
      localStorage.setItem("user", JSON.stringify(data.user));

      setMessage(`Welcome ${data.user.name}! Logged in successfully.`);
      setTimeout(() => {
        navigate("/owner/dashboard");
      }, 1500);
    } catch (err) {
      setMessage(err.message || "Login failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.body}>
      {/* Abstract background decorations */}
      <div style={styles.backgroundDecorations}>
        <div style={styles.decorationTopLeft}></div>
        <div style={styles.decorationBottomRight}></div>
      </div>

      {/* Main Content Area */}
      <main style={styles.main}>
        {/* Login Card Container */}
        <div style={styles.card}>
          {/* Headline and Body Text */}
          <div style={styles.cardHeader}>
            <div style={styles.iconCircle}>
              <span style={styles.materialIcon}>domain_verification</span>
            </div>
            <h1 style={styles.title}>Partner Dashboard Login</h1>
            <p style={styles.subtitle}>
              Manage your property and revenue with ease.
            </p>
          </div>

          {/* Message Display */}
          {message && (
            <div
              style={
                message.includes("Welcome") || message.includes("successfully")
                  ? styles.success
                  : message === "Loading..."
                    ? styles.loading
                    : styles.error
              }
            >
              {message}
            </div>
          )}

          <form style={styles.form} onSubmit={handleSubmit}>
            {/* Business Email Field */}
            <div style={styles.inputGroup}>
              <label style={styles.label}>Business Email</label>
              <div style={styles.inputWrapper}>
                <span style={styles.inputIcon}>mail</span>
                <input
                  type="email"
                  name="email"
                  placeholder="e.g. owner@parking-corp.com"
                  value={form.email}
                  onChange={handleChange}
                  style={styles.input}
                  required
                />
              </div>
            </div>

            {/* Password Field */}
            <div style={styles.inputGroup}>
              <div style={styles.passwordLabelRow}>
                <label style={styles.label}>Password</label>
                <a href="#" style={styles.forgotPassword}>
                  Forgot Password?
                </a>
              </div>
              <div style={styles.inputWrapper}>
                <span style={styles.inputIcon}>lock</span>
                <input
                  type="password"
                  name="password"
                  placeholder="Enter your password"
                  value={form.password}
                  onChange={handleChange}
                  style={styles.input}
                  required
                />
              </div>
            </div>

            {/* Login Button */}
            <button
              type="submit"
              disabled={isLoading}
              style={{
                ...styles.submitButton,
                ...(isLoading ? styles.submitButtonDisabled : {}),
              }}
            >
              <span>{isLoading ? "Signing in..." : "Login to Dashboard"}</span>
              {!isLoading && (
                <span style={styles.arrowIcon}>arrow_forward</span>
              )}
            </button>
          </form>

          {/* Switch Role & Security Badge */}
          <div style={styles.footer}>
            <button
              onClick={() => navigate("/login")}
              style={styles.switchButton}
            >
              <span style={styles.switchIcon}>swap_horiz</span>
              Switch to User Login
            </button>
          </div>
        </div>

        {/* Footer Info */}
        <footer style={styles.pageFooter}>
          <div style={styles.footerLinks}>
            <a href="#" style={styles.footerLink}>
              Privacy Policy
            </a>
            <a href="#" style={styles.footerLink}>
              Terms of Service
            </a>
            <a href="#" style={styles.footerLink}>
              Contact Support
            </a>
          </div>
          <p style={styles.footerCopyright}>
            © 2024 NExtSPot Inc. Professional Parking Solutions.
          </p>
        </footer>
      </main>
    </div>
  );
}

const styles = {
  body: {
    fontFamily:
      "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    backgroundColor: "#f6f6f8",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    position: "relative",
  },

  // Background decorations
  backgroundDecorations: {
    position: "fixed",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: 0,
    pointerEvents: "none",
    opacity: 0.5,
    overflow: "hidden",
  },
  decorationTopLeft: {
    position: "absolute",
    top: "-96px",
    left: "-96px",
    width: "384px",
    height: "384px",
    background: "rgba(48, 110, 232, 0.1)",
    borderRadius: "50%",
    filter: "blur(100px)",
  },
  decorationBottomRight: {
    position: "absolute",
    bottom: 0,
    right: 0,
    width: "500px",
    height: "500px",
    background: "rgba(48, 110, 232, 0.05)",
    borderRadius: "50%",
    filter: "blur(120px)",
  },

  // Header
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottom: "1px solid #e7ebf3",
    padding: "12px 40px",
    backgroundColor: "#ffffff",
    position: "relative",
    zIndex: 10,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    color: "#306ee8",
  },
  logoContainer: {
    width: "32px",
    height: "32px",
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
    color: "#0e121b",
  },

  // Main Content
  main: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "48px 16px",
    position: "relative",
    zIndex: 10,
  },
  card: {
    width: "100%",
    maxWidth: "480px",
    backgroundColor: "#ffffff",
    border: "1px solid #e7ebf3",
    borderRadius: "12px",
    boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
    padding: "40px",
  },
  cardHeader: {
    marginBottom: "32px",
    textAlign: "center",
  },
  iconCircle: {
    display: "flex",
    justifyContent: "center",
    marginBottom: "16px",
  },
  materialIcon: {
    fontFamily: "'Material Symbols Outlined'",
    fontSize: "36px",
    color: "#306ee8",
    fontWeight: "normal",
    fontStyle: "normal",
    display: "inline-block",
    lineHeight: 1,
  },
  title: {
    fontSize: "28px",
    fontWeight: "700",
    color: "#0e121b",
    margin: "0 0 8px 0",
    lineHeight: 1.2,
  },
  subtitle: {
    fontSize: "16px",
    color: "#4e6797",
    margin: 0,
  },

  // Messages
  error: {
    padding: "12px",
    backgroundColor: "#fef2f2",
    color: "#b91c1c",
    borderRadius: "8px",
    border: "1px solid #fecaca",
    fontSize: "14px",
    marginBottom: "24px",
  },
  success: {
    padding: "12px",
    backgroundColor: "#f0fdf4",
    color: "#166534",
    borderRadius: "8px",
    border: "1px solid #bbf7d0",
    fontSize: "14px",
    marginBottom: "24px",
  },
  loading: {
    padding: "12px",
    backgroundColor: "#eff6ff",
    color: "#1d4ed8",
    borderRadius: "8px",
    border: "1px solid #c7d2fe",
    fontSize: "14px",
    marginBottom: "24px",
  },

  // Form
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "24px",
  },
  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  label: {
    fontSize: "14px",
    fontWeight: "600",
    color: "#0e121b",
  },
  passwordLabelRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  forgotPassword: {
    fontSize: "12px",
    fontWeight: "600",
    color: "#306ee8",
    textDecoration: "none",
  },
  inputWrapper: {
    position: "relative",
  },
  inputIcon: {
    fontFamily: "'Material Symbols Outlined'",
    position: "absolute",
    left: "16px",
    top: "50%",
    transform: "translateY(-50%)",
    color: "#4e6797",
    fontSize: "20px",
    pointerEvents: "none",
  },
  input: {
    width: "100%",
    height: "56px",
    paddingLeft: "48px",
    paddingRight: "16px",
    borderRadius: "8px",
    border: "1px solid #d0d7e7",
    backgroundColor: "#f8f9fc",
    fontSize: "16px",
    color: "#0e121b",
    outline: "none",
    transition: "all 0.2s",
    boxSizing: "border-box",
  },
  rememberMeContainer: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    paddingLeft: "4px",
  },
  checkbox: {
    width: "16px",
    height: "16px",
    borderRadius: "4px",
    border: "1px solid #d0d7e7",
    cursor: "pointer",
  },
  checkboxLabel: {
    fontSize: "12px",
    color: "#4e6797",
    cursor: "pointer",
  },
  submitButton: {
    width: "100%",
    height: "56px",
    backgroundColor: "#306ee8",
    color: "#ffffff",
    borderRadius: "8px",
    fontWeight: "700",
    fontSize: "16px",
    border: "none",
    cursor: "pointer",
    transition: "all 0.2s",
    boxShadow: "0 4px 14px rgba(48, 110, 232, 0.2)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "8px",
  },
  submitButtonDisabled: {
    backgroundColor: "#9ca3af",
    cursor: "not-allowed",
  },
  arrowIcon: {
    fontFamily: "'Material Symbols Outlined'",
    fontSize: "20px",
  },

  // Footer in Card
  footer: {
    marginTop: "32px",
    paddingTop: "32px",
    borderTop: "1px solid #e7ebf3",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "24px",
  },
  switchButton: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: "#4e6797",
    fontSize: "14px",
    fontWeight: "500",
    background: "none",
    border: "none",
    cursor: "pointer",
    transition: "color 0.2s",
  },
  switchIcon: {
    fontFamily: "'Material Symbols Outlined'",
    fontSize: "18px",
  },
  securityBadge: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    backgroundColor: "#f8f9fc",
    padding: "8px 16px",
    borderRadius: "9999px",
    border: "1px solid #d0d7e7",
  },
  securityIcon: {
    fontFamily: "'Material Symbols Outlined'",
    fontSize: "16px",
    color: "#16a34a",
  },
  securityText: {
    fontSize: "10px",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    fontWeight: "700",
    color: "#4e6797",
  },

  // Page Footer
  pageFooter: {
    marginTop: "32px",
    textAlign: "center",
    color: "#4e6797",
    fontSize: "12px",
  },
  footerLinks: {
    display: "flex",
    justifyContent: "center",
    gap: "16px",
    marginBottom: "8px",
  },
  footerLink: {
    color: "#4e6797",
    textDecoration: "none",
    transition: "text-decoration 0.2s",
  },
  footerCopyright: {
    margin: 0,
  },
};

// Add Google Fonts and Material Icons
const link1 = document.createElement("link");
link1.href =
  "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap";
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
    border-color: #306ee8 !important;
    box-shadow: 0 0 0 3px rgba(48, 110, 232, 0.1) !important;
  }
  
  button:not(:disabled):hover {
    opacity: 0.9;
  }
  
  a:hover {
    text-decoration: underline;
  }
  
  .material-symbols-outlined {
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  }
`;
document.head.appendChild(styleSheet);

export default OwnerLogin;
