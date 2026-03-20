import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function Register() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    role: "user",
    businessName: "",
    taxId: "",
  });
  const [message, setMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value });
  };

  const handleRoleChange = (role) => {
    setForm({ ...form, role });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setMessage("Loading...");

    try {
      const res = await fetch("http://localhost:5000/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || "Registration failed");
      setMessage(data.message);

      // Redirect to appropriate login page after successful registration
      setTimeout(() => {
        if (form.role === "owner") {
          navigate("/owner-login");
        } else {
          navigate("/login");
        }
      }, 2000);
    } catch (err) {
      setMessage(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.body}>
      {/* Header / TopNavBar */}
      

      {/* Main Registration Area */}
      <main style={styles.main}>
        <div style={styles.card}>
          {/* Headline & Meta Text */}
          <div style={styles.cardHeader}>
            <h1 style={styles.title}>Join NExtSPot</h1>
            <p style={styles.subtitle}>Start your journey with us today.</p>
          </div>

          {/* Segmented Buttons - Role Selection */}
          <div style={styles.segmentedContainer}>
            <div style={styles.segmentedButtons}>
              <label
                style={{
                  ...styles.segmentButton,
                  ...(form.role === "user" ? styles.segmentButtonActive : {}),
                }}
              >
                <span style={styles.segmentButtonText}>I want to Park</span>
                <input
                  type="radio"
                  name="user-role"
                  value="user"
                  checked={form.role === "user"}
                  onChange={() => handleRoleChange("user")}
                  style={styles.hiddenRadio}
                />
              </label>
              <label
                style={{
                  ...styles.segmentButton,
                  ...(form.role === "owner" ? styles.segmentButtonActive : {}),
                }}
              >
                <span style={styles.segmentButtonText}>
                  I own a Parking Spot
                </span>
                <input
                  type="radio"
                  name="user-role"
                  value="owner"
                  checked={form.role === "owner"}
                  onChange={() => handleRoleChange("owner")}
                  style={styles.hiddenRadio}
                />
              </label>
            </div>
          </div>

          {/* Message Display */}
          {message && (
            <div
              style={
                message.includes("success") ||
                message.includes("Success") ||
                message.includes("created")
                  ? styles.success
                  : message === "Loading..."
                    ? styles.loading
                    : styles.error
              }
            >
              {message}
            </div>
          )}

          {/* Registration Form */}
          <form style={styles.form} onSubmit={handleSubmit}>
            {/* Full Name */}
            <div style={styles.inputGroup}>
              <label style={styles.label}>Full Name</label>
              <input
                type="text"
                name="name"
                placeholder="Jane Doe"
                value={form.name}
                onChange={handleChange}
                style={styles.input}
                required
              />
            </div>

            {/* Email Address */}
            <div style={styles.inputGroup}>
              <label style={styles.label}>Email Address</label>
              <input
                type="email"
                name="email"
                placeholder="jane@example.com"
                value={form.email}
                onChange={handleChange}
                style={styles.input}
                required
              />
            </div>


            {/* Password */}
            <div style={styles.inputGroup}>
              <label style={styles.label}>Password</label>
              <div style={styles.passwordWrapper}>
                <input
                  type={showPassword ? "text" : "password"}
                  name="password"
                  placeholder="Min. 8 characters"
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
                  <span style={styles.materialIcon}>
                    {showPassword ? "visibility_off" : "visibility"}
                  </span>
                </button>
              </div>
            </div>

            {/* Join Button */}
            <button
              type="submit"
              disabled={isLoading}
              style={{
                ...styles.submitButton,
                ...(isLoading ? styles.submitButtonDisabled : {}),
              }}
            >
              {isLoading ? "Creating Account..." : "Join NExtSPot"}
            </button>
          </form>

          {/* Footer Links */}
          <div style={styles.footer}>
            <p style={styles.footerText}>
              Already have an account?{" "}
              <button
                onClick={() => navigate("/login")}
                style={styles.loginLink}
              >
                Log in
              </button>
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={styles.pageFooter}>
        <div style={styles.footerLinks}>
          <a href="#" style={styles.footerLink}>
            Terms of Service
          </a>
          <a href="#" style={styles.footerLink}>
            Privacy Policy
          </a>
          <a href="#" style={styles.footerLink}>
            Cookie Policy
          </a>
        </div>
        <p style={styles.footerCopyright}>
          © 2024 NExtSPot Inc. All rights reserved.
        </p>
      </footer>
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
    color: "#0d121b",
  },

  // Header
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    borderBottom: "1px solid #e7ebf3",
    padding: "12px 40px",
    backgroundColor: "rgba(255, 255, 255, 0.8)",
    backdropFilter: "blur(12px)",
    position: "sticky",
    top: 0,
    zIndex: 50,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "16px",
    color: "#0d121b",
  },
  logoContainer: {
    width: "24px",
    height: "24px",
    color: "#135bec",
  },
  logo: {
    width: "100%",
    height: "100%",
  },
  headerTitle: {
    fontSize: "20px",
    fontWeight: "700",
    margin: 0,
    letterSpacing: "-0.015em",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "32px",
  },
  loginButton: {
    minWidth: "84px",
    padding: "10px 16px",
    backgroundColor: "#135bec",
    color: "#ffffff",
    borderRadius: "12px",
    fontSize: "14px",
    fontWeight: "700",
    border: "none",
    cursor: "pointer",
    transition: "background-color 0.2s",
  },

  // Main Content
  main: {
    flex: 1,
    background: "linear-gradient(135deg, #f0f7ff 0%, #e0f2fe 100%)",
    padding: "48px 16px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  card: {
    width: "100%",
    maxWidth: "520px",
    backgroundColor: "#ffffff",
    borderRadius: "12px",
    boxShadow:
      "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04), 0 0 0 1px rgba(147, 197, 253, 0.2)",
    padding: "48px",
  },
  cardHeader: {
    textAlign: "center",
    marginBottom: "32px",
  },
  title: {
    fontSize: "32px",
    fontWeight: "700",
    color: "#0d121b",
    margin: "0 0 8px 0",
    letterSpacing: "-0.025em",
    lineHeight: 1.2,
  },
  subtitle: {
    fontSize: "16px",
    color: "#4c669a",
    margin: 0,
  },

  // Segmented Buttons
  segmentedContainer: {
    marginBottom: "32px",
  },
  segmentedButtons: {
    display: "flex",
    height: "48px",
    backgroundColor: "#e7ebf3",
    borderRadius: "12px",
    padding: "4px",
  },
  segmentButton: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    borderRadius: "8px",
    padding: "0 8px",
    transition: "all 0.2s",
    color: "#4c669a",
    fontSize: "14px",
    fontWeight: "600",
  },
  segmentButtonActive: {
    backgroundColor: "#ffffff",
    boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
    color: "#135bec",
  },
  segmentButtonText: {
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  hiddenRadio: {
    position: "absolute",
    opacity: 0,
    width: 0,
    height: 0,
  },

  // Messages
  error: {
    padding: "12px",
    backgroundColor: "#fef2f2",
    color: "#b91c1c",
    borderRadius: "12px",
    border: "1px solid #fecaca",
    fontSize: "14px",
    marginBottom: "20px",
  },
  success: {
    padding: "12px",
    backgroundColor: "#f0fdf4",
    color: "#166534",
    borderRadius: "12px",
    border: "1px solid #bbf7d0",
    fontSize: "14px",
    marginBottom: "20px",
  },
  loading: {
    padding: "12px",
    backgroundColor: "#eff6ff",
    color: "#1d4ed8",
    borderRadius: "12px",
    border: "1px solid #c7d2fe",
    fontSize: "14px",
    marginBottom: "20px",
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
    gap: "4px",
  },
  label: {
    fontSize: "14px",
    fontWeight: "600",
    color: "#0d121b",
    marginBottom: "8px",
  },
  input: {
    width: "100%",
    height: "56px",
    padding: "15px",
    borderRadius: "12px",
    border: "1px solid #cfd7e7",
    backgroundColor: "#ffffff",
    fontSize: "16px",
    color: "#0d121b",
    outline: "none",
    transition: "all 0.2s",
    boxSizing: "border-box",
  },
  ownerFields: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
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
    color: "#9ca3af",
    cursor: "pointer",
    padding: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  materialIcon: {
    fontFamily: "'Material Symbols Outlined'",
    fontSize: "20px",
  },
  submitButton: {
    width: "100%",
    height: "56px",
    backgroundColor: "#135bec",
    color: "#ffffff",
    borderRadius: "12px",
    fontSize: "16px",
    fontWeight: "700",
    border: "none",
    cursor: "pointer",
    transition: "all 0.2s",
    boxShadow: "0 10px 15px -3px rgba(19, 91, 236, 0.2)",
  },
  submitButtonDisabled: {
    backgroundColor: "#9ca3af",
    cursor: "not-allowed",
  },

  // Footer in Card
  footer: {
    marginTop: "32px",
    textAlign: "center",
  },
  footerText: {
    fontSize: "14px",
    color: "#4c669a",
    margin: 0,
  },
  loginLink: {
    color: "#135bec",
    fontWeight: "700",
    background: "none",
    border: "none",
    cursor: "pointer",
    textDecoration: "none",
  },

  // Page Footer
  pageFooter: {
    backgroundColor: "#ffffff",
    padding: "32px 40px",
    borderTop: "1px solid #e7ebf3",
    textAlign: "center",
  },
  footerLinks: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: "32px",
    marginBottom: "16px",
    flexWrap: "wrap",
  },
  footerLink: {
    fontSize: "12px",
    color: "#4c669a",
    textDecoration: "none",
    fontWeight: "500",
    transition: "color 0.2s",
  },
  footerCopyright: {
    fontSize: "12px",
    color: "#4c669a",
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
  "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap";
link2.rel = "stylesheet";
document.head.appendChild(link2);

// Add CSS for focus states and hover effects
const styleSheet = document.createElement("style");
styleSheet.type = "text/css";
styleSheet.innerText = `
  input:focus {
    border-color: #135bec !important;
    box-shadow: 0 0 0 3px rgba(19, 91, 236, 0.1) !important;
  }
  
  button:not(:disabled):hover {
    opacity: 0.9;
  }
  
  a:hover {
    color: #135bec !important;
  }
`;
document.head.appendChild(styleSheet);

export default Register;