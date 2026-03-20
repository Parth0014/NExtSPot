import React, { useState, useEffect } from "react";
import {
  MapPin,
  CheckCircle,
  Wifi,
  Calendar,
  TrendingUp,
  Globe,
  Share2,
  Mail,
} from "lucide-react";

function ParkEaseHome() {
  const [scrollY, setScrollY] = useState(0);
  const [windowWidth, setWindowWidth] = useState(
    typeof window !== "undefined" ? window.innerWidth : 1024,
  );

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    const handleResize = () => {
      setWindowWidth(window.innerWidth);
    };

    window.addEventListener("scroll", handleScroll);
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <div style={styles.container}>
      {/* Navigation */}
      <header
        style={{
          ...styles.header,
          backgroundColor:
            scrollY > 50 ? "rgba(246,246,248,0.95)" : "rgba(246,246,248,0.8)",
          borderBottom:
            scrollY > 50
              ? "1px solid rgba(231,235,243,0.95)"
              : "1px solid rgba(231,235,243,0.6)",
          boxShadow: scrollY > 50 ? "0 6px 30px rgba(13,18,27,0.06)" : "none",
          padding: windowWidth >= 1024 ? "1rem 10rem" : "1rem 1.5rem",
        }}
        className="header-nav"
      >
        <div style={styles.nav}>
          <div style={styles.logo}>
            <div style={styles.logoIcon}>
              <MapPin size={24} color="#ffffff" />
            </div>
            <h2 style={styles.logoText}>NExtSPot</h2>
          </div>

          <nav style={styles.navLinks} className="nav-links">
            <a href="#features" style={styles.navLink}>
              Features
            </a>
            <a href="#solutions" style={styles.navLink}>
              Solutions
            </a>
            <a href="#partners" style={styles.navLink}>
              Partners
            </a>
            <a href="#pricing" style={styles.navLink}>
              Pricing
            </a>
          </nav>

          <div style={styles.navRight}>
            <button
              onClick={() => (window.location.href = "/login")}
              style={styles.getStartedBtn}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = "rgba(19, 91, 236, 0.9)";
                e.target.style.transform = "translateY(-2px)";
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = "#135bec";
                e.target.style.transform = "translateY(0)";
              }}
            >
              Get Started
            </button>
          </div>
        </div>
      </header>

      <main style={styles.main}>
        {/* Hero Section */}
        <section style={styles.hero} className="hero-section">
          <div style={styles.heroContent} className="hero-content">
            <div style={styles.heroLeft} data-animate="fade-left">
              <div style={styles.heroTextContent}>
                <span style={styles.badge}>
                  <CheckCircle size={14} color="#135bec" />
                  TRUSTED BY ME AND MY TEAM 🙌😊
                </span>

                <h1 style={styles.heroTitle} className="hero-title">
                  Parking Management,{" "}
                  <span style={styles.heroTitleAccent}>Simplified.</span>
                </h1>

                <p style={styles.heroDescription}>
                  The all-in-one platform for effortless booking and intelligent
                  lot management. Scale your business with data-driven insights.
                </p>
              </div>

              <div style={styles.heroButtons}>
                <button
                  onClick={() => (window.location.href = "/login")}
                  style={styles.primaryButton}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow =
                      "0 12px 30px rgba(19, 91, 236, 0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow =
                      "0 8px 25px rgba(19, 91, 236, 0.25)";
                  }}
                >
                  Book a Spot
                </button>

                <button
                  onClick={() => (window.location.href = "/owner-login")}
                  style={styles.secondaryButton}
                  onMouseEnter={(e) => {
                    e.target.style.backgroundColor = "#f8f9fc";
                    e.target.style.transform = "translateY(-2px)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.backgroundColor = "#ffffff";
                    e.target.style.transform = "translateY(0)";
                  }}
                >
                  Partner with Us
                </button>
              </div>

              <div style={styles.socialProof}>
                <div style={styles.avatarGroup}>
                  <div
                    style={{
                      ...styles.avatar,
                      backgroundImage:
                        "url('https://lh3.googleusercontent.com/aida-public/AB6AXuDH6R8EH17jasIWvEMtGtYauA5ffMWKSQ5EvsKmZw15kTkzhqFCxQY6WJoXbHNtdzrkp6spHryWbRuANZwS15-8grKcv426Y-iLB7LxdkNsyWHvPfQLRDQx1PNTIwrf8-k1vjxwq0deMPHm34dhtN45hu1--PD_S3Ie5JYWihzhtgcU67Igg-WNuJZBh3r0oIy7Yva3q0EMJ9ItPBCllXYkbdcdBl68frNqtxR3qkjKFCRAQyTALr5n0m1mlc01ahOpH56-iJmjFqnL')",
                    }}
                  ></div>
                  <div
                    style={{
                      ...styles.avatar,
                      backgroundImage:
                        "url('https://lh3.googleusercontent.com/aida-public/AB6AXuAf-XuteIY430Tb9XGkuoO16G6m2d6ktX5rfjX0rO5UruT2nRoZpKW384GLTVIMSJ5-Z4IdlEr7NoIQE4hm-4z1jBPwL_LyQujULUyb_YYAedXCCu3RIIO_JEo5FmQuuHwPn3Kdwr5WIjHXzFF3gxPR3F0fgX8luIc95UQaQ08nDyJHlZTuepastN0VlJ7NHLMD3psv-VRwbXhmk13jUMJKVsVZA5S2BCXomBlv7qalB_5wdt-QY3clC8602V3Z5-3A55zSEIOMKnp2')",
                    }}
                  ></div>
                  <div
                    style={{
                      ...styles.avatar,
                      backgroundImage:
                        "url('https://lh3.googleusercontent.com/aida-public/AB6AXuBO68mFf67uHoiXK29R-OWaBIPUlmrqcjhnAC395aoJ373KTJsR__z2FI6iBzc2tba3HRF8sVQVVgY23gBscRarlDGqom6x7ny51rYhJvcExoatqo1PdYirLnP3gHOYysGK7Lnvch0eVp-WjYKEjmF1MK_6HME6-TfV2G-Pw1ftCmcVvPf9sJD0L2zIOh5oIM-R44_0fBF2ZbWw5Wt-aBt6qxAbAiifSfU6FSBHqhiSVIW2i3JY9tYMDnfJKztFF4bTn6B057kKGKId')",
                    }}
                  ></div>
                </div>
                <p style={styles.socialProofText}>Join Us</p>
              </div>
            </div>

            <div style={styles.heroRight} data-animate="fade-right">
              <div style={styles.imageWrapper}>
                <div style={styles.glowTopRight}></div>
                <div style={styles.glowBottomLeft}></div>
                <div
                  style={{
                    ...styles.heroImage,
                    backgroundImage:
                      "url('https://lh3.googleusercontent.com/aida-public/AB6AXuA0OYqTuRtfzdJcgtjvZO6-rvN_Bp5fpWk3VQ-t_g8VhPE1w9B0dtqSWJ1SH7heWPC791rdg9I3eQZe0e_NtAQ26jdi1rBr2BYPrYIGYklHgmgPPO4FVb0ElHCYQnWQ9I6UQApVU12iFzcLTSGZQfYBJQf-g52pSUSRDLAVDLy1SF2KW0E6G6onVjqWajcm9xIKNmmjGb5Up-vK8iKSb65ii7zCwFcLPA6uVlySJJK8HPQOFq_wx8Nt8KTAWy-vciKRMZvdRe22I9oH')",
                    transform: `translateY(${scrollY * 0.05}px)`,
                  }}
                >
                  <div style={styles.floatingCard}>
                    <div style={styles.cardIcon}>
                      <CheckCircle size={20} color="#10b981" />
                    </div>
                    <div>
                      <div style={styles.cardTitle}>Available Spots</div>
                      <div style={styles.cardSubtitle}>128 Spaces</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Trust Badges Section */}
        <section style={styles.trustBadges}>
          <p style={styles.trustText}>
            TRUSTED BY 500+ PARKING FACILITIES NATIONWIDE
          </p>
          <div style={styles.brandLogos}>
            <div style={styles.brandLogo}>CITYPARK</div>
            <div style={styles.brandLogo}>METROGO</div>
            <div style={styles.brandLogo}>URBAN_LOT</div>
            <div style={styles.brandLogo}>SAFEHUB</div>
            <div style={styles.brandLogo}>EZ_STAY</div>
          </div>
        </section>

        {/* Features Section */}
        <section
          id="features"
          style={styles.features}
          className="features-section"
        >
          <div style={styles.sectionHeader}>
            <p style={styles.sectionLabel}>THE FUTURE OF PARKING</p>
            <h2 style={styles.sectionTitle} className="section-title">
              Modern Solutions for Modern Facilities
            </h2>
            <p style={styles.sectionDescription}>
              Our platform uses cutting-edge technology to streamline parking
              operations for both owners and drivers.
            </p>
          </div>

          <div style={styles.featuresGrid} className="features-grid">
            {/* Feature 1 */}
            <div style={styles.featureCard} data-animate="fade-up">
              <div style={styles.featureIconWrapper}>
                <Wifi size={32} color="#135bec" />
              </div>
              <h3 style={styles.featureTitle}>Real-time AI Detection</h3>
              <p style={styles.featureDescription}>
                Know your occupancy status every second with our advanced
                computer vision tech. Zero hardware maintenance required.
              </p>
              <div
                style={{
                  ...styles.featureImage,
                  backgroundImage:
                    "url('https://lh3.googleusercontent.com/aida-public/AB6AXuB5fYHeImBBshDJSBJQFxFHXUvStvKUhsAixAd627F3lg5hR4AZZ14urMT_l2XI-6VaoT0RCPeBtEFWLPFLrJsdarXoolAJ9Vnp0T1HYOHZW290oL1yFpjUGSnZkWB1zlDHKYzLwKKlS6o30c1jjBpxy8bendnFn6GWtWi0JkFpan8znCjpgf14y_-GjlsY3eMGSvHiQFDYXkuFXFVJM8oD2ycFbMXj0vxhDLMR_Q5hnoWK7Q5m2dDgLUWZk6jr8K8mdUVjSVSeJmlg')",
                }}
              ></div>
            </div>

            {/* Feature 2 */}
            <div style={styles.featureCard} data-animate="fade-up">
              <div style={styles.featureIconWrapper}>
                <Calendar size={32} color="#135bec" />
              </div>
              <h3 style={styles.featureTitle}>Easy Reservations</h3>
              <p style={styles.featureDescription}>
                Find and pay for your spot in under 30 seconds via our secure
                mobile portal. Guaranteed spot upon arrival.
              </p>
              <div
                style={{
                  ...styles.featureImage,
                  backgroundImage:
                    "url('https://lh3.googleusercontent.com/aida-public/AB6AXuBFPrvg-vosDAALe1sTb8cxCT6G7MqbidvwrigUoBt-dUlC85KJeqqMVeNrCDBrWiU4eNy1jcW4SLyh3JIpvqxM6Mx4WZeHnEJrinnBJ2S7FcYlh65nZPOB_Nf-DLYgYXlw1NiKHBijApLz9I83TJCKfy9QQb-TBR-TyxlhI_NvL-hJTw0UrhvvUO-OYXf_u2TS-WkOqXxHlMuakNeIPWBUZyINjgWPEWinD7xLdggyF2vIOKJ7VT0O5_fpEfbZMCrIQf-mrCzO26uL')",
                }}
              ></div>
            </div>

            {/* Feature 3 */}
            <div style={styles.featureCard} data-animate="fade-up">
              <div style={styles.featureIconWrapper}>
                <TrendingUp size={32} color="#135bec" />
              </div>
              <h3 style={styles.featureTitle}>Owner Analytics</h3>
              <p style={styles.featureDescription}>
                Track revenue and facility readiness to maximize your facility's
                daily revenue. Track performance from any device.
              </p>
              <div
                style={{
                  ...styles.featureImage,
                  backgroundImage:
                    "url('https://lh3.googleusercontent.com/aida-public/AB6AXuDmLqX5osuw2q4S-ZZxUnUKuz397TZOmyCRlrtY8fC8TvlSZ1npGZVUDrv6_cMXBz7FF_0S48qsDZ8FY9S_0XPCKhMQEtZjAQlmFVUlTyyAbLVCi8Zc4e0LyWB3lpVpZr84JWyiSWVpQnxLyWuEXBEew8n8rUqve4wmYbcpuak6eqeI5oyVSXHfSvqOa_jiL9YxixR9NDgNtO4n0hdTdkLYf6VbnN6ulit8bjjZdqkkTt7V1qHQtGS7bMiyk2FWNY_K4Py8bwG2SChw')",
                }}
              ></div>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section style={styles.stats}>
          <div style={styles.skewedBg}></div>
          <div style={styles.statsContent} className="stats-content">
            <div style={styles.statItem}>
              <p style={styles.statNumber} className="stat-number">
                99.9%
              </p>
              <p style={styles.statLabel}>SYSTEM UPTIME</p>
            </div>
            <div style={styles.statItem}>
              <p style={styles.statNumber} className="stat-number">
                500k+
              </p>
              <p style={styles.statLabel}>SPOTS MANAGED</p>
            </div>
            <div style={styles.statItem}>
              <p style={styles.statNumber} className="stat-number">
                2min
              </p>
              <p style={styles.statLabel}>AVG. SUPPORT RESPONSE</p>
            </div>
            <div style={styles.statItem}>
              <p style={styles.statNumber} className="stat-number">
                15%
              </p>
              <p style={styles.statLabel}>REVENUE INCREASE</p>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section style={styles.cta} className="cta-section">
          <div style={styles.ctaCard}>
            <div style={styles.ctaCircle}></div>
            <div style={styles.ctaContent}>
              <h2 style={styles.ctaTitle} className="cta-title">
                Ready to revolutionize your parking experience?
              </h2>
              <p style={styles.ctaDescription}>
                Whether you're a driver looking for a stress-free commute or an
                owner seeking to optimize operations, we've got you covered.
              </p>
              <div style={styles.ctaButtons}>
                <button
                  onClick={() => (window.location.href = "/login")}
                  style={styles.ctaPrimaryButton}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "scale(1.02)";
                    e.target.style.boxShadow =
                      "0 12px 30px rgba(19, 91, 236, 0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "scale(1)";
                    e.target.style.boxShadow =
                      "0 8px 25px rgba(19, 91, 236, 0.2)";
                  }}
                >
                  Start Free Trial
                </button>
                <button
                  onClick={() => (window.location.href = "/owner-login")}
                  style={styles.ctaSecondaryButton}
                  onMouseEnter={(e) => {
                    e.target.style.backgroundColor = "#0d121b";
                    e.target.style.color = "#ffffff";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.backgroundColor = "transparent";
                    e.target.style.color = "#0d121b";
                  }}
                >
                  Talk to Sales
                </button>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer style={styles.footer} className="footer-section">
        <div style={styles.footerContent} className="footer-content">
          <div style={styles.footerColumn}>
            <div style={styles.footerLogo}>
              <div style={styles.footerLogoIcon}>
                <MapPin size={20} color="#ffffff" />
              </div>
              <h2 style={styles.footerLogoText}>NExtSPot</h2>
            </div>
            <p style={styles.footerDescription}>
              Making urban mobility seamless through intelligent parking
              technology and user-centric design since 2018.
            </p>
            <div style={styles.socialLinks}>
              <a href="#" style={styles.socialLink}>
                <Globe size={20} />
              </a>
              <a href="#" style={styles.socialLink}>
                <Share2 size={20} />
              </a>
              <a href="#" style={styles.socialLink}>
                <Mail size={20} />
              </a>
            </div>
          </div>

          <div style={styles.footerColumn}>
            <h3 style={styles.footerColumnTitle}>Product</h3>
            <ul style={styles.footerList}>
              <li>
                <a href="#" style={styles.footerLink}>
                  Spot Finder
                </a>
              </li>
              <li>
                <a href="#" style={styles.footerLink}>
                  Management Portal
                </a>
              </li>
              <li>
                <a href="#" style={styles.footerLink}>
                  API for Developers
                </a>
              </li>
              <li>
                <a href="#" style={styles.footerLink}>
                  Hardware Solutions
                </a>
              </li>
            </ul>
          </div>

          <div style={styles.footerColumn}>
            <h3 style={styles.footerColumnTitle}>Company</h3>
            <ul style={styles.footerList}>
              <li>
                <a href="#" style={styles.footerLink}>
                  About Us
                </a>
              </li>
              <li>
                <a href="#" style={styles.footerLink}>
                  Sustainability
                </a>
              </li>
              <li>
                <a href="#" style={styles.footerLink}>
                  Careers
                </a>
              </li>
              <li>
                <a href="#" style={styles.footerLink}>
                  Press Kit
                </a>
              </li>
            </ul>
          </div>

          <div style={styles.footerColumn}>
            <h3 style={styles.footerColumnTitle}>Stay Updated</h3>
            <p style={styles.newsletterText}>
              Join our newsletter for the latest parking tips and tech.
            </p>
            <form style={styles.newsletterForm}>
              <input
                type="email"
                placeholder="Enter your email"
                style={styles.newsletterInput}
              />
              <button
                type="submit"
                style={styles.newsletterButton}
                onMouseEnter={(e) =>
                  (e.target.style.backgroundColor = "rgba(19, 91, 236, 0.9)")
                }
                onMouseLeave={(e) =>
                  (e.target.style.backgroundColor = "#135bec")
                }
              >
                Subscribe
              </button>
            </form>
          </div>
        </div>

        <div style={styles.footerBottom} className="footer-bottom">
          <p style={styles.copyright}>
            © 2024 NExtSPot Inc. All rights reserved.
          </p>
          <div style={styles.footerLinks}>
            <a href="#" style={styles.footerBottomLink}>
              Privacy Policy
            </a>
            <a href="#" style={styles.footerBottomLink}>
              Terms of Service
            </a>
            <a href="#" style={styles.footerBottomLink}>
              Cookie Settings
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    backgroundColor: "#f6f6f8",
    minHeight: "100vh",
    color: "#0d121b",
    overflowX: "hidden",
  },

  // Header Styles
  header: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    backdropFilter: "blur(12px) saturate(120%)",
    WebkitBackdropFilter: "blur(12px) saturate(120%)",
    borderBottom: "1px solid rgba(231,235,243,0.9)",
    zIndex: 1000,
    transition: "all 0.28s ease",
    padding: "1rem 1.5rem",
  },

  "@media (min-width: 1024px)": {
    header: {
      padding: "1rem 10rem",
    },
  },

  nav: {
    maxWidth: "1200px",
    margin: "0 auto",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  logo: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
  },

  logoIcon: {
    backgroundColor: "#135bec",
    padding: "0.375rem",
    borderRadius: "8px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },

  logoText: {
    fontSize: "1.25rem",
    fontWeight: "700",
    color: "rgba(13,18,27,0.95)",
    margin: 0,
    lineHeight: "1.25",
    letterSpacing: "-0.025em",
  },

  navLinks: {
    display: "flex",
    gap: "2.5rem",
    alignItems: "center",
  },

  navLink: {
    textDecoration: "none",
    color: "rgba(76,102,154,0.85)",
    fontSize: "0.875rem",
    fontWeight: "600",
    transition: "color 0.3s ease",
    cursor: "pointer",
  },

  navRight: {
    display: "flex",
    alignItems: "center",
    gap: "1rem",
  },

  getStartedBtn: {
    minWidth: "100px",
    padding: "0.625rem 1.25rem",
    backgroundColor: "#135bec",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    fontSize: "0.875rem",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.3s ease",
    boxShadow: "0 4px 15px rgba(19, 91, 236, 0.2)",
  },

  userAvatar: {
    width: "40px",
    height: "40px",
    borderRadius: "50%",
    border: "2px solid rgba(19, 91, 236, 0.2)",
    padding: "2px",
  },

  avatarImage: {
    width: "100%",
    height: "100%",
    borderRadius: "50%",
    backgroundSize: "cover",
    backgroundPosition: "center",
  },

  // Main Styles
  main: {
    width: "100%",
  },

  // Hero Section
  hero: {
    padding: "10rem 1.5rem 4rem",
    maxWidth: "1400px",
    margin: "0 auto",
  },

  heroContent: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "3rem",
    alignItems: "center",
  },

  heroLeft: {
    display: "flex",
    flexDirection: "column",
    gap: "2rem",
  },

  heroTextContent: {
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },

  badge: {
    display: "inline-flex",
    alignItems: "center",
    gap: "0.5rem",
    padding: "0.25rem 0.75rem",
    borderRadius: "25px",
    backgroundColor: "rgba(19, 91, 236, 0.1)",
    color: "#135bec",
    fontSize: "0.75rem",
    fontWeight: "700",
    letterSpacing: "0.05em",
    textTransform: "uppercase",
    width: "fit-content",
  },

  heroTitle: {
    fontSize: "3rem",
    fontWeight: "900",
    lineHeight: "1.1",
    color: "#0d121b",
    margin: 0,
    letterSpacing: "-0.025em",
  },

  heroTitleAccent: {
    color: "#135bec",
  },

  heroDescription: {
    fontSize: "1.125rem",
    color: "#4c669a",
    lineHeight: "1.625",
    maxWidth: "500px",
    fontWeight: "400",
  },

  heroButtons: {
    display: "flex",
    gap: "1rem",
    flexWrap: "wrap",
  },

  primaryButton: {
    minWidth: "160px",
    padding: "1rem 2rem",
    backgroundColor: "#135bec",
    color: "#ffffff",
    border: "none",
    borderRadius: "12px",
    fontSize: "1rem",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.3s ease",
    boxShadow: "0 8px 25px rgba(19, 91, 236, 0.25)",
  },

  secondaryButton: {
    minWidth: "160px",
    padding: "1rem 2rem",
    backgroundColor: "#ffffff",
    color: "#0d121b",
    border: "1px solid #e7ebf3",
    borderRadius: "12px",
    fontSize: "1rem",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.3s ease",
  },

  socialProof: {
    display: "flex",
    alignItems: "center",
    gap: "1.5rem",
    paddingTop: "1rem",
    borderTop: "1px solid #e7ebf3",
  },

  avatarGroup: {
    display: "flex",
    marginLeft: "-0.75rem",
  },

  avatar: {
    width: "40px",
    height: "40px",
    borderRadius: "50%",
    border: "2px solid #ffffff",
    backgroundSize: "cover",
    backgroundPosition: "center",
    marginLeft: "-0.75rem",
  },

  socialProofText: {
    color: "#4c669a",
    fontSize: "0.875rem",
    fontWeight: "500",
  },

  heroRight: {
    position: "relative",
  },

  imageWrapper: {
    position: "relative",
  },

  glowTopRight: {
    position: "absolute",
    top: "-40px",
    right: "-40px",
    width: "256px",
    height: "256px",
    backgroundColor: "rgba(19, 91, 236, 0.1)",
    borderRadius: "50%",
    filter: "blur(60px)",
    zIndex: -1,
  },

  glowBottomLeft: {
    position: "absolute",
    bottom: "-40px",
    left: "-40px",
    width: "256px",
    height: "256px",
    backgroundColor: "rgba(107, 142, 142, 0.1)",
    borderRadius: "50%",
    filter: "blur(60px)",
    zIndex: -1,
  },

  heroImage: {
    width: "100%",
    maxWidth: "500px",
    aspectRatio: "4/3",
    backgroundSize: "cover",
    backgroundPosition: "center",
    borderRadius: "16px",
    boxShadow: "0 25px 50px rgba(0, 0, 0, 0.1)",
    overflow: "hidden",
    border: "8px solid #ffffff",
    position: "relative",
    transition: "transform 0.3s ease",
    margin: "0 auto",
  },

  floatingCard: {
    position: "absolute",
    bottom: "24px",
    right: "24px",
    backgroundColor: "#ffffff",
    padding: "1rem",
    borderRadius: "12px",
    boxShadow: "0 8px 25px rgba(0, 0, 0, 0.1)",
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
    animation: "float 3s ease-in-out infinite",
  },

  cardIcon: {
    backgroundColor: "rgba(16, 185, 129, 0.1)",
    padding: "0.5rem",
    borderRadius: "8px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },

  cardTitle: {
    fontSize: "0.625rem",
    fontWeight: "700",
    color: "#4c669a",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
  },

  cardSubtitle: {
    fontSize: "1.25rem",
    fontWeight: "900",
    color: "#0d121b",
  },

  // Trust Badges
  trustBadges: {
    padding: "3rem 1.5rem",
    backgroundColor: "#ffffff",
    borderTop: "1px solid #e7ebf3",
    borderBottom: "1px solid #e7ebf3",
  },

  trustText: {
    textAlign: "center",
    fontSize: "0.875rem",
    fontWeight: "700",
    color: "#4c669a",
    letterSpacing: "0.1em",
    marginBottom: "2rem",
    textTransform: "uppercase",
  },

  brandLogos: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    gap: "4rem",
    flexWrap: "wrap",
  },

  brandLogo: {
    fontSize: "1.5rem",
    fontWeight: "900",
    color: "#9ca3af",
    letterSpacing: "-0.05em",
  },

  // Features Section
  features: {
    padding: "5rem 1.5rem",
    maxWidth: "1400px",
    margin: "0 auto",
  },

  sectionHeader: {
    textAlign: "center",
    marginBottom: "4rem",
  },

  sectionLabel: {
    fontSize: "0.875rem",
    fontWeight: "700",
    color: "#135bec",
    letterSpacing: "0.2em",
    marginBottom: "1rem",
    textTransform: "uppercase",
  },

  sectionTitle: {
    fontSize: "1.875rem",
    fontWeight: "900",
    color: "#0d121b",
    marginBottom: "1rem",
    lineHeight: "1.25",
    letterSpacing: "-0.025em",
  },

  sectionDescription: {
    fontSize: "1.125rem",
    color: "#4c669a",
    maxWidth: "600px",
    margin: "0 auto",
    lineHeight: "1.625",
    fontWeight: "400",
  },

  featuresGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: "2rem",
  },

  featureCard: {
    backgroundColor: "#ffffff",
    padding: "2rem",
    borderRadius: "16px",
    border: "1px solid #e7ebf3",
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
    transition: "all 0.3s ease",
  },

  featureIconWrapper: {
    width: "60px",
    height: "60px",
    backgroundColor: "rgba(19, 91, 236, 0.1)",
    borderRadius: "12px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },

  featureTitle: {
    fontSize: "1.25rem",
    fontWeight: "700",
    color: "#0d121b",
    margin: 0,
  },

  featureDescription: {
    color: "#4c669a",
    lineHeight: "1.6",
    fontSize: "0.9375rem",
  },

  featureImage: {
    width: "100%",
    aspectRatio: "16/9",
    backgroundSize: "cover",
    backgroundPosition: "center",
    borderRadius: "8px",
    marginTop: "auto",
    border: "1px solid #e7ebf3",
  },

  // Stats Section
  stats: {
    backgroundColor: "#135bec",
    padding: "5rem 1.5rem",
    color: "#ffffff",
    position: "relative",
    overflow: "hidden",
  },

  skewedBg: {
    position: "absolute",
    top: 0,
    right: 0,
    width: "50%",
    height: "100%",
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    transform: "skewX(-20deg) translateX(50%)",
    pointerEvents: "none",
  },

  statsContent: {
    maxWidth: "1200px",
    margin: "0 auto",
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: "3rem",
    textAlign: "center",
    position: "relative",
    zIndex: 10,
  },

  statItem: {
    display: "flex",
    flexDirection: "column",
    gap: "0.5rem",
  },

  statNumber: {
    fontSize: "2.25rem",
    fontWeight: "900",
    margin: 0,
  },

  statLabel: {
    fontSize: "0.875rem",
    fontWeight: "700",
    color: "rgba(255, 255, 255, 0.7)",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
  },

  // CTA Section
  cta: {
    padding: "5rem 1.5rem 8rem",
  },

  ctaCard: {
    maxWidth: "1000px",
    margin: "0 auto",
    backgroundColor: "#ffffff",
    borderRadius: "32px",
    padding: "4rem",
    border: "1px solid #e7ebf3",
    boxShadow: "0 25px 50px rgba(0, 0, 0, 0.05)",
    position: "relative",
    overflow: "hidden",
  },

  ctaCircle: {
    position: "absolute",
    bottom: "-96px",
    right: "-96px",
    width: "256px",
    height: "256px",
    backgroundColor: "rgba(19, 91, 236, 0.05)",
    borderRadius: "50%",
  },

  ctaContent: {
    position: "relative",
    zIndex: 10,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    gap: "2rem",
  },

  ctaTitle: {
    fontSize: "2.25rem",
    fontWeight: "900",
    color: "#0d121b",
    lineHeight: "1.25",
    maxWidth: "700px",
    margin: 0,
    letterSpacing: "-0.025em",
  },

  ctaDescription: {
    fontSize: "1.125rem",
    color: "#4c669a",
    maxWidth: "600px",
    lineHeight: "1.6",
  },

  ctaButtons: {
    display: "flex",
    gap: "1rem",
    flexWrap: "wrap",
    justifyContent: "center",
  },

  ctaPrimaryButton: {
    minWidth: "200px",
    padding: "1rem 2rem",
    backgroundColor: "#135bec",
    color: "#ffffff",
    border: "none",
    borderRadius: "12px",
    fontSize: "1rem",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.3s ease",
    boxShadow: "0 8px 25px rgba(19, 91, 236, 0.2)",
  },

  ctaSecondaryButton: {
    minWidth: "200px",
    padding: "1rem 2rem",
    backgroundColor: "transparent",
    color: "#0d121b",
    border: "2px solid #0d121b",
    borderRadius: "12px",
    fontSize: "1rem",
    fontWeight: "700",
    cursor: "pointer",
    transition: "all 0.3s ease",
  },

  // Footer
  footer: {
    backgroundColor: "#ffffff",
    borderTop: "1px solid #e7ebf3",
    padding: "5rem 1.5rem",
  },

  footerContent: {
    maxWidth: "1200px",
    margin: "0 auto",
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: "3rem",
    marginBottom: "5rem",
  },

  footerColumn: {
    display: "flex",
    flexDirection: "column",
    gap: "1.5rem",
  },

  footerLogo: {
    display: "flex",
    alignItems: "center",
    gap: "0.75rem",
  },

  footerLogoIcon: {
    backgroundColor: "#135bec",
    padding: "0.25rem",
    borderRadius: "8px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },

  footerLogoText: {
    fontSize: "1.25rem",
    fontWeight: "700",
    color: "#0d121b",
    margin: 0,
  },

  footerDescription: {
    fontSize: "0.875rem",
    color: "#4c669a",
    lineHeight: "1.6",
  },

  socialLinks: {
    display: "flex",
    gap: "1rem",
  },

  socialLink: {
    width: "40px",
    height: "40px",
    borderRadius: "8px",
    backgroundColor: "#f0f2f7",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#4c669a",
    textDecoration: "none",
    transition: "all 0.3s ease",
  },

  footerColumnTitle: {
    fontSize: "1rem",
    fontWeight: "700",
    color: "#0d121b",
    margin: 0,
  },

  footerList: {
    listStyle: "none",
    padding: 0,
    margin: 0,
    display: "flex",
    flexDirection: "column",
    gap: "1rem",
  },

  footerLink: {
    fontSize: "0.875rem",
    color: "#4c669a",
    textDecoration: "none",
    transition: "color 0.3s ease",
  },

  newsletterText: {
    fontSize: "0.875rem",
    color: "#4c669a",
  },

  newsletterForm: {
    display: "flex",
    flexDirection: "column",
    gap: "0.75rem",
  },

  newsletterInput: {
    width: "100%",
    backgroundColor: "#f0f2f7",
    border: "none",
    borderRadius: "8px",
    height: "48px",
    padding: "0 1rem",
    fontSize: "0.875rem",
    outline: "none",
  },

  newsletterButton: {
    width: "100%",
    backgroundColor: "#135bec",
    color: "#ffffff",
    fontWeight: "700",
    height: "48px",
    borderRadius: "8px",
    border: "none",
    cursor: "pointer",
    transition: "all 0.3s ease",
  },

  footerBottom: {
    maxWidth: "1200px",
    margin: "0 auto",
    paddingTop: "2rem",
    borderTop: "1px solid #e7ebf3",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },

  copyright: {
    fontSize: "0.75rem",
    color: "#9ca3af",
  },

  footerLinks: {
    display: "flex",
    gap: "1.5rem",
  },

  footerBottomLink: {
    fontSize: "0.75rem",
    color: "#9ca3af",
    textDecoration: "none",
    transition: "color 0.3s ease",
  },
};
export default ParkEaseHome;
