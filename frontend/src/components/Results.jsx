import { matchRoasters } from "../data/roasters";
import { formatSavedDate } from "../hooks/useLocalStorage";

function CoffeeMatchCard({ match, rank, roasters }) {
  const rankLabels = ["Top Match", "Runner Up", "Third Pick"];

  // Find roasters that specialize in this origin
  const relevantRoasters = roasters.filter((r) =>
    r.specialties.includes(match.country)
  ).slice(0, 2);

  return (
    <div className="match-card">
      <div className="match-rank">
        {rankLabels[rank] || `#${rank + 1}`}
      </div>
      <div className="match-content">
        <div className="match-header">
          <h3 className="match-country">{match.country}</h3>
          <span className="match-similarity">{(match.similarity * 100).toFixed(0)}%</span>
        </div>
        <div className="match-process">{match.process}</div>

        {match.scores && Object.keys(match.scores).length > 0 && (
          <div className="match-scores">
            {["aroma", "flavor", "acidity", "body"].map((attr) =>
              match.scores[attr] != null ? (
                <div key={attr} className="score-chip">
                  <span className="score-label">{attr}</span>
                  <span className="score-value">{Number(match.scores[attr]).toFixed(1)}</span>
                </div>
              ) : null
            )}
          </div>
        )}

        {relevantRoasters.length > 0 && (
          <div className="match-roasters">
            <p className="match-roasters-label">Buy {match.country} beans from:</p>
            <div className="match-roaster-links">
              {relevantRoasters.map((r) => (
                <a
                  key={r.id}
                  href={r.website}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="match-roaster-link"
                >
                  {r.name}
                  <ExternalIcon />
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function RoasterCard({ roaster }) {
  return (
    <a
      href={roaster.website}
      target="_blank"
      rel="noopener noreferrer"
      className="roaster-card"
    >
      <div className="roaster-content">
        <div className="roaster-header">
          <h4 className="roaster-name">{roaster.name}</h4>
          <span className="roaster-price">{roaster.priceRange}</span>
        </div>
        <p className="roaster-location">{roaster.location}</p>
        <p className="roaster-style">{roaster.style}</p>
        <div className="roaster-tags">
          {roaster.flavorProfile.notes.slice(0, 3).map((note) => (
            <span key={note} className="roaster-tag">
              {note}
            </span>
          ))}
        </div>
      </div>
      <div className="roaster-arrow">
        <ExternalIcon />
      </div>
    </a>
  );
}

function PreferenceSummary({ preferences }) {
  const items = [];

  if (preferences.origins?.length > 0) {
    items.push({ label: "Origins", value: preferences.origins.slice(0, 3).join(", ") });
  }
  if (preferences.notes?.length > 0) {
    items.push({ label: "Flavors", value: preferences.notes.slice(0, 3).join(", ") });
  }
  if (preferences.acidityPref) {
    const acidityLabels = { high: "Bright", medium: "Balanced", low: "Mellow" };
    items.push({ label: "Acidity", value: acidityLabels[preferences.acidityPref] || preferences.acidityPref });
  }
  if (preferences.processing?.length > 0) {
    items.push({ label: "Processing", value: preferences.processing[0].replace(" / Wet", "").replace(" / Dry", "") });
  }

  if (items.length === 0) return null;

  return (
    <div className="preference-summary">
      {items.map((item) => (
        <div key={item.label} className="pref-item">
          <span className="pref-label">{item.label}</span>
          <span className="pref-value">{item.value}</span>
        </div>
      ))}
    </div>
  );
}

function ExternalIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path
        d="M6 3H3V13H13V10M9 3H13M13 3V7M13 3L7 9"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Results({ coffeeMatches, preferences, savedAt, onStartOver, onRetake }) {
  // Get roaster recommendations based on preferences
  const roasterMatches = matchRoasters({
    origins: preferences.origins || [],
    notes: preferences.notes || [],
    acidity: preferences.acidityPref || "medium",
    processing: preferences.processing || [],
  });

  return (
    <div className="results">
      <div className="results-header">
        <div className="results-header-left">
          <h2 className="results-title">Your Coffee Profile</h2>
          {savedAt && (
            <p className="results-saved">Saved {formatSavedDate(savedAt)}</p>
          )}
        </div>
        <div className="results-actions">
          <button className="action-btn secondary" onClick={onRetake} type="button">
            <RefreshIcon /> Retake
          </button>
        </div>
      </div>

      <PreferenceSummary preferences={preferences} />

      <section className="results-section">
        <h3 className="section-title">Your Top Matches</h3>
        <p className="section-desc">
          Based on your taste preferences, look for coffees with these profiles
        </p>
        <div className="matches-grid">
          {coffeeMatches.map((match, idx) => (
            <CoffeeMatchCard
              key={match.id}
              match={match}
              rank={idx}
              roasters={roasterMatches}
            />
          ))}
        </div>
      </section>

      <section className="results-section">
        <h3 className="section-title">Recommended Roasters</h3>
        <p className="section-desc">
          These roasters specialize in coffees that match your taste. All ship nationwide.
        </p>
        <div className="roasters-list">
          {roasterMatches.map((roaster) => (
            <RoasterCard key={roaster.id} roaster={roaster} />
          ))}
        </div>
      </section>

      <div className="results-footer">
        <p>Not quite right?</p>
        <button className="text-btn" onClick={onRetake} type="button">
          Retake the survey
        </button>
      </div>
    </div>
  );
}

function RefreshIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path
        d="M2 8C2 4.68629 4.68629 2 8 2C10.0503 2 11.8651 3.00976 13 4.5M14 8C14 11.3137 11.3137 14 8 14C5.94969 14 4.13489 12.9902 3 11.5M13 2V5H10M3 14V11H6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
