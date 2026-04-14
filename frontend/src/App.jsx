import { useState, useEffect } from "react";
import Survey from "./components/Survey";
import Results from "./components/Results";
import CoffeeCup from "./components/CoffeeCup";
import { surveyToPreferences } from "./data/survey";
import { useBrewMatchProfile, formatSavedDate } from "./hooks/useLocalStorage";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

// Dummy results for when API is unavailable
const DUMMY_RESULTS = [
  {
    id: 1,
    similarity: 0.95,
    country: "Ethiopia",
    process: "Washed / Wet",
    scores: { aroma: 8.5, flavor: 8.67, aftertaste: 8.25, acidity: 8.42, body: 7.75, balance: 8.42 },
  },
  {
    id: 2,
    similarity: 0.88,
    country: "Colombia",
    process: "Washed / Wet",
    scores: { aroma: 8.42, flavor: 8.5, aftertaste: 8.0, acidity: 8.17, body: 8.17, balance: 8.42 },
  },
  {
    id: 3,
    similarity: 0.81,
    country: "Kenya",
    process: "Washed / Wet",
    scores: { aroma: 8.58, flavor: 8.58, aftertaste: 8.1, acidity: 8.67, body: 7.83, balance: 8.33 },
  },
];

function parseProcess(metadata) {
  const raw = metadata?.["Processing Method"] ?? metadata?.["Processing.Method"];
  if (!raw || raw === "NaN" || raw === "nan" || raw === "null") return "Unknown";
  return raw;
}

async function fetchRecommendations(preferences) {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      preferences: {
        aroma: preferences.aroma,
        flavor: preferences.flavor,
        aftertaste: preferences.aftertaste,
        acidity: preferences.acidity,
        body: preferences.body,
        balance: preferences.balance,
        uniformity: preferences.uniformity,
        clean_cup: preferences.clean_cup,
        sweetness: preferences.sweetness,
      },
      model: "classical",
      k: 3,
    }),
  });

  if (!res.ok) throw new Error(`Server error ${res.status}`);
  const data = await res.json();

  return data.recommendations.map((rec) => ({
    id: rec.id,
    similarity: rec.similarity,
    country: rec.country ?? rec.metadata?.["Country of Origin"] ?? "Unknown",
    process: parseProcess(rec.metadata),
    scores: rec.scores ?? {},
  }));
}

function LandingPage({ onStart, savedProfile, onViewSaved }) {
  return (
    <div className="landing">
      <div className="landing-content">
        <div className="landing-cup">
          <CoffeeCup phase="idle" />
        </div>
        <h2 className="landing-headline">Find your perfect coffee</h2>
        <p className="landing-subhead">
          Answer a few questions about your taste preferences, and we'll match you
          with coffee profiles and roasters you'll love.
        </p>

        {savedProfile ? (
          <div className="landing-actions">
            <button className="cta-btn large" onClick={onViewSaved} type="button">
              <ProfileIcon /> View my profile
            </button>
            <button className="cta-btn large secondary" onClick={onStart} type="button">
              <RefreshIcon /> Retake survey
            </button>
            <p className="landing-note">
              Profile saved {formatSavedDate(savedProfile.savedAt)}
            </p>
          </div>
        ) : (
          <>
            <button className="cta-btn large" onClick={onStart} type="button">
              <CoffeeIcon /> Let's get started
            </button>
            <p className="landing-note">Takes about 60 seconds</p>
          </>
        )}
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="loading-state">
      <div className="loading-cup">
        <CoffeeCup phase="pouring" />
      </div>
      <h2 className="loading-title">Brewing your recommendations...</h2>
      <p className="loading-subtitle">Analyzing your taste profile</p>
    </div>
  );
}

function CoffeeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <path
        d="M6 1V3M9 1V3M12 1V3M3 6H15V14C15 15.1046 14.1046 16 13 16H5C3.89543 16 3 15.1046 3 14V6ZM15 9H16C16.5523 9 17 9.44772 17 10V11C17 11.5523 16.5523 12 16 12H15"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ProfileIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <circle cx="9" cy="6" r="3" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M3 15C3 12.2386 5.68629 10 9 10C12.3137 10 15 12.2386 15 15"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
      <path
        d="M2.5 9C2.5 5.41015 5.41015 2.5 9 2.5C11.3066 2.5 13.3482 3.63599 14.5 5.375M15.5 9C15.5 12.5899 12.5899 15.5 9 15.5C6.69338 15.5 4.65176 14.364 3.5 12.625M14.5 2.5V5.5H11.5M3.5 15.5V12.5H6.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function App() {
  const { profile, saveProfile, clearProfile } = useBrewMatchProfile();
  const [screen, setScreen] = useState("landing");
  const [results, setResults] = useState(null);
  const [preferences, setPreferences] = useState(null);

  // Check for saved profile on mount
  useEffect(() => {
    if (profile?.results && profile?.preferences) {
      // We have a saved profile, stay on landing to let user choose
    }
  }, [profile]);

  const handleStartSurvey = () => {
    setScreen("survey");
  };

  const handleViewSaved = () => {
    if (profile?.results && profile?.preferences) {
      setResults(profile.results);
      setPreferences(profile.preferences);
      setScreen("results");
    }
  };

  const handleSurveyComplete = async (responses) => {
    setScreen("loading");

    // Convert survey responses to taste preferences
    const prefs = surveyToPreferences(responses);
    setPreferences(prefs);

    let recommendations;
    try {
      recommendations = await fetchRecommendations(prefs);
      console.log("API results:", recommendations);
    } catch (err) {
      console.error("API failed, using fallback:", err);
      recommendations = DUMMY_RESULTS;
    }

    setResults(recommendations);

    // Save to localStorage
    saveProfile({
      surveyResponses: responses,
      preferences: prefs,
      results: recommendations,
    });

    // Small delay for animation
    setTimeout(() => {
      setScreen("results");
    }, 1500);
  };

  const handleRetake = () => {
    setScreen("survey");
  };

  const handleGoHome = () => {
    setScreen("landing");
  };

  return (
    <div className="app">
      <header className="header">
        <button className="logo-btn" onClick={handleGoHome} type="button">
          <BeanLogo />
          <div className="brand">
            <h1>BrewMatch</h1>
            <p>Find your perfect cup</p>
          </div>
        </button>
      </header>

      <main className="main">
        {screen === "landing" && (
          <LandingPage
            onStart={handleStartSurvey}
            savedProfile={profile}
            onViewSaved={handleViewSaved}
          />
        )}
        {screen === "survey" && <Survey onComplete={handleSurveyComplete} />}
        {screen === "loading" && <LoadingState />}
        {screen === "results" && results && (
          <Results
            coffeeMatches={results}
            preferences={preferences}
            savedAt={profile?.savedAt}
            onStartOver={handleGoHome}
            onRetake={handleRetake}
          />
        )}
      </main>

      <footer className="footer">
        <p>
          Powered by machine learning trained on Coffee Quality Institute data
        </p>
      </footer>
    </div>
  );
}

function BeanLogo() {
  return (
    <img
      src="/logo.png"
      alt="BrewMatch"
      className="logo-img"
      onError={(e) => {
        e.target.style.display = "none";
      }}
    />
  );
}
