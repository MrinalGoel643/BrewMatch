import { useState } from "react";
import CoffeeCup from "./components/CoffeeCup";
import PreferencesPanel from "./components/PreferencesPanel";
import ResultCard from "./components/ResultCard";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

// ── Dummy data — used when backend is unavailable ──────────────────────────
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
  if (!raw || raw === "NaN" || raw === "nan" || raw === "null") return "—";
  return raw;
}

async function fetchRecommendations({
  aroma, flavor, aftertaste, acidity, body, balance,
  uniformity, cleanCup, sweetness, model,
}) {
  const recRes = await fetch(`${API_BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      preferences: {
        aroma, flavor, aftertaste, acidity, body, balance,
        uniformity,
        clean_cup: cleanCup,
        sweetness,
      },
      model,
      k: 3,
    }),
  });

  if (!recRes.ok) throw new Error(`Server error ${recRes.status}`);
  const recData = await recRes.json();

  return recData.recommendations.map((rec) => ({
    id:         rec.id,
    similarity: rec.similarity,
    country:    rec.country ?? rec.metadata?.["Country of Origin"] ?? "Unknown",
    process:    parseProcess(rec.metadata),
    scores:     rec.scores ?? {},
  }));
}

function BeanLogoIcon() {
  return (
    <img
      src="/logo.png"
      alt="BrewMatch logo"
      style={{ width: 38, height: 38, objectFit: "contain" }}
      onError={(e) => { e.target.style.display = "none"; }}
    />
  );
}

export default function App() {
  const [aroma,      setAroma]      = useState(8.0);
  const [flavor,     setFlavor]     = useState(7.5);
  const [aftertaste, setAftertaste] = useState(7.0);
  const [acidity,    setAcidity]    = useState(7.5);
  const [body,       setBody]       = useState(8.0);
  const [balance,    setBalance]    = useState(7.5);
  const [uniformity, setUniformity] = useState(10.0);
  const [cleanCup,   setCleanCup]   = useState(10.0);
  const [sweetness,  setSweetness]  = useState(10.0);
  const [model,      setModel]      = useState("classical");

  const [phase,   setPhase]   = useState("idle");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFind = async () => {
    if (loading) return;
    setLoading(true);
    setResults(null);
    setPhase("pouring");

    let details;
    try {
      details = await fetchRecommendations({
        aroma, flavor, aftertaste, acidity, body, balance,
        uniformity, cleanCup, sweetness, model,
      });
      console.log("✅ API results:", details);
    } catch (err){
      console.error("❌ API failed, using dummy:", err);
      details = DUMMY_RESULTS;
    }

    setTimeout(() => {
      setPhase("done");
      setTimeout(() => {
        setResults(details);
        setLoading(false);
      }, 700);
    }, 1500);
  };

  return (
    <div className="app">
      <header className="header">
        <BeanLogoIcon />
        <div className="brand">
          <h1>BrewMatch</h1>
          <p>Find your best sip</p>
        </div>
      </header>

      <main className="layout">
        <PreferencesPanel
          aroma={aroma}           setAroma={setAroma}
          flavor={flavor}         setFlavor={setFlavor}
          aftertaste={aftertaste} setAftertaste={setAftertaste}
          acidity={acidity}       setAcidity={setAcidity}
          body={body}             setBody={setBody}
          balance={balance}       setBalance={setBalance}
          uniformity={uniformity} setUniformity={setUniformity}
          cleanCup={cleanCup}     setCleanCup={setCleanCup}
          sweetness={sweetness}   setSweetness={setSweetness}
          model={model}           setModel={setModel}
          loading={loading}
          onFind={handleFind}
        />

        <div className="cup-side">
          <CoffeeCup phase={phase} />
          <p className="cup-hint">
            {results
              ? "Your top matches ↓"
              : "Adjust your preferences and find your match"}
          </p>
          {results && <ResultCard results={results} />}
        </div>
      </main>
    </div>
  );
}