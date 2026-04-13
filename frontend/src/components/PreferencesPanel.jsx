const MODEL_OPTIONS = [
  { value: "naive",     label: "Naive baseline" },
  { value: "classical", label: "Classical ML" },
  { value: "deep",      label: "Deep learning" },
];

function Slider({ label, value, onChange, min = 0, max = 10, step = 0.5 }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="field">
      <label>{label}</label>
      <div className="slider-row">
        <span className="slider-label-sm">Low</span>
        <div className="slider-track">
          <div
            className="slider-bg"
            style={{
              background: `linear-gradient(to right, var(--br500) ${pct}%, var(--bar-bg) ${pct}%)`,
            }}
          />
          <input
            type="range"
            min={min} max={max} step={step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
          />
        </div>
        <span className="slider-label-sm" style={{ textAlign: "right" }}>High</span>
        <span className="slider-val">{value.toFixed(1)}</span>
      </div>
    </div>
  );
}

export default function PreferencesPanel({
  aroma,      setAroma,
  flavor,     setFlavor,
  aftertaste, setAftertaste,
  acidity,    setAcidity,
  body,       setBody,
  balance,    setBalance,
  uniformity, setUniformity,
  cleanCup,   setCleanCup,
  sweetness,  setSweetness,
  model,      setModel,
  loading,    onFind,
}) {
  return (
    <div className="panel">
      <p className="panel-title">Your Preferences</p>

      <Slider label="Aroma"      value={aroma}      onChange={setAroma} />
      <Slider label="Flavor"     value={flavor}     onChange={setFlavor} />
      <Slider label="Aftertaste" value={aftertaste} onChange={setAftertaste} />
      <Slider label="Acidity"    value={acidity}    onChange={setAcidity} />
      <Slider label="Body"       value={body}       onChange={setBody} />
      <Slider label="Balance"    value={balance}    onChange={setBalance} />
      <Slider label="Uniformity" value={uniformity} onChange={setUniformity} />
      <Slider label="Clean Cup"  value={cleanCup}   onChange={setCleanCup} />
      <Slider label="Sweetness"  value={sweetness}  onChange={setSweetness} />

      {/* Model selector — uncomment when backend supports multiple models
      <div className="field">
        <label>Model</label>
        <div className="model-tabs">
          {MODEL_OPTIONS.map((m) => (
            <button
              key={m.value}
              className={`model-tab${model === m.value ? " active" : ""}`}
              onClick={() => setModel(m.value)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div> */}

      <button className="cta-btn" onClick={onFind} disabled={loading}>
        {loading ? <><SpinnerIcon /> Matching…</> : <><SearchIcon /> Find my BrewMatch</>}
      </button>
    </div>
  );
}

function SearchIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <circle cx="6" cy="6" r="4.2" stroke="currentColor" strokeWidth="1.4" />
      <path d="M9.5 9.5L13 13" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function SpinnerIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <circle cx="7.5" cy="7.5" r="5.5" stroke="currentColor" strokeWidth="1.5" strokeDasharray="18 14" opacity="0.8">
        <animateTransform attributeName="transform" type="rotate" from="0 7.5 7.5" to="360 7.5 7.5" dur="0.75s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}