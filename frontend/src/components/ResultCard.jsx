import { useState, useEffect } from "react";

const DIMS = ["aroma", "flavor", "acidity", "body", "balance", "aftertaste"];
const RANK_LABELS = ["Best Match", "2nd Match", "3rd Match"];

function ScoreBar({ label, value }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    setWidth(0);
    const t = setTimeout(() => setWidth((value / 10) * 100), 80);
    return () => clearTimeout(t);
  }, [value]);

  return (
    <div className="rc-score-row">
      <span className="rc-score-lbl">{label.charAt(0).toUpperCase() + label.slice(1)}</span>
      <div className="rc-bar-bg">
        <div className="rc-bar-fill" style={{ width: `${width}%` }} />
      </div>
      <span className="rc-score-num">{Number(value).toFixed(2)}</span>
    </div>
  );
}

export default function ResultCarousel({ results }) {
  const [idx, setIdx] = useState(0);
  const bean = results[idx];
  const scores = bean.scores ?? {};
  const availableDims = DIMS.filter((d) => scores[d] != null);

  const prev = () => setIdx((i) => (i - 1 + results.length) % results.length);
  const next = () => setIdx((i) => (i + 1) % results.length);

  return (
    <div className="carousel-card">
      {/* Header row */}
      <div className="carousel-header">
        <div className="carousel-meta">
          <span className="rc-rank">{RANK_LABELS[idx] ?? `#${idx + 1}`}</span>
          <span className="rc-similarity">{(bean.similarity * 100).toFixed(0)}% match</span>
        </div>
        <div className="carousel-nav">
          <button
            className="carousel-btn"
            onClick={prev}
            disabled={results.length <= 1}
            aria-label="Previous"
          >
            &#8249;
          </button>
          <span className="carousel-pips">
            {results.map((_, i) => (
              <span
                key={i}
                className={`carousel-pip${i === idx ? " active" : ""}`}
                onClick={() => setIdx(i)}
              />
            ))}
          </span>
          <button
            className="carousel-btn"
            onClick={next}
            disabled={results.length <= 1}
            aria-label="Next"
          >
            &#8250;
          </button>
        </div>
      </div>

      {/* Country + process */}
      <div className="rc-country">
        <CountryDot />
        {bean.country}
      </div>
      <div className="rc-process-row">
        <span className="rc-process">{bean.process}</span>
      </div>

      {/* Score bars */}
      {/* {availableDims.length > 0 && (
        <div className="rc-scores">
          {availableDims.map((d) => (
            <ScoreBar key={`${idx}-${d}`} label={d} value={scores[d]} />
          ))}
        </div>
      )} */}
    </div>
  );
}

function CountryDot() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none" style={{ flexShrink: 0 }}>
      <circle cx="6.5" cy="6.5" r="6" fill="#E8DDD5" stroke="#D4C5B5" strokeWidth="0.5" />
      <circle cx="6.5" cy="6.5" r="3" fill="#C9A882" />
    </svg>
  );
}