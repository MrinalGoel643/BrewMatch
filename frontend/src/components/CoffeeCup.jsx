// CoffeeCup.jsx
// phase: "idle" | "pouring" | "done"

import { useState, useEffect } from "react";

export default function CoffeeCup({ phase }) {
  const isPouring = phase === "pouring";
  const isDone = phase === "done" || phase === "revealed";

  const [fill, setFill] = useState(0);

  useEffect(() => {
    if (isPouring) {
      setFill(0);
      let start = null;
      const duration = 1800;
      const step = (ts) => {
        if (!start) start = ts;
        const p = Math.min((ts - start) / duration, 1);
        setFill(p);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
    if (phase === "idle") setFill(0);
  }, [phase]);

  // coffeeY starts exactly at the inner rim (193) and rises upward
  const maxFillHeight = 58;
  const coffeeY = 193 - fill * maxFillHeight;
  const coffeeHeight = 193 - coffeeY;
  const streamOpacity = isPouring ? 1 : 0;
  const steamOpacity = isDone ? 1 : 0;
  const dripX = 130;

  return (
    <svg
      width="300"
      height="375"
      viewBox="0 0 240 300"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ overflow: "visible" }}
    >
      <defs>
        {/* Clip matches inner rim ellipse + cup body exactly */}
        <clipPath id="cup-fill-clip">
          <path d="M73 193 Q70 242 82 254 Q98 264 120 264 Q142 264 158 254 Q170 242 167 193 Z" />
        </clipPath>
      </defs>

      {/* ── ESPRESSO MACHINE ── */}
      <rect x="40" y="10" width="160" height="85" rx="14" fill="#3D2B1F" />
      <rect x="48" y="16" width="144" height="6" rx="3" fill="#5A3E2F" opacity="0.6" />
      <rect x="50" y="25" width="140" height="55" rx="8" fill="#2C1E14" />

      {/* Gauge */}
      <circle cx="80" cy="52" r="18" fill="#3D2B1F" stroke="#5A3E2F" strokeWidth="1.5" />
      <circle cx="80" cy="52" r="13" fill="#1E1208" />
      <circle cx="80" cy="52" r="9" fill="#2C1E14" />
      <line x1="80" y1="52" x2="86" y2="46" stroke="#C9A882" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="80" cy="52" r="2" fill="#C9A882" />

      {/* Buttons */}
      <circle cx="145" cy="44" r="7" fill="#1E1208" stroke="#5A3E2F" strokeWidth="1" />
      <circle cx="145" cy="44" r="4" fill={isPouring ? "#7EB97E" : "#3D2B1F"} style={{ transition: "fill 0.3s" }} />
      <circle cx="163" cy="44" r="7" fill="#1E1208" stroke="#5A3E2F" strokeWidth="1" />
      <circle cx="163" cy="44" r="4" fill="#5A3E2F" />

      {/* Display */}
      <rect x="130" y="55" width="50" height="16" rx="3" fill="#1E1208" />
      <rect x="133" y="58" width={isPouring || isDone ? 40 : 0} height="10" rx="2" fill="#4A7A4A" style={{ transition: "width 1.8s linear" }} />

      {/* Portafilter arm */}
      <rect x="108" y="82" width="44" height="10" rx="5" fill="#5A3E2F" />
      <rect x={dripX - 8} y="90" width="16" height="12" rx="4" fill="#4A3020" />
      <circle cx={dripX - 2} cy="99" r="2" fill="#2C1E14" />
      <circle cx={dripX + 4} cy="99" r="2" fill="#2C1E14" />
      <rect x="40" y="88" width="160" height="8" rx="2" fill="#2C1E14" />

      {/* ── STREAM ── */}
      <g style={{ opacity: streamOpacity, transition: "opacity 0.25s" }}>
        <path
          d={`M${dripX - 2} 101 C${dripX} 135 ${dripX + 1} 160 ${dripX} ${Math.max(coffeeY + 2, 103)}`}
          stroke="#6B3A1F"
          strokeWidth="3.5"
          strokeLinecap="round"
          fill="none"
        />
        <path
          d={`M${dripX + 1} 101 C${dripX + 1} 135 ${dripX + 2} 160 ${dripX + 1} ${Math.max(coffeeY + 2, 103)}`}
          stroke="#8B5030"
          strokeWidth="1"
          strokeLinecap="round"
          fill="none"
          opacity="0.5"
        />
        {fill > 0.05 && (
          <ellipse cx={dripX} cy={coffeeY + 1} rx="4" ry="2" fill="#8B5030" opacity="0.7" />
        )}
      </g>

      {/* ── CUP BODY ── */}
      <path
        d="M62 193 Q60 242 74 255 Q92 267 120 267 Q148 267 166 255 Q180 242 178 193 Z"
        fill="#FDFAF7"
        stroke="#EAE2D8"
        strokeWidth="1"
      />
      {/* Inner shadow */}
      <path
        d="M70 196 Q68 240 78 252 Q94 264 120 265"
        stroke="#EAE2D8"
        strokeWidth="5"
        strokeLinecap="round"
        fill="none"
        opacity="0.4"
      />

      {/* Handle */}
      <path d="M174 215 Q198 213 200 233 Q202 253 176 251" stroke="#E0D5C8" strokeWidth="11" strokeLinecap="round" fill="none" />
      <path d="M174 217 Q194 215 196 233 Q198 251 176 249" stroke="#FDFAF7" strokeWidth="6" strokeLinecap="round" fill="none" />

      {/* ── COFFEE LIQUID (clipped strictly inside cup walls) ── */}
      <g clipPath="url(#cup-fill-clip)">
        {coffeeHeight > 0 && (
          <rect x="60" y={coffeeY} width="120" height={coffeeHeight + 8} fill="#7B4020" />
        )}
        {coffeeHeight > 2 && (
          <ellipse cx="120" cy={coffeeY} rx="47" ry="6" fill="#8B5030" />
        )}
        {isDone && (
          <>
            <ellipse cx="120" cy={coffeeY} rx="44" ry="5.5" fill="#C9A882" opacity="0.9" />
            <circle cx="106" cy={coffeeY - 1} r="3" fill="rgba(255,255,255,0.22)" />
            <circle cx="120" cy={coffeeY + 1} r="2.5" fill="rgba(255,255,255,0.16)" />
            <circle cx="134" cy={coffeeY - 1} r="3" fill="rgba(255,255,255,0.22)" />
          </>
        )}
      </g>

      {/* ── RIM drawn on top to cap overflow cleanly ── */}
      <ellipse cx="120" cy="193" rx="58" ry="11" fill="#DDD0C4" />
      <ellipse cx="120" cy="191" rx="56" ry="9" fill="#EAE0D5" />
      <ellipse cx="120" cy="191" rx="47" ry="6.5" fill={coffeeHeight > 4 ? "#8B5030" : "#D4C5B5"} style={{ transition: "fill 0.4s" }} />


      {/* Saucer */}
      <ellipse cx="120" cy="278" rx="68" ry="8.5" fill="#D4C5B5" />
      <ellipse cx="120" cy="275" rx="66" ry="7" fill="#E8DDD5" />

      {/* ── STEAM — moves straight up ── */}
      <g style={{ opacity: steamOpacity, transition: "opacity 0.8s ease 0.4s" }}>
        {[
          { x: 108, d: "0s",    y1: 180, y2: 158 },
          { x: 120, d: "0.45s", y1: 180, y2: 152 },
          { x: 132, d: "0.9s",  y1: 180, y2: 158 },
        ].map(({ x, d, y1, y2 }, i) => (
          <line
            key={i}
            x1={x} y1={y1} x2={x} y2={y2}
            stroke="#C9A882"
            strokeWidth="2.5"
            strokeLinecap="round"
            opacity="0.6"
            style={{ animation: `steamRise 1.7s ease-in-out ${d} infinite` }}
          />
        ))}
      </g>
    </svg>
  );
}