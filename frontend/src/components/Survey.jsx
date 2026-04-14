import { useState } from "react";
import { SURVEY_QUESTIONS } from "../data/survey";

function ProgressBar({ current, total }) {
  return (
    <div className="survey-progress">
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${((current + 1) / total) * 100}%` }}
        />
      </div>
      <span className="progress-text">
        {current + 1} of {total}
      </span>
    </div>
  );
}

function OptionCard({ option, selected, onToggle, type }) {
  const isSelected = type === "multi-select" ? selected.includes(option.value) : selected === option.value;

  return (
    <button
      className={`option-card${isSelected ? " selected" : ""}`}
      onClick={() => onToggle(option.value)}
      type="button"
    >
      {option.emoji && <span className="option-emoji">{option.emoji}</span>}
      <div className="option-content">
        <span className="option-label">{option.label}</span>
        {option.hint && <span className="option-hint">{option.hint}</span>}
      </div>
      <div className={`option-check${isSelected ? " checked" : ""}`}>
        {isSelected && <CheckIcon />}
      </div>
    </button>
  );
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path
        d="M3 7L6 10L11 4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Survey({ onComplete }) {
  const [step, setStep] = useState(0);
  const [responses, setResponses] = useState({});

  const question = SURVEY_QUESTIONS[step];
  const currentValue = responses[question.id] || (question.type === "multi-select" ? [] : null);

  const handleToggle = (value) => {
    if (question.type === "multi-select") {
      const current = responses[question.id] || [];
      const updated = current.includes(value)
        ? current.filter((v) => v !== value)
        : [...current, value];
      setResponses({ ...responses, [question.id]: updated });
    } else {
      setResponses({ ...responses, [question.id]: value });
    }
  };

  const canProceed = () => {
    if (question.optional) return true;
    if (question.type === "multi-select") {
      return (responses[question.id] || []).length > 0;
    }
    return responses[question.id] != null;
  };

  const handleNext = () => {
    if (step < SURVEY_QUESTIONS.length - 1) {
      setStep(step + 1);
    } else {
      onComplete(responses);
    }
  };

  const handleBack = () => {
    if (step > 0) {
      setStep(step - 1);
    }
  };

  const handleSkip = () => {
    if (step < SURVEY_QUESTIONS.length - 1) {
      setStep(step + 1);
    } else {
      onComplete(responses);
    }
  };

  return (
    <div className="survey">
      <ProgressBar current={step} total={SURVEY_QUESTIONS.length} />

      <div className="survey-question">
        <h2 className="survey-title">{question.title}</h2>
        {question.subtitle && <p className="survey-subtitle">{question.subtitle}</p>}

        <div className={`options-grid${question.options.length <= 3 ? " few-options" : ""}`}>
          {question.options.map((option) => (
            <OptionCard
              key={option.value}
              option={option}
              selected={currentValue}
              onToggle={handleToggle}
              type={question.type}
            />
          ))}
        </div>
      </div>

      <div className="survey-nav">
        {step > 0 && (
          <button className="survey-btn secondary" onClick={handleBack} type="button">
            <BackIcon /> Back
          </button>
        )}
        <div className="survey-nav-right">
          {question.optional && (
            <button className="survey-btn text" onClick={handleSkip} type="button">
              Skip
            </button>
          )}
          <button
            className="survey-btn primary"
            onClick={handleNext}
            disabled={!canProceed()}
            type="button"
          >
            {step === SURVEY_QUESTIONS.length - 1 ? "Find my match" : "Continue"}
            {step < SURVEY_QUESTIONS.length - 1 && <NextIcon />}
          </button>
        </div>
      </div>
    </div>
  );
}

function BackIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path
        d="M10 12L6 8L10 4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function NextIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path
        d="M6 4L10 8L6 12"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
