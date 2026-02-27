/**
 * App.jsx — Root component for the Ham / Spam Checker UI.
 *
 * Architecture overview:
 *   <App>                — manages all state and API calls
 *     <InfoTooltip>      — hover badge showing model metadata
 *     <Spinner>          — animated ring shown while the API is in-flight
 *     <ResultPanel>      — colour-coded score + progress bar
 *
 * Data flow:
 *   User types in textarea
 *     → useEffect fires after 600 ms debounce
 *       → POST /predict { text }
 *         → setState(spam_probability)
 *           → <ResultPanel> re-renders with new score
 */

import { useState, useEffect, useRef } from 'react'

// ---------------------------------------------------------------------------
// Model metadata — displayed in the <InfoTooltip> hover card.
// Update these if the model is retrained with different settings.
// ---------------------------------------------------------------------------
const MODEL_ROWS = [
  { key: 'Type',           val: 'TF-IDF + Dense Neural Network' },
  { key: 'Note',           val: 'Not an LLM — classical ML classifier' },
  { key: 'Vectorizer',     val: 'TF-IDF, 5,000 features' },
  { key: 'Architecture',   val: 'Dense 64 → Dense 32 → Sigmoid output' },
  { key: 'Training data',  val: '~101K labelled emails (spam_ham_dataset)' },
  { key: 'Train/test',     val: '80% / 20% split' },
  { key: 'Framework',      val: 'TensorFlow / Keras + scikit-learn' },
]

// ---------------------------------------------------------------------------
// <InfoTooltip>
// A small pill in the top-right corner of the card.
// Hover state is tracked with useState so the tooltip can be conditionally
// rendered — pure CSS :hover can't conditionally mount JSX.
// ---------------------------------------------------------------------------
function InfoTooltip() {
  const [visible, setVisible] = useState(false)

  return (
    <div
      className="info-trigger"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      ⓘ Model Info

      {/* Tooltip is only mounted while hovered — CSS fade-up animation plays on mount */}
      {visible && (
        <div className="info-tooltip">
          <div className="info-heading">About this classifier</div>
          {MODEL_ROWS.map(({ key, val }) => (
            <div className="info-row" key={key}>
              <span className="info-key">{key}</span>
              <span className="info-val">{val}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// <Spinner>
// Shown while the /predict API call is in-flight.
// Four staggered divs create a smooth multi-ring effect via CSS animation.
// ---------------------------------------------------------------------------
function Spinner() {
  return (
    <div className="spinner-container">
      <div className="spinner-ring">
        <div /><div /><div /><div />
      </div>
      <p className="spinner-text">Analysing your message...</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// <ResultPanel>
// Renders the spam score returned by the API.
// The progress bar animates via CSS transition on its `width` property.
// Colour switches from green (ham) to red (spam) at the 50% threshold.
// ---------------------------------------------------------------------------
function ResultPanel({ probability }) {
  const pct    = Math.round(probability * 100) // float → integer percentage
  const isSpam = pct >= 50
  const color  = isSpam ? '#f87171' : '#4ade80'  // red : green
  const label  = isSpam ? '🚨 Likely Spam' : '✅ Looks Like Ham'

  return (
    <div className="result-panel">
      <div className="result-label" style={{ color }}>{label}</div>

      {/* Animated progress bar — width driven by the spam probability */}
      <div className="progress-bg">
        <div
          className="progress-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>

      <p className="result-pct">
        <span style={{ color, fontWeight: 700, fontSize: '1.4rem' }}>{pct}%</span>
        <span className="result-pct-label"> likelihood of spam</span>
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// <App> — main component
// ---------------------------------------------------------------------------
export default function App() {
  // ── State ────────────────────────────────────────────────────────────────
  const [text,    setText]    = useState('')    // current textarea content
  const [loading, setLoading] = useState(false) // true while API call is pending
  const [result,  setResult]  = useState(null)  // spam_probability float, or null
  const [error,   setError]   = useState(null)  // error message string, or null

  // Ref to hold the active debounce timer across renders.
  // useRef is used (not useState) because updating it must NOT trigger a re-render.
  const debounceRef = useRef(null)

  // ── API call ─────────────────────────────────────────────────────────────
  /**
   * Sends the text to the Flask /predict endpoint and updates state.
   * Called by the debounced useEffect — not directly by user events.
   */
  const checkText = async (textToCheck) => {
    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textToCheck }),
      })

      if (!res.ok) throw new Error(`Server responded with ${res.status}`)

      const data = await res.json()
      setResult(data.spam_probability) // float in [0, 1]

    } catch (err) {
      // Network errors (API not running) or non-2xx responses land here
      console.error('[SpamChecker] predict error:', err)
      setError('Could not reach the API. Is api.py running on port 5000?')

    } finally {
      // Always clear the loading state, whether the call succeeded or failed
      setLoading(false)
    }
  }

  // ── Debounced effect ─────────────────────────────────────────────────────
  /**
   * Re-runs every time `text` changes.
   * Waits 600 ms after the user stops typing before firing the API call —
   * this prevents a request on every single keystroke.
   *
   * Cleanup function: cancels any pending timer when text changes again
   * before the 600 ms has elapsed, or when the component unmounts.
   */
  useEffect(() => {
    // Cancel any previously scheduled call
    if (debounceRef.current) clearTimeout(debounceRef.current)

    // If the box is empty, clear results immediately (no API call needed)
    if (!text.trim()) {
      setResult(null)
      setError(null)
      return
    }

    // Schedule a new call 600 ms in the future
    debounceRef.current = setTimeout(() => checkText(text), 600)

    // Cleanup: cancel the timer if text changes before 600 ms is up
    return () => clearTimeout(debounceRef.current)
  }, [text])

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleClear = () => {
    setText('')    // clears textarea (controlled component)
    setResult(null)
    setError(null)
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      <div className="card">

        {/* Model info pill — top-right corner, visible on hover */}
        <InfoTooltip />

        <div className="card-header">
          <span className="shield-icon">🛡️</span>
          <h1 className="title">Ham / Spam Checker</h1>
          <p className="subtitle">
            Start typing — the score updates automatically
            <span className="live-badge">● LIVE</span>
          </p>
        </div>

        {/* Controlled textarea — React owns the value via `text` state */}
        <textarea
          className="textarea"
          rows={8}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your email or message here..."
        />

        <div className="button-row">
          <button
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={!text && result === null} // disable when nothing to clear
          >
            🗑️ Clear Text
          </button>
        </div>

        {/* Conditional rendering — only one of these shows at a time */}
        {loading               && <Spinner />}
        {result !== null && !loading && <ResultPanel probability={result} />}
        {error                 && <div className="error-box">{error}</div>}

      </div>
    </div>
  )
}
