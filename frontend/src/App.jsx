/**
 * App.jsx — Root component for the Ham / Spam Checker UI.
 *
 * Architecture overview:
 *   <App>                — manages all state and API calls
 *     <InfoTooltip>      — hover badge showing both model metadata side-by-side
 *     <ExamplesPanel>    — expandable panel of canned texts (shows on button click)
 *     <ResultPanel>      — colour-coded score + progress bar (one per model)
 *
 * Data flow:
 *   User types (or selects canned text)
 *     → useEffect fires after 600 ms debounce
 *       → POST /predict { text }
 *         → setState({ tfidf, mbert })
 *           → two <ResultPanel>s re-render with new scores
 */

import { useState, useEffect, useRef } from 'react'

// ---------------------------------------------------------------------------
// Model metadata — displayed in the <InfoTooltip> hover card.
// ---------------------------------------------------------------------------
const TFIDF_ROWS = [
  { key: 'Type',         val: 'Classical ML' },
  { key: 'Vectorizer',   val: 'TF-IDF (5,000 features)' },
  { key: 'Architecture', val: '64 → 32 → Sigmoid' },
  { key: 'Framework',    val: 'TensorFlow / Keras' },
  { key: 'Epochs',       val: '5' },
  { key: 'Speed',        val: 'Very fast (~1 ms)' },
]

const MBERT_ROWS = [
  { key: 'Type',         val: 'Transformer (BERT)' },
  { key: 'Base model',   val: 'bert-base-multilingual-cased' },
  { key: 'Architecture', val: '12-layer, 179M params' },
  { key: 'Framework',    val: 'PyTorch / HuggingFace' },
  { key: 'Epochs',       val: '2  (fine-tuned)' },
  { key: 'Speed',        val: 'Slower (~200 ms CPU)' },
]

const SHARED_STATS = [
  { key: 'Training data',   val: '5,171 labelled emails' },
  { key: 'Train / test',    val: '80% / 20% split' },
  { key: 'TF-IDF accuracy', val: '~98%' },
  { key: 'mBERT accuracy',  val: '98.84% (epoch 2)' },
]

// ---------------------------------------------------------------------------
// Canned example messages — grouped by expected model behaviour.
//
// WHY three categories?
//   "Both say Ham"    — shows the models agree on obvious legitimate mail
//   "Both say Spam"   — shows the models agree on obvious spam
//   "May differ"      — the interesting cases:
//     TF-IDF is keyword-based so it flags trigger words regardless of context.
//     mBERT reads full context so it can tell "free tickets (work award)" from
//     "free prize click here", and can catch subtle sales pitches that contain
//     no classic spam words at all.
// ---------------------------------------------------------------------------
const CANNED_EXAMPLES = [
  {
    categoryLabel: '✅ Both say Ham',
    categoryDesc:  'Clear legitimate messages — both models should give a low spam score.',
    items: [
      {
        label: 'Meeting reminder',
        text:
          'Subject: Team sync tomorrow at 2pm\n\n' +
          'Hi everyone,\n\n' +
          'Just a reminder that our weekly team sync is scheduled for tomorrow at 2:00 pm in ' +
          'Conference Room B. Please review the agenda I sent on Monday and come prepared with ' +
          'your project updates.\n\n' +
          'Looking forward to seeing you all there.\n\n' +
          'Best,\nSarah',
      },
      {
        label: 'Catch-up from a friend',
        text:
          'Hey!\n\n' +
          'It was so great running into you at the conference last week. We should definitely ' +
          'catch up properly — are you free for lunch any day next week? I know a great new ' +
          'Italian place that just opened near the office.\n\n' +
          'Let me know what works for you.\n\nCheers,\nMike',
      },
    ],
  },
  {
    categoryLabel: '🚨 Both say Spam',
    categoryDesc:  'Classic spam patterns — both models should give a high spam score.',
    items: [
      {
        label: 'Prize winner!!!',
        text:
          'CONGRATULATIONS!!! You have been SELECTED as our LUCKY WINNER!\n\n' +
          'Claim your FREE $1,000 gift card RIGHT NOW! This is a limited time offer — ' +
          'act IMMEDIATELY before it expires! Click the link below to claim your prize. ' +
          'No purchase necessary. 100% guaranteed. Winner selected from millions of entries.\n\n' +
          'CLICK HERE TO CLAIM YOUR FREE PRIZE NOW!!!',
      },
      {
        label: 'Account suspended',
        text:
          'URGENT NOTICE: Your account has been suspended due to suspicious activity.\n\n' +
          'You must verify your personal details IMMEDIATELY to restore access. ' +
          'Failure to respond within 24 hours will result in permanent account closure ' +
          'and loss of all funds.\n\n' +
          'Verify now at: http://secure-account-verify-login.com/confirm\n\n' +
          'Enter your username, password, and credit card number to confirm your identity.',
      },
    ],
  },
  {
    categoryLabel: '🔀 Models diverge',
    categoryDesc:
      'Each example exposes a different strength or weakness. ' +
      'Watch how the two scores pull apart — and think about which model is right.',
    items: [
      {
        label: 'Job offer — TF-IDF over-triggers',
        text:
          'Subject: Offer letter – Software Engineer role\n\n' +
          'Hi Alex,\n\n' +
          'Congratulations! We are pleased to formally offer you the Software Engineer ' +
          'position at Meridian Systems.\n\n' +
          'This offer is time-sensitive – we need your acceptance confirmed within 48 hours ' +
          'as we are coordinating start dates across several candidates. Please click the link ' +
          'below to sign your offer letter and confirm your start date.\n\n' +
          'We look forward to welcoming you to the team.\n\n' +
          'Best regards,\nClaire Sutton\nTalent Acquisition, Meridian Systems',
      },
      {
        label: 'Spanish spam — mBERT multilingual',
        text:
          'Asunto: \u00a1FELICITACIONES! Ha sido seleccionado como nuestro GANADOR\n\n' +
          'Estimado usuario,\n\n' +
          'Su direcci\u00f3n de correo electr\u00f3nico ha sido seleccionada en nuestro sorteo ' +
          'internacional. Ha ganado la incre\u00edble suma de 85.000 euros. Para reclamar su ' +
          'premio AHORA, haga clic en el enlace de abajo e ingrese su informaci\u00f3n personal.\n\n' +
          '\u00a1Esta oferta expira en 24 horas! No pierda esta oportunidad \u00fanica en la vida.\n\n' +
          'HAGA CLIC AQU\u00cd PARA RECLAMAR SU PREMIO GRATIS AHORA',
      },
      {
        label: 'Grandparent scam — mBERT fooled',
        text:
          'Hi Nan,\n\n' +
          'It is me, your grandson Jake. I am in a bit of trouble and too embarrassed ' +
          'to call Mum. I had an accident in the car abroad and I need some money to pay ' +
          'the local garage before they will release the car. It is only 400 pounds. ' +
          'I promise I will pay you back next week.\n\n' +
          'Can you send it to my friend\'s account? He is here with me. ' +
          'I will text you his details.\n\n' +
          'Please do not tell anyone just yet, I do not want Mum to worry.\n\n' +
          'Love, Jake',
      },
    ],
  },
]

// ---------------------------------------------------------------------------
// <InfoTooltip>
// A small pill in the top-right corner of the card.
// Shows a two-column comparison of both models on hover.
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

      {visible && (
        <div className="info-tooltip info-tooltip-wide">
          <div className="info-heading">About these classifiers</div>

          <div className="info-cols">
            <div>
              <div className="info-col-heading">TF-IDF Model</div>
              {TFIDF_ROWS.map(({ key, val }) => (
                <div className="info-row" key={key}>
                  <span className="info-key">{key}</span>
                  <span className="info-val">{val}</span>
                </div>
              ))}
            </div>
            <div>
              <div className="info-col-heading">mBERT Model</div>
              {MBERT_ROWS.map(({ key, val }) => (
                <div className="info-row" key={key}>
                  <span className="info-key">{key}</span>
                  <span className="info-val">{val}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="info-shared">
            <div className="info-col-heading">Shared training details</div>
            {SHARED_STATS.map(({ key, val }) => (
              <div className="info-row" key={key}>
                <span className="info-key">{key}</span>
                <span className="info-val">{val}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// <ExamplesPanel>
// Expandable panel showing canned texts grouped by category.
// Clicking a chip calls onSelect(text) which populates the textarea.
// ---------------------------------------------------------------------------
function ExamplesPanel({ onSelect }) {
  return (
    <div className="examples-panel">
      {CANNED_EXAMPLES.map((group) => (
        <div className="examples-group" key={group.categoryLabel}>
          <div className="examples-group-label">{group.categoryLabel}</div>
          <div className="examples-group-desc">{group.categoryDesc}</div>
          <div className="examples-chips">
            {group.items.map((item) => (
              <button
                key={item.label}
                className="example-chip"
                onClick={() => onSelect(item.text)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// <ResultPanel>
// Renders the spam score for one model.
//
// Three states:
//   probability=null, loading=false → placeholder  (card always occupies space)
//   probability=null, loading=true  → pulsing bar  (API call in flight)
//   probability=float, loading=false → real result  (color-coded score)
// ---------------------------------------------------------------------------
function ResultPanel({ modelName, probability, loading }) {
  const isReady = probability !== null

  const pct    = isReady ? Math.round(probability * 100) : 0
  const isSpam = isReady && pct >= 50
  const color  = isReady ? (isSpam ? '#f87171' : '#4ade80') : '#475569'
  const label  = loading  ? 'Analysing…'
               : isReady  ? (isSpam ? '🚨 Likely Spam' : '✅ Looks Like Ham')
               :             '— awaiting input —'

  return (
    <div className={`result-panel${!isReady && !loading ? ' result-panel--placeholder' : ''}`}>
      <div className="result-model-name">{modelName}</div>
      <div className="result-label" style={{ color }}>{label}</div>

      <div className="progress-bg">
        <div
          className={`progress-fill${loading ? ' progress-fill--loading' : ''}`}
          style={{ width: loading ? '100%' : `${pct}%`, background: color }}
        />
      </div>

      <p className="result-pct">
        {isReady
          ? <><span style={{ color, fontWeight: 700, fontSize: '1.4rem' }}>{pct}%</span>
               <span className="result-pct-label"> spam likelihood</span></>
          : <span className="result-pct-label">{loading ? '…' : '—'}</span>
        }
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// <App> — main component
// ---------------------------------------------------------------------------
export default function App() {
  const [text,         setText]         = useState('')
  const [loading,      setLoading]      = useState(false)
  const [result,       setResult]       = useState(null)
  const [error,        setError]        = useState(null)
  const [showExamples, setShowExamples] = useState(false)

  const debounceRef = useRef(null)

  // ── API call ─────────────────────────────────────────────────────────────
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

      // Guard: if fields are missing the old api.py is probably still running
      if (typeof data.tfidf !== 'number' || typeof data.mbert !== 'number') {
        throw new Error('Unexpected response from API — restart api.py with the latest code')
      }

      setResult({ tfidf: data.tfidf, mbert: data.mbert })

    } catch (err) {
      console.error('[SpamChecker] predict error:', err)
      setError('Could not reach the API. Is api.py running on port 5000?')

    } finally {
      setLoading(false)
    }
  }

  // ── Debounced effect ─────────────────────────────────────────────────────
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)

    if (!text.trim()) {
      setResult(null)
      setError(null)
      return
    }

    debounceRef.current = setTimeout(() => checkText(text), 600)
    return () => clearTimeout(debounceRef.current)
  }, [text])

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleClear = () => {
    setText('')
    setResult(null)
    setError(null)
  }

  const handleSelectExample = (exampleText) => {
    setText(exampleText)
    setShowExamples(false)  // collapse the panel after selection
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      <div className="card">

        <InfoTooltip />

        <div className="card-header">
          <span className="shield-icon">🛡️</span>
          <h1 className="title">Ham / Spam Checker</h1>
          <p className="subtitle">
            Start typing — both models score automatically
            <span className="live-badge">● LIVE</span>
          </p>
        </div>

        <textarea
          className="textarea"
          rows={8}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your email or message here, or try a canned example below…"
        />

        <div className="button-row">
          <button
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={!text && result === null}
          >
            🗑️ Clear
          </button>
          <button
            className={`btn btn-secondary${showExamples ? ' btn-active' : ''}`}
            onClick={() => setShowExamples(v => !v)}
          >
            📋 {showExamples ? 'Hide Examples' : 'Try Examples'}
          </button>
        </div>

        {showExamples && (
          <ExamplesPanel onSelect={handleSelectExample} />
        )}

        {/* Results row is ALWAYS rendered so card height never changes.
            probability=null shows a placeholder; loading=true shows a pulse. */}
        <div className="results-row">
          <ResultPanel modelName="TF-IDF" probability={result?.tfidf ?? null} loading={loading} />
          <ResultPanel modelName="mBERT"  probability={result?.mbert ?? null} loading={loading} />
        </div>

        {error && <div className="error-box">{error}</div>}

      </div>
    </div>
  )
}
