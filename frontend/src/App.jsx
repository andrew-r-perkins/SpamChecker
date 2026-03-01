/**
 * App.jsx — Root component for the Ham / Spam Checker UI.
 *
 * Architecture overview:
 *   <App>                — manages all state and API calls
 *     <InfoTooltip>      — hover badge showing active model metadata
 *     <ExamplesPanel>    — expandable panel of canned texts (shows on button click)
 *     <ResultPanel>      — colour-coded score + progress bar (one per active model)
 *
 * Data flow:
 *   On mount  → GET /config → sets activeModels ([{key, name}, ...])
 *   User types → useEffect fires after 600 ms debounce
 *     → POST /predict { text }
 *       → setState(scores)  e.g. { tfidf: 0.97, mbert: 0.95, distilbert: 0.94 }
 *         → one <ResultPanel> per active model re-renders with new score
 *
 * Which models render depends entirely on what the server loaded at startup
 * (controlled by api.py --models flag). No frontend changes needed when
 * adding or removing models.
 */

import { useState, useEffect, useRef } from 'react'

// ---------------------------------------------------------------------------
// Model metadata — displayed in the <InfoTooltip> hover card.
// Add a new entry here whenever a new model is added to the backend.
// ---------------------------------------------------------------------------
const MODEL_INFO = {
  tfidf: {
    heading: 'TF-IDF + NN',
    rows: [
      { key: 'Type',          val: 'Classical ML' },
      { key: 'Vectorizer',    val: 'TF-IDF (5,000 features)' },
      { key: 'Architecture',  val: '64 → 32 → Sigmoid' },
      { key: 'Framework',     val: 'TensorFlow / Keras' },
      { key: 'Epochs',        val: '5' },
      { key: 'Test accuracy', val: '~98%' },
      { key: 'Speed',         val: 'Very fast (~1 ms)' },
    ],
  },
  mbert: {
    heading: 'mBERT',
    rows: [
      { key: 'Type',          val: 'Transformer (BERT)' },
      { key: 'Base model',    val: 'bert-base-multilingual-cased' },
      { key: 'Architecture',  val: '12-layer, 179M params' },
      { key: 'Framework',     val: 'PyTorch / HuggingFace' },
      { key: 'Epochs',        val: '2  (fine-tuned)' },
      { key: 'Test accuracy', val: '98.84% (epoch 2)' },
      { key: 'Speed',         val: 'Slower (~200 ms CPU)' },
    ],
  },
  distilbert: {
    heading: 'DistilBERT',
    rows: [
      { key: 'Type',          val: 'Transformer (distilled BERT)' },
      { key: 'Base model',    val: 'distilbert-base-multilingual-cased' },
      { key: 'Architecture',  val: '6-layer, 66M params' },
      { key: 'Framework',     val: 'PyTorch / HuggingFace' },
      { key: 'Epochs',        val: '2  (fine-tuned)' },
      { key: 'Test accuracy', val: 'TBD — needs training' },
      { key: 'Speed',         val: '~2× faster than mBERT' },
    ],
  },
}

const SHARED_STATS = [
  { key: 'Training data', val: '5,171 labelled emails' },
  { key: 'Train / test',  val: '80% / 20% split' },
  { key: 'Label split',   val: 'Ham 71% / Spam 29%' },
]

// ---------------------------------------------------------------------------
// Canned example messages — grouped by expected model behaviour.
// ---------------------------------------------------------------------------
const CANNED_EXAMPLES = [
  {
    categoryLabel: '✅ Both say Ham',
    categoryDesc:  'Clear legitimate messages — all models should give a low spam score.',
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
    categoryDesc:  'Classic spam patterns — all models should give a high spam score.',
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
      'Watch how scores pull apart — and think about which model is right.',
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
        label: 'Spanish spam — multilingual edge',
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
        label: 'Grandparent scam — context fools BERT',
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
// Renders one column per active model on hover.
// ---------------------------------------------------------------------------
function InfoTooltip({ activeModels }) {
  const [visible, setVisible] = useState(false)

  // Tooltip width scales with number of active models
  const tooltipWidth =
    activeModels.length <= 1 ? 300
    : activeModels.length === 2 ? 520
    : 700

  return (
    <div
      className="info-trigger"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      ⓘ Model Info

      {visible && activeModels.length > 0 && (
        <div className="info-tooltip info-tooltip-wide" style={{ width: tooltipWidth }}>
          <div className="info-heading">About these classifiers</div>

          <div
            className="info-cols"
            style={{ gridTemplateColumns: `repeat(${activeModels.length}, 1fr)` }}
          >
            {activeModels.map(({ key }) => {
              const info = MODEL_INFO[key]
              if (!info) return null
              return (
                <div key={key}>
                  <div className="info-col-heading">{info.heading}</div>
                  {info.rows.map(({ key: rowKey, val }) => (
                    <div className="info-row" key={rowKey}>
                      <span className="info-key">{rowKey}</span>
                      <span className="info-val">{val}</span>
                    </div>
                  ))}
                </div>
              )
            })}
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
  const [activeModels, setActiveModels] = useState([])   // [{key, name}, ...]
  const [text,         setText]         = useState('')
  const [loading,      setLoading]      = useState(false)
  const [result,       setResult]       = useState(null) // { tfidf: float, ... }
  const [error,        setError]        = useState(null)
  const [showExamples, setShowExamples] = useState(false)

  const debounceRef = useRef(null)

  // ── Fetch active models from server on mount ──────────────────────────────
  useEffect(() => {
    fetch('/config')
      .then(r => r.json())
      .then(d => setActiveModels(d.models ?? []))
      .catch(() => {
        // Fallback: show TF-IDF panel so UI is never completely empty
        setActiveModels([{ key: 'tfidf', name: 'TF-IDF + NN' }])
        setError('Could not load /config. Is api.py running on port 5000?')
      })
  }, [])

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

      // Guard: ensure at least one active model returned a numeric score
      const hasValid = activeModels.some(m => typeof data[m.key] === 'number')
      if (!hasValid) {
        throw new Error('Unexpected response from API — restart api.py with the latest code')
      }

      setResult(data)

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
    setShowExamples(false)
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      <div className="card">

        <InfoTooltip activeModels={activeModels} />

        <div className="card-header">
          <span className="shield-icon">🛡️</span>
          <h1 className="title">Ham / Spam Checker</h1>
          <p className="subtitle">
            Start typing — active models score automatically
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
            One panel per active model. probability=null shows a placeholder;
            loading=true shows a pulse animation. */}
        <div className="results-row">
          {activeModels.map(m => (
            <ResultPanel
              key={m.key}
              modelName={m.name}
              probability={result ? (result[m.key] ?? null) : null}
              loading={loading}
            />
          ))}
        </div>

        {error && <div className="error-box">{error}</div>}

      </div>
    </div>
  )
}
