import { useState } from 'react'
import { api } from '../api.js'
import { Card, ErrorBox, LevelBadge, PageHeader } from '../components/ui.jsx'
import { LEVEL_COLORS, pct, wordCount } from '../utils.js'

const EXAMPLE = 'In this lesson we study convolutional neural networks for image classification. A convolution slides a small filter over the image to detect edges, pooling layers shrink the feature maps, and the whole network is trained with backpropagation and gradient descent.'

export default function Predict() {
  const [text, setText] = useState(EXAMPLE)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const words = wordCount(text)

  const run = async (e) => {
    e.preventDefault()
    setBusy(true); setError(null)
    try {
      setResult(await api('/api/predict', { method: 'POST', body: JSON.stringify({ text }) }))
    } catch (err) {
      setError(err.message); setResult(null)
    } finally { setBusy(false) }
  }

  return (
    <>
      <PageHeader title="Predict level & topic of a new video">
        Paste a transcript. The model places it in a topic cluster, predicts its level (a classifier trained on the
        corpus levels) and finds the most similar existing videos.
      </PageHeader>
      <div className="grid two">
        <Card title="Transcript">
          <form onSubmit={run} className="form">
            <textarea rows={11} value={text} onChange={(e) => setText(e.target.value)} aria-label="Transcript" />
            <div className="row">
              <span className={`small ${words < 20 ? 'neg' : 'muted'}`}>{words} words {words < 20 && '(need ≥ 20)'}</span>
              <button className="btn primary" disabled={busy || words < 20}>{busy ? 'Predicting…' : 'Predict'}</button>
            </div>
          </form>
          {error && <ErrorBox error={error} />}
        </Card>
        <Card title="Prediction">
          {!result && <p className="muted">Run a prediction to see the result.</p>}
          {result && (
            <div className="details">
              <p className="big">Level: <LevelBadge level={result.level} /></p>
              {Object.entries(result.probabilities).map(([k, v]) => (
                <div key={k} className="prob">
                  <span>{k}</span>
                  <div className="prob-track"><div style={{ width: pct(v), background: LEVEL_COLORS[k] }} /></div>
                  <span className="num small">{pct(v, 1)}</span>
                </div>
              ))}
              <p className="muted small">Classifier cross-validated accuracy on the corpus: {pct(result.predictor_cv_accuracy, 1)}</p>
              <h4>Topic cluster</h4>
              <p>{result.cluster_name}</p>
              <h4>Concepts recognised</h4>
              <div className="chips">{result.concepts_found.length ? result.concepts_found.map((c) => <span key={c} className="chip">{c}</span>) : <span className="muted small">none</span>}</div>
              <h4>Most similar videos</h4>
              <ul className="plain">
                {result.similar_videos.map((v) => <li key={v.video_id}>{v.title} <span className="muted small">· similarity {v.similarity.toFixed(2)}</span></li>)}
              </ul>
            </div>
          )}
        </Card>
      </div>
    </>
  )
}
