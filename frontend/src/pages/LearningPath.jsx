import { useState } from 'react'
import { useApi } from '../api.js'
import { Card, ErrorBox, LevelBadge, Loading, PageHeader } from '../components/ui.jsx'

export default function LearningPath() {
  const initial = sessionStorage.getItem('ln-goal') || 'backpropagation'
  const [input, setInput] = useState(initial)
  const [goal, setGoal] = useState(initial)
  const concepts = useApi('/api/concepts')
  const path = useApi(goal ? `/api/learning-path?goal=${encodeURIComponent(goal)}` : '/api/learning-path')

  const submit = (e) => {
    e.preventDefault()
    sessionStorage.setItem('ln-goal', input)
    setGoal(input.trim())
  }

  return (
    <>
      <PageHeader title="Learning path generator">
        Pick a goal concept. The path lists its prerequisites from the prerequisite graph in topological order
        (easier first), and recommends a video for each step.
      </PageHeader>
      <form className="toolbar" onSubmit={submit}>
        <input list="concept-list" value={input} onChange={(e) => setInput(e.target.value)} placeholder="e.g. backpropagation" aria-label="Goal concept" />
        <datalist id="concept-list">{concepts.data?.map((c) => <option key={c.concept} value={c.concept} />)}</datalist>
        <button className="btn primary" type="submit">Generate path</button>
        <button className="btn" type="button" onClick={() => { setInput(''); setGoal('') }}>Full curriculum</button>
      </form>
      {path.loading && <Loading text="Building path…" />}
      {path.error && <ErrorBox error={path.error} />}
      {path.data && (
        <Card title={path.data.goal ? `Path to “${path.data.goal}”` : 'Full curriculum (all concepts)'} subtitle={`${path.data.steps.length} steps · beginner → advanced`}>
          {goal && path.data.goal && path.data.goal !== goal.toLowerCase() && (
            <p className="note">“{goal}” is not an extracted concept, so the closest match <b>{path.data.goal}</b> was used.</p>
          )}
          <ol className="timeline">
            {path.data.steps.map((s) => (
              <li key={s.concept} className={`step level-${s.level}`}>
                <div className="step-num">{s.step}</div>
                <div className="step-body">
                  <div className="step-title">{s.concept} <LevelBadge level={s.level} /></div>
                  {s.video ? (
                    <div className="muted small">▶ {s.video.url ? <a href={s.video.url} target="_blank" rel="noreferrer">{s.video.title}</a> : s.video.title} · {s.video.educator}</div>
                  ) : <div className="muted small">No dedicated video found</div>}
                </div>
                <div className="step-diff" title="difficulty"><div style={{ width: `${Math.round(s.difficulty * 100)}%` }} /></div>
              </li>
            ))}
          </ol>
        </Card>
      )}
    </>
  )
}
