import { useState } from 'react'
import { api } from '../api.js'
import { Card, ErrorBox, PageHeader } from '../components/ui.jsx'
import { wordCount } from '../utils.js'

export default function AddVideo({ onDataChanged }) {
  const [form, setForm] = useState({ educator: '', title: '', youtube_url: '', transcript: '' })
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(null); setStatus(null)
    try {
      const body = { ...form, transcript: form.transcript || null, youtube_url: form.youtube_url || null }
      const res = await api('/api/videos', { method: 'POST', body: JSON.stringify(body) })
      setStatus(`Added ${res.added}. The corpus now has ${res.summary.videos} videos and ${res.summary.concepts} concepts.`)
      setForm({ educator: '', title: '', youtube_url: '', transcript: '' })
      setTimeout(onDataChanged, 1500)
    } catch (err) {
      setError(err.message)
    } finally { setBusy(false) }
  }

  return (
    <>
      <PageHeader title="Add a video">
        Paste a transcript, or give a YouTube URL to fetch its captions (needs internet; not every video has
        captions). The whole analysis is rebuilt afterwards.
      </PageHeader>
      <Card className="narrow">
        <form className="form" onSubmit={submit}>
          <label>Educator / channel<input required value={form.educator} onChange={set('educator')} placeholder="Channel name" /></label>
          <label>Video title<input required value={form.title} onChange={set('title')} placeholder="e.g. Gradient descent explained" /></label>
          <label>YouTube URL <span className="muted small">(optional if you paste a transcript)</span><input value={form.youtube_url} onChange={set('youtube_url')} placeholder="https://www.youtube.com/watch?v=…" /></label>
          <label>Transcript <span className="muted small">({wordCount(form.transcript)} words)</span><textarea rows={9} value={form.transcript} onChange={set('transcript')} /></label>
          <button className="btn primary" disabled={busy || (!form.transcript && !form.youtube_url)}>{busy ? 'Adding & re-analysing…' : 'Add video'}</button>
        </form>
        {status && <div className="state ok">{status}</div>}
        {error && <ErrorBox error={error} />}
      </Card>
    </>
  )
}
