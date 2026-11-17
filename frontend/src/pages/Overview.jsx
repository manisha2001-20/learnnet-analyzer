import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useApi } from '../api.js'
import { Card, ErrorBox, LevelBadge, Loading, PageHeader, Stat } from '../components/ui.jsx'
import { LEVEL_COLORS, formatNumber, pct } from '../utils.js'

export default function Overview() {
  const summary = useApi('/api/summary')
  const concepts = useApi('/api/concepts')
  const videos = useApi('/api/videos')
  if (summary.loading) return <Loading text="Analysing transcripts…" />
  if (summary.error) return <ErrorBox error={summary.error} onRetry={summary.reload} />
  const s = summary.data
  const levels = Object.entries(s.level_counts).map(([name, value]) => ({ name, value }))
  const topConcepts = (concepts.data || []).slice(0, 12).map((c) => ({ concept: c.concept, pagerank: +(c.pagerank * 100).toFixed(2) }))

  return (
    <>
      <PageHeader title="YouTube educator network study">
        Transcripts are turned into concepts, semantic embeddings, topic clusters, a concept graph and
        beginner-to-advanced learning paths.
        {s.synthetic_videos > 0 && <> <span className="badge warn">{s.synthetic_videos} synthetic sample videos</span></>}
      </PageHeader>
      <div className="grid stats">
        <Stat label="Videos" value={s.videos} hint={`${s.educators} educators`} />
        <Stat label="Concepts" value={s.concepts} hint={`${s.concept_edges} relations`} />
        <Stat label="Prerequisite links" value={s.prerequisite_edges} hint="RefD + teaching order" />
        <Stat label="Topic clusters" value={s.clusters} hint={`silhouette ${s.silhouette.toFixed(3)}`} />
        <Stat label="Level predictor" value={pct(s.predictor.cv_accuracy)} hint={`${s.predictor.cv_folds}-fold CV · baseline ${pct(s.predictor.majority_baseline)}`} />
      </div>
      <div className="grid two">
        <Card title="Most central concepts" subtitle="PageRank in the concept graph (×100)">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={topConcepts} layout="vertical" margin={{ left: 20, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--grid)" />
              <XAxis type="number" tick={{ fill: 'var(--muted)', fontSize: 12 }} />
              <YAxis type="category" dataKey="concept" width={150} tick={{ fill: 'var(--text)', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Bar dataKey="pagerank" fill="var(--primary)" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Videos by level" subtitle="From concept difficulty + readability">
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={levels} dataKey="value" nameKey="name" innerRadius={70} outerRadius={110} paddingAngle={3} label={(e) => `${e.name} (${e.value})`}>
                {levels.map((l) => <Cell key={l.name} fill={LEVEL_COLORS[l.name]} />)}
              </Pie>
              <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <Card title="Videos" subtitle={`Embedder: ${s.embedder} · built in ${s.build_seconds}s`}>
        {videos.loading ? <Loading /> : videos.error ? <ErrorBox error={videos.error} /> : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Title</th><th>Educator</th><th>Level</th><th>Cluster</th><th>Key concepts</th><th className="num">Views</th></tr></thead>
              <tbody>
                {videos.data.map((v) => (
                  <tr key={v.video_id}>
                    <td className="strong">{v.url ? <a href={v.url} target="_blank" rel="noreferrer">{v.title}</a> : v.title}</td>
                    <td>{v.educator}</td>
                    <td><LevelBadge level={v.level} /></td>
                    <td className="muted small">{v.cluster_name}</td>
                    <td className="small">{v.top_concepts.slice(0, 4).join(', ')}</td>
                    <td className="num">{formatNumber(v.views)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  )
}
