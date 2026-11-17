import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { useApi } from '../api.js'
import { Card, ErrorBox, Loading, PageHeader } from '../components/ui.jsx'
import { colorFor } from '../utils.js'

function PointTip({ active, payload }) {
  if (!active || !payload?.length) return null
  const v = payload[0].payload
  return (
    <div className="tip"><b>{v.title}</b><div className="muted small">{v.educator} · {v.level}</div><div className="small">{v.cluster_name}</div></div>
  )
}

export default function Clusters() {
  const clusters = useApi('/api/clusters')
  const videos = useApi('/api/videos')
  if (clusters.loading || videos.loading) return <Loading />
  if (clusters.error || videos.error) return <ErrorBox error={clusters.error || videos.error} onRetry={() => { clusters.reload(); videos.reload() }} />
  const sil = Object.entries(clusters.data.silhouette_by_k).map(([k, v]) => ({ k: +k, silhouette: v }))
  return (
    <>
      <PageHeader title="Topic clusters">
        Each video's transcript is embedded with Sentence-Transformers and grouped with K-means. The number of
        clusters (k = {clusters.data.k}) is chosen by the highest silhouette score. Points are a 2-D PCA projection.
      </PageHeader>
      <div className="grid two">
        <Card title="Video map" subtitle="PCA projection of transcript embeddings">
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis type="number" dataKey="x" name="PC1" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
              <YAxis type="number" dataKey="y" name="PC2" tick={{ fill: 'var(--muted)', fontSize: 11 }} />
              <ZAxis range={[90, 90]} />
              <Tooltip content={<PointTip />} />
              {clusters.data.clusters.map((c) => (
                <Scatter key={c.id} name={c.name} data={videos.data.filter((v) => v.cluster === c.id)} fill={colorFor(c.id)} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Choosing k" subtitle="Silhouette score (cosine) for each candidate k">
          <ResponsiveContainer width="100%" height={420}>
            <BarChart data={sil}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="k" tick={{ fill: 'var(--muted)' }} />
              <YAxis tick={{ fill: 'var(--muted)', fontSize: 12 }} />
              <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8 }} />
              <Bar dataKey="silhouette" radius={[6, 6, 0, 0]} fill="var(--primary)" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <div className="grid cluster-cards">
        {clusters.data.clusters.map((c) => (
          <Card key={c.id} title={<span><span className="dot" style={{ background: colorFor(c.id) }} />Cluster {c.id}</span>} subtitle={`${c.size} videos`}>
            <p className="strong">{c.name}</p>
            <ul className="plain small">
              {videos.data.filter((v) => v.cluster === c.id).map((v) => <li key={v.video_id}>{v.title}</li>)}
            </ul>
          </Card>
        ))}
      </div>
    </>
  )
}
