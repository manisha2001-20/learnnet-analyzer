import { useEffect, useState } from 'react'
import Overview from './pages/Overview.jsx'
import ConceptGraph from './pages/ConceptGraph.jsx'
import Clusters from './pages/Clusters.jsx'
import LearningPath from './pages/LearningPath.jsx'
import Educators from './pages/Educators.jsx'
import Predict from './pages/Predict.jsx'
import AddVideo from './pages/AddVideo.jsx'

const PAGES = [
  { id: 'overview', label: 'Overview', icon: '◎', component: Overview },
  { id: 'graph', label: 'Concept graph', icon: '⌘', component: ConceptGraph },
  { id: 'clusters', label: 'Clusters', icon: '◍', component: Clusters },
  { id: 'path', label: 'Learning path', icon: '↗', component: LearningPath },
  { id: 'educators', label: 'Educators', icon: '★', component: Educators },
  { id: 'predict', label: 'Predict', icon: '✦', component: Predict },
  { id: 'add', label: 'Add video', icon: '+', component: AddVideo },
]

function currentPage() {
  const id = window.location.hash.replace('#/', '')
  return PAGES.find((p) => p.id === id) ? id : 'overview'
}

export default function App() {
  const [page, setPage] = useState(currentPage)
  const [menuOpen, setMenuOpen] = useState(false)
  const [version, setVersion] = useState(0) // bump to refetch after data changes

  useEffect(() => {
    const onHash = () => { setPage(currentPage()); setMenuOpen(false) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const Page = PAGES.find((p) => p.id === page).component
  return (
    <div className={`shell ${menuOpen ? 'menu-open' : ''}`}>
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">LN</span>
          <div>
            <div className="brand-name">LearnNet</div>
            <div className="brand-sub">Educator network analyzer</div>
          </div>
        </div>
        <nav>
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className={page === p.id ? 'active' : ''}>
              <span className="nav-icon" aria-hidden>{p.icon}</span>{p.label}
            </a>
          ))}
        </nav>
        <div className="sidebar-foot muted small">FastAPI · Sentence-Transformers · NetworkX · scikit-learn · React</div>
      </aside>
      <div className="main">
        <header className="topbar">
          <button className="menu-btn" aria-label="Toggle menu" onClick={() => setMenuOpen((o) => !o)}>☰</button>
          <span className="topbar-title">{PAGES.find((p) => p.id === page).label}</span>
        </header>
        <main className="content" key={`${page}-${version}`}>
          <Page onDataChanged={() => setVersion((v) => v + 1)} />
        </main>
      </div>
    </div>
  )
}
