import { useState } from 'react'
import { Activity, BarChart3, ChevronRight, FileSearch, Landmark, Menu, Radar, Scale, X } from 'lucide-react'
import { ClaimEvidence } from './pages/ClaimEvidence'
import { Financial } from './pages/Financial'
import { GroundedQA } from './pages/GroundedQA'
import { Overview } from './pages/Overview'
import { RiskRadarPage } from './pages/RiskRadar'
import { SystemStatus } from './pages/SystemStatus'

const pages = [
  ['overview', 'Overview', Landmark],
  ['qa', 'Grounded Q&A', FileSearch],
  ['financial', 'Financial comparison', BarChart3],
  ['risk', 'Risk Radar', Radar],
  ['claim', 'Claim vs Evidence', Scale],
  ['status', 'System status', Activity],
] as const

export function App() {
  const [page, setPage] = useState('overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const navigate = (next: string) => { setPage(next); setMenuOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }) }
  const content = page === 'qa' ? <GroundedQA /> : page === 'financial' ? <Financial /> : page === 'risk' ? <RiskRadarPage /> : page === 'claim' ? <ClaimEvidence /> : page === 'status' ? <SystemStatus /> : <Overview navigate={navigate} />
  return <div className="app-shell"><button className="mobile-menu" aria-label="Toggle navigation" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X /> : <Menu />}</button><aside className={menuOpen ? 'sidebar open' : 'sidebar'}><div className="brand"><div className="brand-symbol">A</div><div><strong>Due Diligence</strong><span>AI Copilot</span></div></div><nav aria-label="Primary navigation">{pages.map(([key, label, Icon]) => <button className={page === key ? 'active' : ''} key={key} onClick={() => navigate(key)}><Icon size={17} /><span>{label}</span>{page === key && <ChevronRight size={14} />}</button>)}</nav><div className="sidebar-footer"><span className="live-dot" /> Local intelligence stack<small>Validated on Apple filings</small></div></aside><main>{content}</main></div>
}
