import { useEffect } from 'react'
import { Activity, Database, GitBranch, Server } from 'lucide-react'
import { api } from '../api/client'
import { ErrorState, LoadingState, PageHeader, Panel, RequestMeta, StatusBadge } from '../components/Shared'
import { useAnalysis } from '../hooks/useAnalysis'
import type { ReadinessResponse } from '../types/api'

const icons = { configuration: Activity, database: Database, migrations: GitBranch, ollama: Server }

export function SystemStatus() {
  const analysis = useAnalysis<ReadinessResponse>()
  const refresh = () => void analysis.run(() => api.ready())
  useEffect(() => { refresh() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  return <div className="page"><PageHeader eyebrow="Operations" title="System status" description="Dependency-aware readiness without exposing secrets or configuration values." />
    <div className="status-toolbar"><button className="secondary-button" onClick={refresh}>Refresh status</button></div>{analysis.loading && <LoadingState label="Checking local services" />}{analysis.error && <ErrorState error={analysis.error} />}{analysis.result && <><Panel className="overall-status"><div><span className="kicker">Readiness</span><h2>{analysis.result.data.status === 'ready' ? 'All required systems operational' : 'System not ready'}</h2></div><StatusBadge state={analysis.result.data.status} /></Panel><div className="health-grid">{Object.entries(analysis.result.data.checks).map(([name, check]) => { const Icon = icons[name as keyof typeof icons] ?? Activity; return <Panel className="health-card" key={name}><Icon size={20} /><div><h3>{name}</h3><p>{check.detail ?? 'Dependency check passed.'}</p></div><StatusBadge state={check.status} /></Panel> })}</div><RequestMeta requestId={analysis.result.requestId} /></>}
  </div>
}
