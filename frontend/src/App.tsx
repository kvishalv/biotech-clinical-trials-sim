import { useState } from 'react'
import { Sidebar } from '@/components/Sidebar'
import { TrialConfigurator } from '@/components/TrialConfigurator'
import { RunDashboard } from '@/components/RunDashboard'
import { BiomarkerExplorer } from '@/components/BiomarkerExplorer'
import { CausalOutcomes } from '@/components/CausalOutcomes'
import { ExperimentRegistry } from '@/components/ExperimentRegistry'
import { addToRegistry } from '@/lib/api'
import type { SimulationResponse, View } from '@/types'

export default function App() {
  const [view, setView] = useState<View>('configurator')
  const [currentResult, setCurrentResult] = useState<SimulationResponse | null>(null)

  const handleSimulationSuccess = (result: SimulationResponse) => {
    setCurrentResult(result)
    addToRegistry(result)
    setView('run-dashboard')
  }

  const handleLoadRun = (result: SimulationResponse) => {
    setCurrentResult(result)
    setView('run-dashboard')
  }

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar active={view} onChange={setView} hasRun={!!currentResult} />

      <main className="flex-1 overflow-y-auto">
        {view === 'configurator' && (
          <TrialConfigurator onSuccess={handleSimulationSuccess} />
        )}

        {view === 'run-dashboard' && currentResult && (
          <RunDashboard
            result={currentResult}
            onOpenBiomarkers={() => setView('biomarker-explorer')}
          />
        )}

        {view === 'run-dashboard' && !currentResult && (
          <EmptyState
            title="No run loaded"
            message="Run a simulation from the Trial Configurator, or load one from the Registry."
          />
        )}

        {view === 'biomarker-explorer' && currentResult && (
          <BiomarkerExplorer
            runId={currentResult.run_id}
            nWeeks={currentResult.n_weeks}
            burninWeeks={4}
          />
        )}

        {view === 'biomarker-explorer' && !currentResult && (
          <EmptyState
            title="No run loaded"
            message="Run a simulation first to explore biomarker trajectories."
          />
        )}

        {view === 'causal-outcomes' && currentResult && (
          <CausalOutcomes result={currentResult} />
        )}

        {view === 'causal-outcomes' && !currentResult && (
          <EmptyState
            title="No run loaded"
            message="Run a simulation first to view causal analysis."
          />
        )}

        {view === 'registry' && (
          <ExperimentRegistry
            currentRunId={currentResult?.run_id ?? null}
            onLoadRun={handleLoadRun}
          />
        )}
      </main>
    </div>
  )
}

function EmptyState({ title, message }: { title: string; message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[400px] gap-3 text-slate-400">
      <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
        <span className="text-2xl">⚗️</span>
      </div>
      <p className="text-base font-medium text-slate-600">{title}</p>
      <p className="text-sm text-center max-w-xs">{message}</p>
    </div>
  )
}
