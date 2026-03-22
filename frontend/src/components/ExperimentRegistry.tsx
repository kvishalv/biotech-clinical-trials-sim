import { useState, useEffect } from 'react'
import { useMutation } from '@tanstack/react-query'
import type { DriftResponse } from '@/types'
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from '@/components/ui/table'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api, getRegistry, clearRegistry, type RegistryEntry } from '@/lib/api'
import { cn, fmtNum, formatDate, copyToClipboard } from '@/lib/utils'
import type { SimulationResponse } from '@/types'
import {
  Copy,
  Check,
  ArrowUpDown,
  RefreshCw,
  AlertCircle,
  GitCompare,
  Trash2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'

type SortKey = keyof Pick<RegistryEntry, 'timestamp' | 'n_patients' | 'n_weeks' | 'elapsed_seconds'>
type SortDir = 'asc' | 'desc'

interface Props {
  currentRunId: string | null
  onLoadRun: (result: SimulationResponse) => void
}

interface DriftVariable {
  variable: string
  drifted: boolean
  p_value?: number
  test?: string
}

export function ExperimentRegistry({ currentRunId, onLoadRun }: Props) {
  const [entries, setEntries] = useState<RegistryEntry[]>([])
  const [sortKey, setSortKey] = useState<SortKey>('timestamp')
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [compareA, setCompareA] = useState<string | null>(null)
  const [compareB, setCompareB] = useState<string | null>(null)
  const [driftOpen, setDriftOpen] = useState(false)

  // Load from localStorage
  useEffect(() => {
    setEntries(getRegistry())
  }, [])

  // Load a run from the server
  const loadMutation = useMutation({
    mutationFn: (runId: string) => api.getSimulation(runId),
    onSuccess: (data) => onLoadRun(data),
  })

  // Drift detection
  const driftMutation = useMutation({
    mutationFn: () =>
      api.drift({ reference_run_id: compareA!, new_run_id: compareB!, check_biomarkers: true }),
  })

  const handleCopy = (runId: string) => {
    copyToClipboard(runId)
    setCopiedId(runId)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sorted = [...entries].sort((a, b) => {
    const valA = a[sortKey]
    const valB = b[sortKey]
    const cmp = String(valA).localeCompare(String(valB), undefined, { numeric: true })
    return sortDir === 'asc' ? cmp : -cmp
  })

  const canCompare = compareA && compareB && compareA !== compareB

  return (
    <div className="py-8 px-6 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Experiment Registry</h1>
          <p className="text-sm text-slate-500 mt-1">
            {entries.length} run{entries.length !== 1 ? 's' : ''} stored locally
          </p>
        </div>
        <div className="flex items-center gap-2">
          {entries.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                clearRegistry()
                setEntries([])
              }}
              className="gap-1.5 text-red-600 border-red-200 hover:bg-red-50"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear
            </Button>
          )}
        </div>
      </div>

      {entries.length === 0 ? (
        <div className="bg-white rounded-lg border border-slate-200 flex flex-col items-center justify-center py-16 gap-3 text-slate-400">
          <RefreshCw className="w-8 h-8 text-slate-200" />
          <p className="text-sm">No runs recorded yet.</p>
          <p className="text-xs">Run a simulation to populate the registry.</p>
        </div>
      ) : (
        <>
          {/* Compare Banner */}
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <div className="flex items-center gap-3">
              <GitCompare className="w-4 h-4 text-slate-500" />
              <span className="text-sm font-medium text-slate-700">Compare Two Runs</span>
              <div className="flex items-center gap-2 ml-auto">
                <RunSelector
                  label="Reference"
                  entries={entries}
                  value={compareA}
                  onChange={setCompareA}
                  exclude={compareB}
                />
                <span className="text-slate-400 text-xs">vs</span>
                <RunSelector
                  label="New Run"
                  entries={entries}
                  value={compareB}
                  onChange={setCompareB}
                  exclude={compareA}
                />
                <Button
                  onClick={() => {
                    driftMutation.mutate()
                    setDriftOpen(true)
                  }}
                  disabled={!canCompare || driftMutation.isPending}
                  size="sm"
                  className="gap-1.5"
                >
                  {driftMutation.isPending ? (
                    <><RefreshCw className="w-3.5 h-3.5 animate-spin" />Detecting…</>
                  ) : (
                    <><GitCompare className="w-3.5 h-3.5" />Detect Drift</>
                  )}
                </Button>
              </div>
            </div>

            {/* Drift Results Panel */}
            {driftOpen && (driftMutation.data || driftMutation.isError) && (
              <div className="mt-4 border-t border-slate-100 pt-4">
                {driftMutation.isError ? (
                  <div className="flex items-center gap-2 text-red-600 text-sm">
                    <AlertCircle className="w-4 h-4" />
                    {(driftMutation.error as Error).message}
                  </div>
                ) : driftMutation.data ? (
                  <DriftPanel data={driftMutation.data} />
                ) : null}
              </div>
            )}
          </div>

          {/* Registry Table */}
          <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow className="bg-slate-50">
                  <TableHead>Run ID</TableHead>
                  <TableHead>Trial</TableHead>
                  <SortableHead label="Patients" sortKey="n_patients" current={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortableHead label="Weeks" sortKey="n_weeks" current={sortKey} dir={sortDir} onSort={handleSort} />
                  <TableHead>Seed</TableHead>
                  <SortableHead label="Runtime" sortKey="elapsed_seconds" current={sortKey} dir={sortDir} onSort={handleSort} />
                  <SortableHead label="Timestamp" sortKey="timestamp" current={sortKey} dir={sortDir} onSort={handleSort} />
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((entry) => (
                  <TableRow
                    key={entry.run_id}
                    className={cn(
                      'cursor-pointer',
                      entry.run_id === currentRunId && 'bg-blue-50',
                    )}
                    onClick={() => loadMutation.mutate(entry.run_id)}
                  >
                    <TableCell>
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-xs text-slate-700 truncate max-w-[120px]">
                          {entry.run_id}
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleCopy(entry.run_id)
                          }}
                          className="text-slate-300 hover:text-slate-600 transition-colors"
                        >
                          {copiedId === entry.run_id ? (
                            <Check className="w-3 h-3 text-emerald-500" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                        {entry.run_id === currentRunId && (
                          <Badge variant="default" className="text-xs py-0">Active</Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-xs text-slate-600 max-w-[140px] truncate">
                      {entry.trial_name}
                    </TableCell>
                    <TableCell className="font-mono text-xs">{entry.n_patients}</TableCell>
                    <TableCell className="font-mono text-xs">{entry.n_weeks}</TableCell>
                    <TableCell className="font-mono text-xs">{entry.seed}</TableCell>
                    <TableCell className="font-mono text-xs">{fmtNum(entry.elapsed_seconds, 2)}s</TableCell>
                    <TableCell className="text-xs text-slate-500">{formatDate(entry.timestamp)}</TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          loadMutation.mutate(entry.run_id)
                        }}
                        disabled={loadMutation.isPending}
                        className="text-xs text-blue-600 hover:text-blue-700"
                      >
                        {loadMutation.isPending && loadMutation.variables === entry.run_id ? (
                          <RefreshCw className="w-3 h-3 animate-spin" />
                        ) : (
                          'Load'
                        )}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </>
      )}
    </div>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function SortableHead({
  label,
  sortKey,
  current,
  dir,
  onSort,
}: {
  label: string
  sortKey: SortKey
  current: SortKey
  dir: SortDir
  onSort: (k: SortKey) => void
}) {
  const active = current === sortKey
  return (
    <TableHead>
      <button
        onClick={() => onSort(sortKey)}
        className="flex items-center gap-1 hover:text-slate-700 transition-colors"
      >
        {label}
        {active ? (
          dir === 'asc' ? (
            <ChevronUp className="w-3 h-3" />
          ) : (
            <ChevronDown className="w-3 h-3" />
          )
        ) : (
          <ArrowUpDown className="w-3 h-3 opacity-40" />
        )}
      </button>
    </TableHead>
  )
}

function RunSelector({
  label,
  entries,
  value,
  onChange,
  exclude,
}: {
  label: string
  entries: RegistryEntry[]
  value: string | null
  onChange: (v: string) => void
  exclude: string | null
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-slate-500">{label}:</span>
      <select
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        className="text-xs border border-slate-200 rounded px-2 py-1 bg-white max-w-[160px] font-mono"
      >
        <option value="">Select run…</option>
        {entries
          .filter((e) => e.run_id !== exclude)
          .map((e) => (
            <option key={e.run_id} value={e.run_id}>
              {e.run_id.slice(0, 20)}… (n={e.n_patients})
            </option>
          ))}
      </select>
    </div>
  )
}

function DriftPanel({ data }: { data: DriftResponse }) {
  return (
    <div>
      <div className="flex items-center gap-3 mb-3">
        <span className="text-sm font-medium text-slate-700">Drift Detection Results</span>
        <Badge variant={data.overall_drift ? 'destructive' : 'sig'}>
          {data.overall_drift ? `${data.n_drifted} variables drifted` : 'No drift detected'}
        </Badge>
        <span className="text-xs text-slate-400">
          {data.n_drifted} / {data.n_tested} variables
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
        {data.results.map((r) => (
          <div
            key={r.variable}
            className={cn(
              'rounded-md px-3 py-2 text-xs border',
              r.drifted
                ? 'bg-red-50 border-red-200 text-red-700'
                : 'bg-emerald-50 border-emerald-200 text-emerald-700',
            )}
          >
            <div className="font-medium truncate capitalize">{String(r.variable).replace(/_/g, ' ')}</div>
            <div className="font-mono mt-0.5 opacity-70">
              {r.drifted ? 'DRIFTED' : 'OK'}
              {r.p_value != null && ` · p=${fmtNum(Number(r.p_value), 3)}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
