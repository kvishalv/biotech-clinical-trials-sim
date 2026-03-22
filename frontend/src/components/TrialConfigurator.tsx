import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Slider } from '@/components/ui/slider'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'
import { cn, fmtNum } from '@/lib/utils'
import type { SimulationResponse } from '@/types'
import { AlertCircle, CheckCircle2, RefreshCw, Play, Info } from 'lucide-react'

interface ArmAllocation {
  name: string
  label: string
  pct: number
}

const DEFAULT_ARMS: ArmAllocation[] = [
  { name: 'placebo', label: 'Placebo', pct: 33 },
  { name: 'low_dose', label: 'Low Dose', pct: 33 },
  { name: 'high_dose', label: 'High Dose', pct: 34 },
]

interface LintError {
  field: string
  message: string
}

function lintConfig(params: {
  nPatients: number
  nWeeks: number
  seed: number
  arms: ArmAllocation[]
}): LintError[] {
  const errors: LintError[] = []
  const { nPatients, nWeeks, seed, arms } = params

  if (nPatients < 30) {
    errors.push({ field: 'n_patients', message: 'Fewer than 30 patients — underpowered for meaningful ATE estimation.' })
  }
  if (nWeeks < 8) {
    errors.push({ field: 'n_weeks', message: 'Trial duration < 8 weeks — insufficient for biomarker stabilisation.' })
  }
  if (nPatients > 500 && nWeeks > 52) {
    errors.push({ field: 'n_patients', message: 'Large cohort + long duration will significantly increase runtime.' })
  }
  const total = arms.reduce((s, a) => s + a.pct, 0)
  if (Math.abs(total - 100) > 2) {
    errors.push({ field: 'arms', message: `Arm allocations sum to ${total}% — must equal 100%.` })
  }
  const anyZero = arms.some((a) => a.pct === 0)
  if (anyZero) {
    errors.push({ field: 'arms', message: 'One or more arms have 0% allocation — remove or assign patients.' })
  }
  if (seed < 0) {
    errors.push({ field: 'seed', message: 'Seed must be a non-negative integer.' })
  }
  return errors
}

interface Props {
  onSuccess: (result: SimulationResponse) => void
}

export function TrialConfigurator({ onSuccess }: Props) {
  const [nPatients, setNPatients] = useState(200)
  const [nWeeks, setNWeeks] = useState(52)
  const [seed, setSeed] = useState(42)
  const [arms, setArms] = useState<ArmAllocation[]>(DEFAULT_ARMS)
  const [lintOpen, setLintOpen] = useState(true)

  const errors = lintConfig({ nPatients, nWeeks, seed, arms })
  const armTotal = arms.reduce((s, a) => s + a.pct, 0)

  const mutation = useMutation({
    mutationFn: () =>
      api.simulate({
        seed,
        n_patients: nPatients,
        n_weeks: nWeeks,
      }),
    onSuccess: (data) => {
      onSuccess(data)
    },
  })

  const updateArm = useCallback((idx: number, val: number) => {
    setArms((prev) => {
      const next = [...prev]
      next[idx] = { ...next[idx], pct: val }
      return next
    })
  }, [])

  const canRun = errors.length === 0 && !mutation.isPending

  return (
    <div className="max-w-2xl mx-auto py-8 px-6 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Trial Configurator</h1>
        <p className="text-sm text-slate-500 mt-1">
          Configure simulation parameters and run a new trial.
        </p>
      </div>

      {/* Cohort Parameters */}
      <Section title="Cohort Parameters">
        <FormRow label="Patients (n)" hint={`${nPatients}`}>
          <Slider
            min={10}
            max={1000}
            step={10}
            value={[nPatients]}
            onValueChange={([v]) => setNPatients(v)}
            className="flex-1"
          />
          <input
            type="number"
            min={10}
            max={1000}
            value={nPatients}
            onChange={(e) => setNPatients(Number(e.target.value))}
            className="w-20 text-right border border-slate-200 rounded-md px-2 py-1 text-sm"
          />
        </FormRow>

        <FormRow label="Duration (weeks)" hint={`${nWeeks} wk`}>
          <Slider
            min={4}
            max={104}
            step={4}
            value={[nWeeks]}
            onValueChange={([v]) => setNWeeks(v)}
            className="flex-1"
          />
          <input
            type="number"
            min={4}
            max={104}
            value={nWeeks}
            onChange={(e) => setNWeeks(Number(e.target.value))}
            className="w-20 text-right border border-slate-200 rounded-md px-2 py-1 text-sm"
          />
        </FormRow>

        <FormRow label="Random seed">
          <input
            type="number"
            value={seed}
            min={0}
            onChange={(e) => setSeed(Math.max(0, Number(e.target.value)))}
            className="w-32 border border-slate-200 rounded-md px-3 py-1.5 text-sm font-mono"
          />
          <p className="text-xs text-slate-400 flex items-center gap-1">
            <Info className="w-3 h-3" />
            Same seed + config → identical run
          </p>
        </FormRow>
      </Section>

      {/* Arm Allocation */}
      <Section title="Arm Allocation">
        <div className="space-y-4">
          {arms.map((arm, idx) => (
            <div key={arm.name} className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-slate-700">{arm.label}</label>
                <span className="text-sm font-mono text-slate-600">{arm.pct}%</span>
              </div>
              <Slider
                min={0}
                max={100}
                step={1}
                value={[arm.pct]}
                onValueChange={([v]) => updateArm(idx, v)}
              />
            </div>
          ))}

          {/* Allocation bar */}
          <div className="mt-3">
            <div className="flex h-3 rounded-full overflow-hidden">
              {arms.map((arm) => (
                <div
                  key={arm.name}
                  style={{ width: `${(arm.pct / armTotal) * 100}%` }}
                  className={cn(
                    'transition-all',
                    arm.name === 'placebo' && 'bg-slate-300',
                    arm.name === 'low_dose' && 'bg-blue-300',
                    arm.name === 'high_dose' && 'bg-blue-600',
                  )}
                />
              ))}
            </div>
            <div className="flex justify-between mt-1">
              {arms.map((arm) => (
                <span key={arm.name} className="text-xs text-slate-400">
                  {arm.label}: {arm.pct}%
                </span>
              ))}
              <span
                className={cn(
                  'text-xs font-medium',
                  Math.abs(armTotal - 100) > 2 ? 'text-red-500' : 'text-emerald-600',
                )}
              >
                Total: {armTotal}%
              </span>
            </div>
          </div>
          <p className="text-xs text-slate-400">
            Note: arm allocations are visualised here but the backend uses default config values.
            Per-arm overrides will be available in a future release.
          </p>
        </div>
      </Section>

      {/* Protocol Lint Panel */}
      <Section
        title={
          <button
            className="flex items-center gap-2 w-full text-left"
            onClick={() => setLintOpen((o) => !o)}
          >
            <span className="text-sm font-semibold text-slate-700">Protocol Lint</span>
            {errors.length === 0 ? (
              <Badge variant="sig" className="text-xs">
                <CheckCircle2 className="w-3 h-3 mr-1" />
                No issues
              </Badge>
            ) : (
              <Badge variant="destructive" className="text-xs">
                <AlertCircle className="w-3 h-3 mr-1" />
                {errors.length} issue{errors.length > 1 ? 's' : ''}
              </Badge>
            )}
            <span className="ml-auto text-slate-400 text-xs">{lintOpen ? '▲' : '▼'}</span>
          </button>
        }
      >
        {lintOpen && (
          <div className="space-y-2 mt-2">
            {errors.length === 0 ? (
              <div className="flex items-center gap-2 text-emerald-600 text-sm py-2">
                <CheckCircle2 className="w-4 h-4" />
                All protocol checks passed. Ready to run.
              </div>
            ) : (
              errors.map((e, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-sm text-red-700 bg-red-50 rounded-md px-3 py-2"
                >
                  <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <div>
                    <span className="font-medium font-mono text-xs text-red-500 mr-1.5">{e.field}</span>
                    {e.message}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </Section>

      {/* Run Button */}
      <div className="flex items-center gap-4 pt-2">
        <Button
          onClick={() => mutation.mutate()}
          disabled={!canRun}
          className="gap-2"
        >
          {mutation.isPending ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              Running simulation…
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run Simulation
            </>
          )}
        </Button>

        {mutation.isError && (
          <p className="text-sm text-red-600 flex items-center gap-1.5">
            <AlertCircle className="w-4 h-4" />
            {(mutation.error as Error).message}
          </p>
        )}
      </div>

      {/* Runtime hint */}
      <p className="text-xs text-slate-400">
        Estimated runtime: ~{fmtNum((nPatients / 200) * (nWeeks / 52) * 2.5, 1)}s for n={nPatients}, {nWeeks}wk
      </p>
    </div>
  )
}

function Section({ title, children }: { title: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-lg border border-slate-200 p-5">
      {typeof title === 'string' ? (
        <h2 className="text-sm font-semibold text-slate-700 mb-4">{title}</h2>
      ) : (
        <div className="mb-4">{title}</div>
      )}
      {children}
    </div>
  )
}

function FormRow({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center gap-4 py-2">
      <label className="w-40 shrink-0 text-sm text-slate-600">
        {label}
        {hint && <span className="ml-1 font-mono text-xs text-slate-400">({hint})</span>}
      </label>
      <div className="flex-1 flex items-center gap-3">{children}</div>
    </div>
  )
}
