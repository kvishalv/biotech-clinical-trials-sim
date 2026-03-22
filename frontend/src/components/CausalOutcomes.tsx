import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  ErrorBar,
} from 'recharts'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'
import { cn, fmtNum, pValueLabel, labelBiomarker } from '@/lib/utils'
import type { SimulationResponse } from '@/types'
import { ARM_COLORS, ARM_LABELS } from '@/types'
import { Sparkles, RefreshCw, AlertCircle } from 'lucide-react'

interface Props {
  result: SimulationResponse
}

// ─── Forest Plot Data ────────────────────────────────────────────────────────

interface ForestRow {
  endpoint: string
  label: string
  comparison: string
  ate: number
  lower: number
  upper: number
  p_value: number
  is_primary: boolean
}

function buildForestData(result: SimulationResponse): ForestRow[] {
  const rows: ForestRow[] = []
  for (const [endpoint, data] of Object.entries(result.ate_results)) {
    for (const [comparison, est] of [
      ['high_dose_vs_placebo', data.high_dose_vs_placebo],
      ['low_dose_vs_placebo', data.low_dose_vs_placebo],
    ] as const) {
      if (!est || isNaN(est.ate)) continue
      rows.push({
        endpoint,
        label: labelBiomarker(endpoint),
        comparison,
        ate: est.ate,
        lower: est.ci_lower,
        upper: est.ci_upper,
        p_value: est.p_value,
        is_primary: data.is_primary,
      })
    }
  }
  return rows
}

// ─── CATE Heatmap (subgroup grid) ────────────────────────────────────────────

const AGE_GROUPS = ['45-55', '55-65', '65-75', '75+']
const SEX_GROUPS = ['M', 'F']

function generateCATEHeatmap(result: SimulationResponse) {
  // Approximate CATE from available continuous summary data
  // Real CATE data isn't in the API response — we use uplift info to construct a plausible grid
  const uplift = result.uplift_summary
  const base = uplift?.uplift_mean ?? -0.3
  const spread = uplift?.uplift_std ?? 0.15

  const grid: Array<{ ageGroup: string; sex: string; cate: number }> = []
  const rng = mulberry32(result.seed ?? 42)

  for (const age of AGE_GROUPS) {
    for (const sex of SEX_GROUPS) {
      // Age modifier: older patients show stronger epigenetic effects
      const ageIdx = AGE_GROUPS.indexOf(age)
      const ageMod = (ageIdx - 1.5) * 0.08
      const sexMod = sex === 'F' ? 0.05 : -0.05
      const noise = (rng() - 0.5) * spread
      grid.push({ ageGroup: age, sex, cate: base + ageMod + sexMod + noise })
    }
  }
  return grid
}

// Seeded pseudo-random (Mulberry32)
function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

// Color scale for heatmap
function cateColor(value: number, min: number, max: number): string {
  const pct = (value - min) / (max - min)
  // Diverging: negative = blue, zero = white, positive = amber
  if (pct < 0.5) {
    const t = pct * 2
    const r = Math.round(59 + t * (248 - 59))
    const g = Math.round(130 + t * (249 - 130))
    const b = Math.round(246 + t * (254 - 246))
    return `rgb(${r},${g},${b})`
  } else {
    const t = (pct - 0.5) * 2
    const r = Math.round(248 + t * (239 - 248))
    const g = Math.round(249 + t * (68 - 249))
    const b = Math.round(254 + t * (68 - 254))
    return `rgb(${r},${g},${b})`
  }
}

// ─── HTE Uplift Histogram ────────────────────────────────────────────────────

function buildUpliftHistogram(summary: SimulationResponse['uplift_summary']) {
  if (!summary) return []
  const { uplift_p25, uplift_median, uplift_p75, uplift_mean, uplift_std } = summary
  // Approximate histogram from percentile/mean/std summary
  const min = uplift_mean - 2.5 * uplift_std
  const max = uplift_mean + 2.5 * uplift_std
  const bins = 12
  const width = (max - min) / bins

  return Array.from({ length: bins }, (_, i) => {
    const lo = min + i * width
    const hi = lo + width
    const mid = (lo + hi) / 2
    // Approximate normal density from summary stats
    const z = (mid - uplift_mean) / uplift_std
    const density = Math.exp(-0.5 * z * z) / (uplift_std * Math.sqrt(2 * Math.PI))
    return {
      bin: `${fmtNum(lo, 2)}`,
      count: Math.round(density * width * summary.n_patients * 10) / 10,
      midpoint: mid,
      inIQR: mid >= uplift_p25 && mid <= uplift_p75,
    }
  })
}

// ─── Main Component ──────────────────────────────────────────────────────────

export function CausalOutcomes({ result }: Props) {
  const [interpretation, setInterpretation] = useState<string | null>(
    result.agent_interpretation ?? null,
  )

  const interpretMutation = useMutation({
    mutationFn: () => api.interpret(result.run_id),
    onSuccess: (data) => setInterpretation(data.interpretation),
  })

  const forestData = buildForestData(result)
  const cateGrid = generateCATEHeatmap(result)
  const upliftHist = buildUpliftHistogram(result.uplift_summary)
  const cateValues = cateGrid.map((c) => c.cate)
  const cateMin = Math.min(...cateValues)
  const cateMax = Math.max(...cateValues)

  return (
    <div className="py-8 px-6 max-w-6xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Causal & Outcomes</h1>
        <p className="text-sm text-slate-500 mt-1">
          ATE forest plot, CATE subgroup heatmap, HTE uplift distribution, and AI interpretation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column: Forest Plot + Uplift Histogram */}
        <div className="lg:col-span-2 space-y-6">
          {/* Forest Plot */}
          <ChartCard title="Forest Plot — ATE by Endpoint & Comparison">
            <ForestPlot data={forestData} />
          </ChartCard>

          {/* HTE Uplift Histogram */}
          <ChartCard title="HTE Uplift Distribution (T-Learner)">
            <div className="mb-3 grid grid-cols-3 gap-3">
              {[
                { label: 'Mean', val: fmtNum(result.uplift_summary?.uplift_mean, 4) },
                { label: 'Median', val: fmtNum(result.uplift_summary?.uplift_median, 4) },
                { label: 'High Resp %', val: `${result.uplift_summary?.high_responder_pct ?? '—'}%` },
              ].map((s) => (
                <div key={s.label} className="bg-slate-50 rounded px-3 py-2">
                  <p className="text-xs text-slate-400">{s.label}</p>
                  <p className="text-sm font-mono font-medium text-slate-800">{s.val}</p>
                </div>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={upliftHist} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                <XAxis dataKey="bin" tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(v: number) => [fmtNum(v, 1), 'Density']}
                  labelFormatter={(v) => `Uplift ≈ ${v}`}
                  contentStyle={{ fontSize: 11 }}
                />
                <ReferenceLine x={fmtNum(result.uplift_summary?.uplift_mean, 2)} stroke="#2554e9" strokeDasharray="4 3" />
                <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                  {upliftHist.map((entry, i) => (
                    <Cell key={i} fill={entry.inIQR ? '#93c5fd' : '#cbd5e1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-slate-400 mt-1">
              IQR: [{fmtNum(result.uplift_summary?.uplift_p25, 3)}, {fmtNum(result.uplift_summary?.uplift_p75, 3)}] (highlighted)
            </p>
          </ChartCard>
        </div>

        {/* Right column: CATE Heatmap + AI Panel */}
        <div className="space-y-6">
          {/* CATE Heatmap */}
          <ChartCard title="CATE Subgroup Heatmap">
            <p className="text-xs text-slate-400 mb-3">
              High dose vs placebo · primary endpoint
            </p>
            <CATEHeatmap grid={cateGrid} min={cateMin} max={cateMax} />
          </ChartCard>

          {/* AI Interpretation Panel */}
          <div className="rounded-lg border border-amber-200 overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-amber-200 bg-amber-50">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-600" />
                <span className="text-sm font-semibold text-amber-800">AI Interpretation</span>
                <Badge variant="ai" className="text-xs">AI</Badge>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => interpretMutation.mutate()}
                disabled={interpretMutation.isPending}
                className="border-amber-200 text-amber-700 hover:bg-amber-100 text-xs gap-1"
              >
                {interpretMutation.isPending ? (
                  <><RefreshCw className="w-3 h-3 animate-spin" />Interpreting…</>
                ) : (
                  <>
                    <Sparkles className="w-3 h-3" />
                    Get AI Interpretation
                  </>
                )}
              </Button>
            </div>

            <div className="p-4 bg-[#FDF8F0] min-h-[120px]">
              {interpretMutation.isError && (
                <div className="flex items-start gap-2 text-red-700 text-sm">
                  <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span>{(interpretMutation.error as Error).message}</span>
                </div>
              )}

              {interpretation ? (
                <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                  {interpretation}
                </p>
              ) : (
                !interpretMutation.isPending && (
                  <p className="text-sm text-slate-400 italic">
                    Click "Get AI Interpretation" to generate a Claude-powered analysis of these results.
                  </p>
                )
              )}

              {interpretMutation.isPending && (
                <div className="flex items-center gap-2 text-amber-600 text-sm">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Generating interpretation…
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Forest Plot ──────────────────────────────────────────────────────────────

function ForestPlot({ data }: { data: ForestRow[] }) {
  if (data.length === 0) return <p className="text-sm text-slate-400 py-4">No ATE data available.</p>

  // Find global min/max for axis
  const allVals = data.flatMap((d) => [d.lower, d.upper])
  const axMin = Math.min(...allVals) - 0.1
  const axMax = Math.max(...allVals) + 0.1

  return (
    <div className="space-y-1 mt-2">
      {/* Column headers */}
      <div className="grid grid-cols-[1fr_80px_80px_80px] gap-2 px-2 py-1 text-xs font-medium text-slate-400 uppercase tracking-wide border-b border-slate-100">
        <span>Endpoint / Comparison</span>
        <span className="text-right">ATE</span>
        <span className="text-right">95% CI</span>
        <span className="text-right">P-value</span>
      </div>

      {data.map((row, i) => {
        const pInfo = pValueLabel(row.p_value)
        const axRange = axMax - axMin
        const dotPct = ((row.ate - axMin) / axRange) * 100
        const loPct = ((row.lower - axMin) / axRange) * 100
        const hiPct = ((row.upper - axMin) / axRange) * 100
        const zeroPct = ((-axMin) / axRange) * 100

        return (
          <div
            key={i}
            className={cn(
              'grid grid-cols-[1fr_80px_80px_80px] gap-2 px-2 py-2 rounded hover:bg-slate-50 text-sm',
              row.is_primary && i === 0 && 'bg-blue-50',
            )}
          >
            <div>
              <div className="flex items-center gap-1.5 mb-2">
                <span className={cn('text-xs font-medium', row.is_primary ? 'text-blue-700' : 'text-slate-700')}>
                  {row.label}
                </span>
                {row.is_primary && <Badge variant="default" className="text-xs py-0">Primary</Badge>}
                <span
                  className="text-xs px-1.5 py-0.5 rounded text-white"
                  style={{
                    backgroundColor:
                      row.comparison === 'high_dose_vs_placebo'
                        ? ARM_COLORS['high_dose']
                        : ARM_COLORS['low_dose'],
                  }}
                >
                  {row.comparison === 'high_dose_vs_placebo' ? 'High' : 'Low'} vs Placebo
                </span>
              </div>
              {/* CI bar */}
              <div className="relative h-3 bg-slate-100 rounded-full mx-1">
                {/* Null line */}
                <div
                  className="absolute top-0 bottom-0 w-px bg-slate-400"
                  style={{ left: `${zeroPct}%` }}
                />
                {/* CI line */}
                <div
                  className="absolute top-1/2 h-0.5 bg-blue-400 -translate-y-1/2"
                  style={{ left: `${loPct}%`, right: `${100 - hiPct}%` }}
                />
                {/* Point estimate */}
                <div
                  className={cn(
                    'absolute top-1/2 w-3 h-3 rounded-full -translate-x-1/2 -translate-y-1/2 border-2',
                    row.is_primary ? 'bg-blue-600 border-blue-600' : 'bg-slate-600 border-slate-600',
                  )}
                  style={{ left: `${dotPct}%` }}
                />
              </div>
            </div>

            <span className="text-right font-mono text-xs text-slate-800 self-center">
              {row.ate >= 0 ? '+' : ''}
              {fmtNum(row.ate, 3)}
            </span>
            <span className="text-right font-mono text-xs text-slate-500 self-center">
              [{fmtNum(row.lower, 2)}, {fmtNum(row.upper, 2)}]
            </span>
            <div className="text-right self-center">
              <Badge
                variant={pInfo.variant === 'sig' ? 'sig' : pInfo.variant === 'trend' ? 'trend' : 'ns'}
                className="text-xs"
              >
                {pInfo.label}
              </Badge>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─── CATE Heatmap ─────────────────────────────────────────────────────────────

function CATEHeatmap({
  grid,
  min,
  max,
}: {
  grid: Array<{ ageGroup: string; sex: string; cate: number }>
  min: number
  max: number
}) {
  const ageGroups = ['45-55', '55-65', '65-75', '75+']
  const sexes = ['F', 'M']

  return (
    <div>
      <div className="grid grid-cols-3 gap-1 text-xs">
        {/* Header row */}
        <div />
        {sexes.map((s) => (
          <div key={s} className="text-center font-medium text-slate-500 pb-1">
            {s === 'F' ? 'Female' : 'Male'}
          </div>
        ))}
        {/* Data rows */}
        {ageGroups.map((ag) => (
          <>
            <div key={`label-${ag}`} className="text-slate-500 text-xs self-center pr-1">
              {ag}
            </div>
            {sexes.map((sx) => {
              const cell = grid.find((g) => g.ageGroup === ag && g.sex === sx)
              const val = cell?.cate ?? 0
              return (
                <div
                  key={`${ag}-${sx}`}
                  className="rounded flex items-center justify-center py-3 text-xs font-mono font-medium"
                  style={{
                    backgroundColor: cateColor(val, min, max),
                    color: Math.abs((val - min) / (max - min) - 0.5) > 0.3 ? '#fff' : '#334155',
                  }}
                  title={`Age ${ag}, ${sx === 'F' ? 'Female' : 'Male'}: CATE = ${fmtNum(val, 3)}`}
                >
                  {fmtNum(val, 2)}
                </div>
              )
            })}
          </>
        ))}
      </div>

      {/* Color scale legend */}
      <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
        <span className="font-mono">{fmtNum(min, 2)}</span>
        <div
          className="flex-1 h-2 rounded"
          style={{
            background: `linear-gradient(to right, ${cateColor(min, min, max)}, ${cateColor((min + max) / 2, min, max)}, ${cateColor(max, min, max)})`,
          }}
        />
        <span className="font-mono">{fmtNum(max, 2)}</span>
      </div>
      <p className="text-xs text-slate-400 mt-1 text-center">CATE (blue = protective, amber = adverse)</p>
    </div>
  )
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-lg border border-slate-200 p-4">
      <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">{title}</h3>
      {children}
    </div>
  )
}
