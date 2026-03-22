import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
  Legend,
} from 'recharts'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/badge'
import { api } from '@/lib/api'
import { cn, labelBiomarker, fmtNum } from '@/lib/utils'
import type { BiomarkerPoint } from '@/types'
import { BIOMARKER_NAMES, ARM_COLORS, ARM_LABELS } from '@/types'
import { RefreshCw, TrendingDown } from 'lucide-react'

interface Props {
  runId: string
  nWeeks: number
  burninWeeks?: number
}

const DEFAULT_SELECTED = new Set([
  'inflammation_index',
  'epigenetic_age_acceleration',
  'metabolic_risk_index',
])

const ARM_FILL_COLORS: Record<string, string> = {
  placebo: '#cbd5e1',
  low_dose: '#bfdbfe',
  high_dose: '#93c5fd',
}

export function BiomarkerExplorer({ runId, nWeeks, burninWeeks = 4 }: Props) {
  const [selected, setSelected] = useState<Set<string>>(DEFAULT_SELECTED)
  const [activeArms, setActiveArms] = useState<Set<string>>(
    new Set(['placebo', 'low_dose', 'high_dose']),
  )

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['biomarkers', runId],
    queryFn: () => api.getBiomarkers(runId),
    enabled: !!runId,
  })

  const toggleBiomarker = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(name)) {
        next.delete(name)
      } else {
        next.add(name)
      }
      return next
    })
  }

  const toggleArm = (arm: string) => {
    setActiveArms((prev) => {
      const next = new Set(prev)
      if (next.has(arm)) {
        if (next.size === 1) return prev // keep at least one
        next.delete(arm)
      } else {
        next.add(arm)
      }
      return next
    })
  }

  // Build chart data per selected biomarker
  const chartDataMap = useMemo(() => {
    if (!data) return {}
    const result: Record<string, Record<number, Record<string, { mean: number; upper: number; lower: number }>>> = {}

    for (const bm of selected) {
      const points = data.biomarkers[bm] ?? []
      const byWeek: Record<number, Record<string, { mean: number; upper: number; lower: number }>> = {}

      for (const pt of points) {
        if (!activeArms.has(pt.arm)) continue
        if (!byWeek[pt.week]) byWeek[pt.week] = {}
        byWeek[pt.week][pt.arm] = {
          mean: pt.mean,
          upper: pt.mean + pt.std,
          lower: pt.mean - pt.std,
        }
      }
      result[bm] = byWeek
    }
    return result
  }, [data, selected, activeArms])

  // Convert to recharts format per biomarker
  const chartSeries = useMemo(() => {
    const out: Record<string, Array<Record<string, number | null>>> = {}
    for (const [bm, byWeek] of Object.entries(chartDataMap)) {
      const weeks = Array.from({ length: nWeeks + 1 }, (_, i) => i)
      out[bm] = weeks.map((w) => {
        const row: Record<string, number | null> = { week: w }
        for (const arm of activeArms) {
          const pt = byWeek[w]?.[arm]
          row[`${arm}_mean`] = pt?.mean ?? null
          row[`${arm}_upper`] = pt?.upper ?? null
          row[`${arm}_lower`] = pt?.lower ?? null
          row[`${arm}_band`] = pt != null ? pt.upper - pt.lower : null
        }
        return row
      })
    }
    return out
  }, [chartDataMap, nWeeks, activeArms])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 gap-2 text-slate-400">
        <RefreshCw className="w-5 h-5 animate-spin" />
        <span>Loading biomarker data…</span>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center h-64 text-red-600 text-sm">
        Error: {(error as Error).message}
      </div>
    )
  }

  return (
    <div className="py-8 px-6 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-semibold text-slate-900">Biomarker Explorer</h1>
        <p className="text-sm text-slate-500 mt-1">
          Longitudinal trajectories with ±1 SD bands. Dashed line marks burn-in end (week {burninWeeks}).
        </p>
      </div>

      <div className="flex gap-6">
        {/* Selector Panel */}
        <div className="w-52 shrink-0 space-y-4">
          {/* Biomarker Checkboxes */}
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Biomarkers</p>
            <div className="space-y-2.5">
              {BIOMARKER_NAMES.map((bm) => (
                <div key={bm} className="flex items-center gap-2.5">
                  <Checkbox
                    id={bm}
                    checked={selected.has(bm)}
                    onCheckedChange={() => toggleBiomarker(bm)}
                  />
                  <label htmlFor={bm} className="text-xs text-slate-600 cursor-pointer leading-tight">
                    {labelBiomarker(bm)}
                  </label>
                </div>
              ))}
            </div>
          </div>

          {/* Arm Toggles */}
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">Arms</p>
            <div className="space-y-2">
              {(['placebo', 'low_dose', 'high_dose'] as const).map((arm) => (
                <button
                  key={arm}
                  onClick={() => toggleArm(arm)}
                  className={cn(
                    'flex items-center gap-2 w-full text-xs rounded px-2 py-1 transition-colors',
                    activeArms.has(arm)
                      ? 'text-slate-800'
                      : 'text-slate-300',
                  )}
                >
                  <span
                    className="w-3 h-3 rounded-full shrink-0"
                    style={{
                      backgroundColor: activeArms.has(arm) ? ARM_COLORS[arm] : '#e2e8f0',
                    }}
                  />
                  {ARM_LABELS[arm]}
                </button>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Legend</p>
            <div className="space-y-1.5 text-xs text-slate-500">
              <div className="flex items-center gap-2">
                <div className="w-6 h-0.5 bg-slate-700" />
                Mean trajectory
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-3 bg-slate-200 opacity-60" />
                ±1 SD band
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 border-t-2 border-dashed border-amber-400" />
                Burn-in end
              </div>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="flex-1 space-y-6 min-w-0">
          {selected.size === 0 ? (
            <div className="bg-white rounded-lg border border-slate-200 flex items-center justify-center h-48 text-slate-400 text-sm">
              Select biomarkers to display
            </div>
          ) : (
            Array.from(selected).map((bm) => {
              const series = chartSeries[bm] ?? []
              if (series.length === 0) return null

              return (
                <div key={bm} className="bg-white rounded-lg border border-slate-200 p-4">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingDown className="w-4 h-4 text-blue-500" />
                    <h3 className="text-sm font-semibold text-slate-800">{labelBiomarker(bm)}</h3>
                    <Badge variant="outline" className="text-xs">{bm}</Badge>
                  </div>

                  <ResponsiveContainer width="100%" height={220}>
                    <ComposedChart data={series} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis
                        dataKey="week"
                        tick={{ fontSize: 10 }}
                        label={{ value: 'Week', position: 'insideBottomRight', offset: -5, fontSize: 10 }}
                      />
                      <YAxis tick={{ fontSize: 10 }} width={45} />
                      <Tooltip
                        formatter={(val: number, name: string) => [fmtNum(val, 3), name]}
                        labelFormatter={(w) => `Week ${w}`}
                        contentStyle={{ fontSize: 11 }}
                      />

                      {/* Burn-in reference line */}
                      <ReferenceLine
                        x={burninWeeks}
                        stroke="#f59e0b"
                        strokeDasharray="5 4"
                        strokeWidth={1.5}
                        label={{ value: 'Burn-in', position: 'top', fontSize: 10, fill: '#f59e0b' }}
                      />

                      {/* SD bands + mean lines per arm */}
                      {Array.from(activeArms).map((arm) => (
                        <g key={arm}>
                          <Area
                            type="monotone"
                            dataKey={`${arm}_upper`}
                            stroke="none"
                            fill={ARM_FILL_COLORS[arm] ?? '#e2e8f0'}
                            fillOpacity={0.4}
                            legendType="none"
                            tooltipType="none"
                            isAnimationActive={false}
                            connectNulls
                          />
                          <Area
                            type="monotone"
                            dataKey={`${arm}_lower`}
                            stroke="none"
                            fill={ARM_FILL_COLORS[arm] ?? '#e2e8f0'}
                            fillOpacity={0.4}
                            legendType="none"
                            tooltipType="none"
                            isAnimationActive={false}
                            connectNulls
                          />
                          <Line
                            type="monotone"
                            dataKey={`${arm}_mean`}
                            stroke={ARM_COLORS[arm]}
                            strokeWidth={2}
                            dot={false}
                            name={ARM_LABELS[arm] ?? arm}
                            connectNulls
                          />
                        </g>
                      ))}

                      <Legend
                        formatter={(value) => <span style={{ fontSize: 11 }}>{value}</span>}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
