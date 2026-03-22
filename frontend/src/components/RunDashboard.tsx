import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn, fmtNum, fmtPct, pValueLabel, copyToClipboard } from '@/lib/utils'
import type { SimulationResponse } from '@/types'
import { ARM_COLORS, ARM_LABELS } from '@/types'
import { Copy, Check, ExternalLink } from 'lucide-react'

interface Props {
  result: SimulationResponse
  onOpenBiomarkers: () => void
}

export function RunDashboard({ result, onOpenBiomarkers }: Props) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    copyToClipboard(result.run_id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Primary endpoint ATE (high dose vs placebo)
  const primaryEndpoint = Object.entries(result.ate_results).find(([, v]) => v.is_primary)
  const primaryATE = primaryEndpoint?.[1].high_dose_vs_placebo
  const pLabel = primaryATE ? pValueLabel(primaryATE.p_value) : null

  const cohort = result.cohort_summary
  const logistic = result.logistic_summary
  const responderRate = logistic.arm_summary?.['high_dose']?.responder_rate ?? logistic.overall_responder_rate

  return (
    <div className="py-8 px-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">{result.trial_name}</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            {result.n_patients} patients · {result.n_weeks} weeks · seed {result.seed} ·{' '}
            {fmtNum(result.elapsed_seconds, 2)}s runtime
          </p>
        </div>
      </div>

      {/* Run ID */}
      <div className="flex items-center gap-2 bg-slate-900 rounded-md px-4 py-2.5 w-fit">
        <span className="text-xs text-slate-400 mr-1">run_id</span>
        <span className="font-mono text-sm text-slate-100">{result.run_id}</span>
        <Button
          variant="ghost"
          size="icon"
          onClick={handleCopy}
          className="h-6 w-6 text-slate-400 hover:text-slate-100"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
        </Button>
      </div>

      {/* Scorecard Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <ScoreCard
          label="ATE (high dose)"
          value={
            primaryATE ? (
              <ATEBar
                ate={primaryATE.ate}
                lower={primaryATE.ci_lower}
                upper={primaryATE.ci_upper}
              />
            ) : (
              '—'
            )
          }
          sub={primaryEndpoint?.[0] ?? ''}
        />
        <ScoreCard
          label="P-value"
          value={
            pLabel ? (
              <Badge
                variant={pLabel.variant === 'sig' ? 'sig' : pLabel.variant === 'trend' ? 'trend' : 'ns'}
              >
                {pLabel.label}
              </Badge>
            ) : (
              '—'
            )
          }
          sub="primary endpoint"
        />
        <ScoreCard
          label="Responder Rate"
          value={fmtPct(responderRate * 100)}
          sub="high dose arm"
        />
        <ScoreCard
          label="Dropout"
          value={fmtPct(cohort.dropout_pct)}
          sub={`${cohort.dropout_n} of ${cohort.n_enrolled}`}
        />
      </div>

      {/* Tabs */}
      <Tabs defaultValue="cohort" className="w-full">
        <TabsList>
          <TabsTrigger value="cohort">Cohort</TabsTrigger>
          <TabsTrigger value="biomarkers">Biomarkers</TabsTrigger>
          <TabsTrigger value="outcomes">Outcomes</TabsTrigger>
        </TabsList>

        {/* Cohort Tab */}
        <TabsContent value="cohort">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            {/* Arm Distribution */}
            <ChartCard title="Arm Distribution">
              <ResponsiveContainer width="100%" height={180}>
                <BarChart
                  data={Object.entries(cohort.arm_distribution).map(([arm, n]) => ({
                    arm: ARM_LABELS[arm] ?? arm,
                    n,
                    fill: ARM_COLORS[arm] ?? '#94a3b8',
                  }))}
                  margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
                >
                  <XAxis dataKey="arm" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v: number) => [v, 'Patients']} />
                  <Bar dataKey="n" radius={[3, 3, 0, 0]}>
                    {Object.entries(cohort.arm_distribution).map(([arm]) => (
                      <Cell key={arm} fill={ARM_COLORS[arm] ?? '#94a3b8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Sex Distribution */}
            <ChartCard title="Sex Distribution">
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Female', value: cohort.sex_f_pct, fill: '#93c5fd' },
                      { name: 'Male', value: 100 - cohort.sex_f_pct, fill: '#64748b' },
                    ]}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={72}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    <Cell fill="#93c5fd" />
                    <Cell fill="#64748b" />
                  </Pie>
                  <Tooltip formatter={(v: number) => [`${fmtNum(v, 1)}%`]} />
                  <Legend iconType="circle" iconSize={8} />
                </PieChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Age & BMI stats */}
            <ChartCard title="Demographics">
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm mt-2">
                <DemStat label="Mean Age" value={`${fmtNum(cohort.age_mean, 1)} yr`} />
                <DemStat label="Age SD" value={`±${fmtNum(cohort.age_std, 1)} yr`} />
                <DemStat label="Mean BMI" value={fmtNum(cohort.bmi_mean, 1)} />
                <DemStat label="BMI SD" value={`±${fmtNum(cohort.bmi_std, 1)}`} />
                <DemStat label="Female" value={fmtPct(cohort.sex_f_pct)} />
                <DemStat label="Enrolled" value={`${cohort.n_enrolled}`} />
              </dl>
            </ChartCard>

            {/* Top Comorbidities */}
            <ChartCard title="Top Comorbidities">
              <div className="space-y-2 mt-2">
                {Object.entries(cohort.top_comorbidities)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 5)
                  .map(([cond, rate]) => (
                    <div key={cond} className="flex items-center gap-2">
                      <div className="flex-1">
                        <div className="flex justify-between text-xs mb-0.5">
                          <span className="text-slate-600 capitalize">{cond.replace(/_/g, ' ')}</span>
                          <span className="font-mono text-slate-500">{fmtPct(rate * 100)}</span>
                        </div>
                        <div className="h-1.5 bg-slate-100 rounded-full">
                          <div
                            className="h-full bg-blue-400 rounded-full"
                            style={{ width: `${rate * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </ChartCard>
          </div>
        </TabsContent>

        {/* Biomarkers Tab */}
        <TabsContent value="biomarkers">
          <div className="mt-4 bg-white rounded-lg border border-slate-200 p-6 flex flex-col items-center gap-4">
            <p className="text-sm text-slate-600">
              View longitudinal biomarker trajectories with ±1 SD bands for all treatment arms.
            </p>
            <Button onClick={onOpenBiomarkers} variant="outline" className="gap-2">
              <ExternalLink className="w-4 h-4" />
              Open Biomarker Explorer
            </Button>
          </div>
        </TabsContent>

        {/* Outcomes Tab */}
        <TabsContent value="outcomes">
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Survival Summary */}
            <ChartCard title="Survival Summary (Weibull AFT)">
              <div className="mt-2 space-y-2">
                {result.survival_summary.arms?.map((arm) => (
                  <div key={arm.arm} className="grid grid-cols-4 gap-2 text-xs border-b border-slate-100 pb-2">
                    <div className="font-medium text-slate-700">{ARM_LABELS[arm.arm] ?? arm.arm}</div>
                    <div>
                      <span className="text-slate-400">Med.</span>{' '}
                      <span className="font-mono">{fmtNum(arm.median_survival_weeks, 1)}w</span>
                    </div>
                    <div>
                      <span className="text-slate-400">52w</span>{' '}
                      <span className="font-mono">{fmtPct(arm.survival_at_52w * 100)}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">HR</span>{' '}
                      <span className="font-mono">
                        {arm.hazard_ratio_vs_placebo != null
                          ? fmtNum(arm.hazard_ratio_vs_placebo, 2)
                          : 'ref'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </ChartCard>

            {/* Responder Rates */}
            <ChartCard title="Responder Rates (Logistic Model)">
              <ResponsiveContainer width="100%" height={160}>
                <BarChart
                  data={Object.entries(logistic.arm_summary ?? {}).map(([arm, s]) => ({
                    arm: ARM_LABELS[arm] ?? arm,
                    rate: Number((s.responder_rate * 100).toFixed(1)),
                    fill: ARM_COLORS[arm] ?? '#94a3b8',
                  }))}
                  margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
                >
                  <XAxis dataKey="arm" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Responder Rate']} />
                  <Bar dataKey="rate" radius={[3, 3, 0, 0]}>
                    {Object.keys(logistic.arm_summary ?? {}).map((arm) => (
                      <Cell key={arm} fill={ARM_COLORS[arm] ?? '#94a3b8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ScoreCard({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="bg-white rounded-lg border border-slate-200 px-4 py-3">
      <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</p>
      <div className="mt-1.5 text-lg font-semibold text-slate-900">{value}</div>
      {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
    </div>
  )
}

function ATEBar({ ate, lower, upper }: { ate: number; lower: number; upper: number }) {
  const isNeg = ate < 0
  return (
    <div className="flex items-center gap-2">
      <span className={cn('text-lg font-mono font-semibold', isNeg ? 'text-blue-700' : 'text-amber-600')}>
        {ate >= 0 ? '+' : ''}
        {fmtNum(ate, 3)}
      </span>
      <span className="text-xs text-slate-400 font-mono">
        [{fmtNum(lower, 2)}, {fmtNum(upper, 2)}]
      </span>
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

function DemStat({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-slate-500">{label}</dt>
      <dd className="font-mono font-medium text-slate-800">{value}</dd>
    </>
  )
}
