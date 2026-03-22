// ─── API Request Types ───────────────────────────────────────────────────────

export interface RunSimulationRequest {
  seed: number
  n_patients: number
  n_weeks: number
  trial_config_path?: string
  biomarker_config_path?: string
  run_agents?: boolean
}

export interface DriftRequest {
  reference_run_id: string
  new_run_id: string
  check_biomarkers?: boolean
}

// ─── ATE Result ──────────────────────────────────────────────────────────────

export interface ATEEstimate {
  ate: number
  se: number
  ci_lower: number
  ci_upper: number
  p_value: number
  n_treated: number
  n_control: number
  week: number
}

export interface EndpointATEResult {
  high_dose_vs_placebo: ATEEstimate
  low_dose_vs_placebo: ATEEstimate
  is_primary: boolean
}

export type ATEResults = Record<string, EndpointATEResult>

// ─── Cohort Summary ──────────────────────────────────────────────────────────

export interface CohortSummary {
  n_enrolled: number
  age_mean: number
  age_std: number
  bmi_mean: number
  bmi_std: number
  sex_f_pct: number
  arm_distribution: Record<string, number>
  site_distribution: Record<string, number>
  dropout_n: number
  dropout_pct: number
  top_comorbidities: Record<string, number>
}

// ─── Survival Summary ────────────────────────────────────────────────────────

export interface SurvivalArmResult {
  arm: string
  median_survival_weeks: number
  survival_at_52w: number
  hazard_ratio_vs_placebo: number | null
  log_rank_p: number
  n_events: number
  n_censored: number
}

export interface SurvivalSummary {
  model: string
  n_patients: number
  n_events_total: number
  arms: SurvivalArmResult[]
}

// ─── Logistic Summary ────────────────────────────────────────────────────────

export interface ArmLogisticResult {
  responder_rate: number
  pred_proba_mean: number
  n: number
}

export interface LogisticSummary {
  model: string
  n_patients: number
  overall_responder_rate: number
  arm_summary: Record<string, ArmLogisticResult>
}

// ─── Continuous Summary ───────────────────────────────────────────────────────

export interface ContinuousSummary {
  model: string
  r_squared: number
  n_obs: number
  coefficients: Record<string, number>
  p_values: Record<string, number>
  confidence_intervals_95: Record<string, [number, number]>
  aic: number
}

// ─── Uplift Summary ──────────────────────────────────────────────────────────

export interface UpliftSummary {
  model: string
  n_patients: number
  uplift_mean: number
  uplift_std: number
  uplift_p25: number
  uplift_median: number
  uplift_p75: number
  high_responder_pct: number
}

// ─── Simulation Response ─────────────────────────────────────────────────────

export interface SimulationResponse {
  run_id: string
  trial_name: string
  seed: number
  n_patients: number
  n_weeks: number
  elapsed_seconds: number
  cohort_summary: CohortSummary
  ate_results: ATEResults
  survival_summary: SurvivalSummary
  logistic_summary: LogisticSummary
  continuous_summary: ContinuousSummary
  uplift_summary: UpliftSummary
  agent_narrative?: string | null
  agent_interpretation?: string | null
}

// ─── Biomarker Data ──────────────────────────────────────────────────────────

export interface BiomarkerPoint {
  week: number
  arm: string
  mean: number
  std: number
  n: number
}

export interface BiomarkersResponse {
  run_id: string
  biomarkers: Record<string, BiomarkerPoint[]>
}

// ─── Lint Response ───────────────────────────────────────────────────────────

export interface LintResponse {
  rule_errors: string[]
  llm_report: string
  has_errors: boolean
}

// ─── Drift Response ──────────────────────────────────────────────────────────

export interface DriftVariableResult {
  variable: string
  test: string
  statistic: number
  p_value: number
  drifted: boolean
  [key: string]: unknown
}

export interface DriftResponse {
  reference_run_id: string
  new_run_id: string
  n_drifted: number
  n_tested: number
  overall_drift: boolean
  results: DriftVariableResult[]
}

// ─── Interpret Response ──────────────────────────────────────────────────────

export interface InterpretResponse {
  run_id: string
  interpretation: string
}

// ─── UI State ────────────────────────────────────────────────────────────────

export type View =
  | 'configurator'
  | 'run-dashboard'
  | 'biomarker-explorer'
  | 'causal-outcomes'
  | 'registry'

export const BIOMARKER_NAMES = [
  'inflammation_index',
  'metabolic_risk_index',
  'epigenetic_age_acceleration',
  'frailty_progression',
  'organ_reserve_score',
  'latent_mitochondrial_dysfunction',
  'immune_resilience',
  'sleep_circadian_disruption',
  'recovery_velocity',
] as const

export type BiomarkerName = (typeof BIOMARKER_NAMES)[number]

export const BIOMARKER_LABELS: Record<BiomarkerName, string> = {
  inflammation_index: 'Inflammation Index',
  metabolic_risk_index: 'Metabolic Risk Index',
  epigenetic_age_acceleration: 'Epigenetic Age Acceleration',
  frailty_progression: 'Frailty Progression',
  organ_reserve_score: 'Organ Reserve Score',
  latent_mitochondrial_dysfunction: 'Mitochondrial Dysfunction',
  immune_resilience: 'Immune Resilience',
  sleep_circadian_disruption: 'Sleep/Circadian Disruption',
  recovery_velocity: 'Recovery Velocity',
}

export const ARM_COLORS: Record<string, string> = {
  placebo: '#94a3b8',
  low_dose: '#60a5fa',
  high_dose: '#2554e9',
}

export const ARM_LABELS: Record<string, string> = {
  placebo: 'Placebo',
  low_dose: 'Low Dose',
  high_dose: 'High Dose',
}
