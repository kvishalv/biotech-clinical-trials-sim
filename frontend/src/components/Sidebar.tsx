import { cn } from '@/lib/utils'
import type { View } from '@/types'
import {
  FlaskConical,
  LayoutDashboard,
  LineChart,
  GitFork,
  Database,
} from 'lucide-react'

interface NavItem {
  id: View
  label: string
  icon: React.ElementType
  subtitle: string
}

const NAV_ITEMS: NavItem[] = [
  {
    id: 'configurator',
    label: 'Trial Configurator',
    icon: FlaskConical,
    subtitle: 'Design & run trials',
  },
  {
    id: 'run-dashboard',
    label: 'Run Dashboard',
    icon: LayoutDashboard,
    subtitle: 'ATE, cohort, outcomes',
  },
  {
    id: 'biomarker-explorer',
    label: 'Biomarker Explorer',
    icon: LineChart,
    subtitle: 'Longitudinal trajectories',
  },
  {
    id: 'causal-outcomes',
    label: 'Causal & Outcomes',
    icon: GitFork,
    subtitle: 'Forest plot, HTE, CATE',
  },
  {
    id: 'registry',
    label: 'Experiment Registry',
    icon: Database,
    subtitle: 'All runs & comparison',
  },
]

interface SidebarProps {
  active: View
  onChange: (v: View) => void
  hasRun: boolean
}

export function Sidebar({ active, onChange, hasRun }: SidebarProps) {
  return (
    <aside className="w-56 shrink-0 bg-white border-r border-slate-200 flex flex-col h-screen sticky top-0">
      {/* Logo / Brand */}
      <div className="px-4 py-5 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-blue-600 flex items-center justify-center">
            <FlaskConical className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-900 leading-tight">Clinical Trial</p>
            <p className="text-xs text-slate-400 leading-tight">Simulator</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-3 px-2">
        <div className="space-y-0.5">
          {NAV_ITEMS.map((item) => {
            const locked = item.id !== 'configurator' && item.id !== 'registry' && !hasRun
            const Icon = item.icon
            return (
              <button
                key={item.id}
                onClick={() => !locked && onChange(item.id)}
                disabled={locked}
                className={cn(
                  'w-full flex items-start gap-3 rounded-md px-3 py-2.5 text-left transition-colors',
                  active === item.id
                    ? 'bg-blue-50 text-blue-700'
                    : locked
                      ? 'text-slate-300 cursor-not-allowed'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900',
                )}
              >
                <Icon
                  className={cn(
                    'w-4 h-4 mt-0.5 shrink-0',
                    active === item.id ? 'text-blue-600' : locked ? 'text-slate-200' : 'text-slate-400',
                  )}
                />
                <div className="min-w-0">
                  <p className="text-sm font-medium leading-tight truncate">{item.label}</p>
                  <p
                    className={cn(
                      'text-xs leading-tight mt-0.5',
                      active === item.id ? 'text-blue-500' : 'text-slate-400',
                    )}
                  >
                    {item.subtitle}
                  </p>
                </div>
              </button>
            )
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-slate-100">
        <p className="text-xs text-slate-400">FastAPI · localhost:8000</p>
      </div>
    </aside>
  )
}
