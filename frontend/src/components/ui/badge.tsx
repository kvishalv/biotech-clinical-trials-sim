import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-blue-600 text-white',
        secondary: 'border-transparent bg-slate-100 text-slate-700',
        sig: 'border-transparent bg-emerald-100 text-emerald-700',
        trend: 'border-transparent bg-amber-100 text-amber-700',
        ns: 'border-transparent bg-slate-100 text-slate-500',
        destructive: 'border-transparent bg-red-100 text-red-700',
        outline: 'text-slate-700 border-slate-200',
        ai: 'border border-amber-200 bg-amber-50 text-amber-700',
      },
    },
    defaultVariants: { variant: 'default' },
  },
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
