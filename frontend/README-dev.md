# Dashboard Dev Notes

## Start

```bash
cd frontend
npm install
npm run dev    # → http://localhost:5173
```

The Vite dev proxy forwards `/simulate`, `/biomarkers`, `/agents`, `/health` to `http://localhost:8000`.

## Build

```bash
npm run build   # outputs to dist/
```

## Structure

```
src/
├── App.tsx                    # Root layout + view router
├── types/index.ts             # All API + UI types
├── lib/
│   ├── api.ts                 # API client + localStorage registry
│   └── utils.ts               # Formatting helpers
└── components/
    ├── Sidebar.tsx
    ├── TrialConfigurator.tsx
    ├── RunDashboard.tsx
    ├── BiomarkerExplorer.tsx
    ├── CausalOutcomes.tsx
    ├── ExperimentRegistry.tsx
    └── ui/                    # shadcn-style primitives
```
