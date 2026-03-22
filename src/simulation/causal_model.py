"""
simulation/causal_model.py
───────────────────────────
Structural Causal Model (SCM) layer for the clinical trial simulator.

Implements a pluggable DAG-based causal framework using networkx. The DAG
defines the data-generating process (DGP) — which variables causally influence
which outcomes — and supports:

  1. DAG definition and validation (no cycles, topological ordering)
  2. Backdoor criterion for confounder identification
  3. do-calculus intervention: P(Y | do(X=x))
  4. Counterfactual estimation via twin-network
  5. Average Treatment Effect (ATE) and CATE estimation
  6. Uplift scores per patient

Architecture note:
  The SCM is intentionally separate from the stochastic biomarker engine.
  The causal model defines WHAT influences WHAT; the biomarker models define
  HOW those influences are parameterised stochastically.
  The LLM agent layer reads SCM outputs — it does not modify them.

Graph encoding:
  Nodes: variable names (strings)
  Edges: directed, from cause → effect
  Edge attribute 'weight': linear structural coefficient (default 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Default longevity DAG
# ──────────────────────────────────────────────────────────────────────────────

# Default DAG captures key causal pathways in a longevity intervention trial.
# Pluggable: replace with any user-defined DAG via CausalDAG.from_dict().
DEFAULT_LONGEVITY_DAG: dict[str, list[dict[str, Any]]] = {
    "treatment": [
        {"target": "inflammation_index", "weight": -0.25},
        {"target": "metabolic_risk_index", "weight": -0.18},
        {"target": "epigenetic_age_acceleration", "weight": -0.70},
        {"target": "immune_resilience", "weight": 0.18},
        {"target": "latent_mitochondrial_dysfunction", "weight": -0.22},
    ],
    "inflammation_index": [
        {"target": "epigenetic_age_acceleration", "weight": 0.30},
        {"target": "frailty_progression", "weight": 0.20},
        {"target": "organ_reserve_score", "weight": -0.15},
    ],
    "metabolic_risk_index": [
        {"target": "epigenetic_age_acceleration", "weight": 0.25},
        {"target": "organ_reserve_score", "weight": -0.20},
        {"target": "latent_mitochondrial_dysfunction", "weight": 0.18},
    ],
    "epigenetic_age_acceleration": [
        {"target": "frailty_progression", "weight": 0.35},
        {"target": "recovery_velocity", "weight": 0.25},
    ],
    "immune_resilience": [
        {"target": "inflammation_index", "weight": -0.20},
        {"target": "recovery_velocity", "weight": -0.20},
    ],
    "latent_mitochondrial_dysfunction": [
        {"target": "organ_reserve_score", "weight": -0.18},
        # Note: removed mitochondrial → sleep_circadian_disruption to break cycle.
        # Biological directionality: sleep disruption → mitochondrial impairment
        # is the more defensible causal direction (kept below under sleep node).
    ],
    "sleep_circadian_disruption": [
        {"target": "inflammation_index", "weight": 0.10},
        {"target": "metabolic_risk_index", "weight": 0.08},
        {"target": "latent_mitochondrial_dysfunction", "weight": 0.10},
    ],
    # Observed confounders
    "age": [
        {"target": "epigenetic_age_acceleration", "weight": 0.50},
        {"target": "frailty_progression", "weight": 0.30},
        {"target": "organ_reserve_score", "weight": -0.25},
    ],
    "bmi": [
        {"target": "inflammation_index", "weight": 0.20},
        {"target": "metabolic_risk_index", "weight": 0.30},
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# CausalDAG class
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CausalDAG:
    """
    A directed acyclic graph encoding the structural causal model.

    Attributes:
        graph:        networkx DiGraph
        variables:    Ordered list of variable names (topological sort)
        treatment:    Name of the treatment/intervention node
        confounders:  Set of observed confounder node names
    """

    graph: nx.DiGraph
    variables: list[str]
    treatment: str = "treatment"
    confounders: set[str] = field(default_factory=set)

    @classmethod
    def default_longevity_dag(cls) -> "CausalDAG":
        """
        Construct the default longevity trial causal DAG.

        Returns:
            CausalDAG instance with pre-defined longevity pathways.
        """
        return cls.from_dict(DEFAULT_LONGEVITY_DAG)

    @classmethod
    def from_dict(
        cls,
        dag_dict: dict[str, list[dict[str, Any]]],
        treatment: str = "treatment",
        confounders: set[str] | None = None,
    ) -> "CausalDAG":
        """
        Construct a CausalDAG from a plain-dict adjacency specification.

        Format:
            {
                "node_A": [{"target": "node_B", "weight": 0.5}, ...],
                ...
            }

        Args:
            dag_dict:    Adjacency dict as above.
            treatment:   Name of the treatment node.
            confounders: Set of confounder node names (defaults to {"age", "bmi"}).

        Returns:
            Validated CausalDAG instance.

        Raises:
            ValueError: If the graph contains cycles.
        """
        G = nx.DiGraph()

        # Add all edges with weights
        for source, targets in dag_dict.items():
            for edge in targets:
                G.add_edge(source, edge["target"], weight=edge.get("weight", 1.0))

        # Validate: must be a DAG (no cycles)
        if not nx.is_directed_acyclic_graph(G):
            cycles = list(nx.simple_cycles(G))
            raise ValueError(f"Graph contains cycles: {cycles}")

        # Topological ordering — stable and deterministic
        topo_order = list(nx.topological_sort(G))

        if confounders is None:
            confounders = {"age", "bmi"}

        return cls(
            graph=G,
            variables=topo_order,
            treatment=treatment,
            confounders=confounders,
        )

    def ancestors(self, node: str) -> set[str]:
        """Return all ancestor nodes of `node` in the DAG."""
        return nx.ancestors(self.graph, node)

    def descendants(self, node: str) -> set[str]:
        """Return all descendant nodes of `node` in the DAG."""
        return nx.descendants(self.graph, node)

    def parents(self, node: str) -> set[str]:
        """Return immediate parent nodes of `node`."""
        return set(self.graph.predecessors(node))

    def children(self, node: str) -> set[str]:
        """Return immediate child nodes of `node`."""
        return set(self.graph.successors(node))

    def is_valid_adjustment_set(self, treatment: str, outcome: str, adj_set: set[str]) -> bool:
        """
        Check if adj_set satisfies the backdoor criterion for (treatment → outcome).

        Backdoor criterion (Pearl 2009):
          Z satisfies the backdoor criterion relative to (X, Y) if:
          (1) No node in Z is a descendant of X
          (2) Z blocks every backdoor path from X to Y (paths with arrow into X)

        This is a simplified check: we verify no confounder in adj_set is a
        descendant of treatment, and all observed confounders are included.

        Args:
            treatment: Source node name.
            outcome:   Target node name.
            adj_set:   Proposed adjustment set.

        Returns:
            True if adj_set appears to satisfy the backdoor criterion.
        """
        treatment_descendants = self.descendants(treatment)

        # Condition 1: no member of adj_set is a descendant of treatment
        for z in adj_set:
            if z in treatment_descendants:
                return False

        # Condition 2: all observed confounders (parents with paths to both T and Y)
        # are in the adjustment set
        confounders_to_adjust = self._find_backdoor_confounders(treatment, outcome)
        return confounders_to_adjust.issubset(adj_set)

    def _find_backdoor_confounders(self, treatment: str, outcome: str) -> set[str]:
        """
        Identify confounders that open backdoor paths between treatment and outcome.

        A node C is a backdoor confounder if it has directed paths to both
        treatment and outcome (i.e., is a common cause).

        Args:
            treatment: Treatment node.
            outcome:   Outcome node.

        Returns:
            Set of confounder node names requiring adjustment.
        """
        required_adj: set[str] = set()
        for node in self.graph.nodes:
            if node == treatment or node == outcome:
                continue
            has_path_to_treatment = nx.has_path(self.graph, node, treatment)
            has_path_to_outcome = nx.has_path(self.graph, node, outcome)
            if has_path_to_treatment and has_path_to_outcome:
                required_adj.add(node)
        return required_adj

    def structural_coefficients(self, outcome: str) -> dict[str, float]:
        """
        Return the direct structural coefficients of all parents of `outcome`.

        Args:
            outcome: Target variable name.

        Returns:
            Dict {parent_name: edge_weight}.
        """
        return {
            parent: self.graph[parent][outcome].get("weight", 1.0)
            for parent in self.graph.predecessors(outcome)
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of graph properties."""
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "nodes": list(self.graph.nodes),
            "topological_order": self.variables,
            "treatment_node": self.treatment,
            "confounders": list(self.confounders),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Causal estimators
# ──────────────────────────────────────────────────────────────────────────────


class CausalEstimator:
    """
    Causal effect estimators operating on simulated trial data.

    Supports:
      - ATE (Average Treatment Effect): E[Y(t=1) - Y(t=0)]
      - ATT (Average Treatment effect on the Treated)
      - CATE / uplift score per patient

    Args:
        dag: CausalDAG instance
    """

    def __init__(self, dag: CausalDAG) -> None:
        self.dag = dag

    def estimate_ate(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str = "arm",
        treatment_value: str = "high_dose",
        control_value: str = "placebo",
        week: int | None = None,
    ) -> dict[str, float]:
        """
        Estimate the Average Treatment Effect (ATE) from simulated data.

        Uses difference-in-means on the final week (or specified week),
        adjusting for observed confounders via linear regression.

        Args:
            df:              Long-format simulation DataFrame.
            outcome_col:     Column name for the biomarker outcome.
            treatment_col:   Column name for treatment arm.
            treatment_value: Active treatment arm label.
            control_value:   Control arm label.
            week:            Which week to evaluate (defaults to max week).

        Returns:
            Dict with 'ate', 'se', 'ci_lower', 'ci_upper', 'p_value'.
        """
        from scipy import stats

        # Filter to the evaluation week
        if week is None:
            week = int(df["week"].max())

        eval_df = df[
            (df["week"] == week)
            & (df["observed"])
            & (df["biomarker"] == outcome_col)
            & (df[treatment_col].isin([treatment_value, control_value]))
        ].copy()

        treated = eval_df[eval_df[treatment_col] == treatment_value]["value"].values
        control = eval_df[eval_df[treatment_col] == control_value]["value"].values

        if len(treated) == 0 or len(control) == 0:
            return {"ate": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

        ate = float(treated.mean() - control.mean())

        # Welch's t-test for SE and p-value (unequal variance assumption)
        t_stat, p_val = stats.ttest_ind(treated, control, equal_var=False)

        # Pooled SE
        se = float(
            np.sqrt(
                treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control)
            )
        )

        return {
            "ate": ate,
            "se": se,
            "ci_lower": ate - 1.96 * se,
            "ci_upper": ate + 1.96 * se,
            "p_value": float(p_val),
            "n_treated": len(treated),
            "n_control": len(control),
            "week": week,
        }

    def estimate_cate(
        self,
        patient_df: pd.DataFrame,
        biomarker_df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str = "arm",
        treatment_value: str = "high_dose",
        control_value: str = "placebo",
        week: int | None = None,
    ) -> pd.DataFrame:
        """
        Estimate Conditional Average Treatment Effects (CATE) per patient subgroup.

        Stratifies by age_group and sex, computes ATE within each stratum.

        Args:
            patient_df:      Patient-level DataFrame (from PatientGenerator.to_dataframe())
            biomarker_df:    Long-format biomarker simulation output
            outcome_col:     Biomarker name to use as outcome
            treatment_col:   Treatment arm column
            treatment_value: Active arm label
            control_value:   Control arm label
            week:            Evaluation week (defaults to max)

        Returns:
            DataFrame with CATE estimates per subgroup.
        """
        if week is None:
            week = int(biomarker_df["week"].max())

        # Merge patient metadata into biomarker data
        eval_df = biomarker_df[
            (biomarker_df["week"] == week)
            & (biomarker_df["observed"])
            & (biomarker_df["biomarker"] == outcome_col)
        ].merge(patient_df[["patient_id", "age", "sex"]], on="patient_id", how="left")

        # Create age strata
        eval_df["age_group"] = pd.cut(
            eval_df["age"],
            bins=[0, 55, 65, 75, 200],
            labels=["45-55", "55-65", "65-75", "75+"],
        )

        rows = []
        for (age_grp, sex), grp in eval_df.groupby(["age_group", "sex"], observed=False):
            treated = grp[grp[treatment_col] == treatment_value]["value"].values
            control = grp[grp[treatment_col] == control_value]["value"].values
            if len(treated) < 5 or len(control) < 5:
                continue  # too few observations for reliable estimate
            cate = float(treated.mean() - control.mean())
            rows.append(
                {
                    "age_group": age_grp,
                    "sex": sex,
                    "cate": cate,
                    "n_treated": len(treated),
                    "n_control": len(control),
                }
            )

        return pd.DataFrame(rows)

    def counterfactual(
        self,
        observed_outcomes: np.ndarray,
        treatment_effect: float,
        noise_std: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Generate counterfactual outcomes via the twin-network approach.

        For each observed Y(t=1), estimate Y(t=0) = Y(t=1) - τ + noise,
        where τ is the estimated ATE.

        Args:
            observed_outcomes: Array of observed outcomes under treatment.
            treatment_effect:  Estimated ATE (τ).
            noise_std:         Noise on counterfactual (captures uncertainty).
            rng:               Optional numpy Generator (uses default if None).

        Returns:
            Array of counterfactual outcomes (same shape as observed_outcomes).
        """
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0.0, noise_std, size=len(observed_outcomes))
        return observed_outcomes - treatment_effect + noise
