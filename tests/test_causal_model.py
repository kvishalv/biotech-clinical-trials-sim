"""tests/test_causal_model.py — Tests for simulation/causal_model.py"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulation.causal_model import CausalDAG, CausalEstimator, DEFAULT_LONGEVITY_DAG


class TestCausalDAG:
    def test_default_dag_loads(self):
        dag = CausalDAG.default_longevity_dag()
        assert dag.graph.number_of_nodes() > 0
        assert dag.graph.number_of_edges() > 0

    def test_default_dag_is_acyclic(self):
        import networkx as nx
        dag = CausalDAG.default_longevity_dag()
        assert nx.is_directed_acyclic_graph(dag.graph)

    def test_topological_order_is_set(self):
        dag = CausalDAG.default_longevity_dag()
        assert len(dag.variables) > 0
        assert "treatment" in dag.variables

    def test_treatment_node_present(self):
        dag = CausalDAG.default_longevity_dag()
        assert "treatment" in dag.graph.nodes

    def test_from_dict_creates_correct_edges(self):
        simple = {"A": [{"target": "B", "weight": 0.5}], "B": [{"target": "C", "weight": 0.3}]}
        dag = CausalDAG.from_dict(simple, treatment="A")
        assert dag.graph.has_edge("A", "B")
        assert dag.graph.has_edge("B", "C")
        assert dag.graph["A"]["B"]["weight"] == 0.5

    def test_cycle_raises_value_error(self):
        cyclic = {
            "A": [{"target": "B", "weight": 1.0}],
            "B": [{"target": "A", "weight": 1.0}],  # creates cycle
        }
        with pytest.raises(ValueError, match="cycles"):
            CausalDAG.from_dict(cyclic)

    def test_ancestors_and_descendants(self):
        dag = CausalDAG.default_longevity_dag()
        # Treatment should have descendants (endpoints it affects)
        descendants = dag.descendants("treatment")
        assert len(descendants) > 0

    def test_structural_coefficients(self):
        dag = CausalDAG.default_longevity_dag()
        coeffs = dag.structural_coefficients("epigenetic_age_acceleration")
        assert len(coeffs) > 0
        # Treatment → epigenetic age should be negative (protective)
        if "treatment" in coeffs:
            assert coeffs["treatment"] < 0

    def test_summary_structure(self):
        dag = CausalDAG.default_longevity_dag()
        s = dag.summary()
        assert "n_nodes" in s
        assert "n_edges" in s
        assert s["n_nodes"] > 0

    def test_backdoor_criterion_check(self):
        dag = CausalDAG.default_longevity_dag()
        # Age and BMI should satisfy backdoor for treatment → epigenetic_age_acceleration
        result = dag.is_valid_adjustment_set(
            treatment="treatment",
            outcome="epigenetic_age_acceleration",
            adj_set={"age", "bmi"},
        )
        # Just check it returns a bool without error
        assert isinstance(result, bool)


class TestCausalEstimator:
    def test_estimate_ate_returns_dict(self, biomarker_df):
        dag = CausalDAG.default_longevity_dag()
        est = CausalEstimator(dag)
        result = est.estimate_ate(
            df=biomarker_df,
            outcome_col="inflammation_index",
            treatment_value="high_dose",
            control_value="placebo",
        )
        assert "ate" in result
        assert "p_value" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_ate_ci_lower_lt_upper(self, biomarker_df):
        dag = CausalDAG.default_longevity_dag()
        est = CausalEstimator(dag)
        result = est.estimate_ate(
            df=biomarker_df,
            outcome_col="inflammation_index",
        )
        if not (np.isnan(result["ci_lower"]) or np.isnan(result["ci_upper"])):
            assert result["ci_lower"] <= result["ci_upper"]

    def test_estimate_cate_returns_dataframe(self, patient_df, biomarker_df):
        import pandas as pd
        dag = CausalDAG.default_longevity_dag()
        est = CausalEstimator(dag)
        cate = est.estimate_cate(
            patient_df=patient_df,
            biomarker_df=biomarker_df,
            outcome_col="inflammation_index",
        )
        assert isinstance(cate, pd.DataFrame)

    def test_counterfactual_same_shape(self):
        dag = CausalDAG.default_longevity_dag()
        est = CausalEstimator(dag)
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cf = est.counterfactual(obs, treatment_effect=0.5, rng=np.random.default_rng(42))
        assert cf.shape == obs.shape
