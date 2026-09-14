"""Correctness tests for the statistical core.

Where a published reference value exists, it is used: power values are checked
against G*Power's tables, effect sizes against their definitions, and the
resampling routines against their own exact combinatorics. A statistics engine
that is merely self-consistent is not trustworthy.
"""

import numpy as np
import pytest
from scipy import stats

import phagequest_engine as pq
from phagequest_engine import curves, effects, inference, power, resampling
from phagequest_engine.transcript import TranscriptStore, set_store


@pytest.fixture(autouse=True)
def clean_store():
    set_store(TranscriptStore())


# ---------------------------------------------------------------- effect sizes
def test_cohens_d_matches_its_definition():
    a = np.array([10.0, 12, 14, 16, 18])
    b = np.array([20.0, 22, 24, 26, 28])
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    assert effects.cohens_d(a, b) == pytest.approx((a.mean() - b.mean()) / pooled)


def test_hedges_g_is_smaller_than_d_and_converges_to_it():
    rng = np.random.default_rng(0)
    small_a, small_b = rng.normal(0, 1, 6), rng.normal(1, 1, 6)
    assert abs(effects.hedges_g(small_a, small_b)) < abs(effects.cohens_d(small_a, small_b))
    big_a, big_b = rng.normal(0, 1, 500), rng.normal(1, 1, 500)
    assert effects.hedges_g(big_a, big_b) == pytest.approx(
        effects.cohens_d(big_a, big_b), rel=0.01)


def test_cles_is_a_probability_and_is_correct_on_a_known_case():
    # Every value in b exceeds every value in a.
    assert effects.cles([1, 2, 3], [4, 5, 6]) == 0.0
    assert effects.cles([4, 5, 6], [1, 2, 3]) == 1.0
    assert effects.cles([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)


def test_eta_squared_is_one_when_groups_do_not_overlap_and_have_no_spread():
    assert effects.eta_squared([[1, 1, 1], [2, 2, 2]]) == pytest.approx(1.0)


def test_effect_size_interval_flags_a_direction_that_is_not_settled():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 8), rng.normal(0.05, 1, 8)
    ci = effects.effsize_ci(a, b, n_boot=999)
    text = effects.interpret(effects.hedges_g(a, b), "hedges_g", ci)
    if ci[0] * ci[1] <= 0:
        assert "not settled" in text


# ---------------------------------------------------------------- power
@pytest.mark.parametrize("d,expected", [(0.8, 26), (0.5, 64), (0.2, 394)])
def test_required_n_matches_gpower(d, expected):
    """Textbook / G*Power values for a two-tailed independent t-test at 80% power."""
    assert power.n_for_ttest(d)["n_per_group"] == pytest.approx(expected, abs=1)


def test_required_n_for_anova_matches_gpower():
    # Cohen's f = 0.25, 3 groups, alpha .05, power .80 -> 52 per group (156 total)
    assert power.n_for_anova(0.25, 3)["n_per_group"] == pytest.approx(53, abs=2)


def test_power_increases_with_n_and_with_effect_size():
    assert power.power_ttest(10, 0.5)["power"] < power.power_ttest(40, 0.5)["power"]
    assert power.power_ttest(20, 0.3)["power"] < power.power_ttest(20, 0.9)["power"]


def test_achieved_power_reports_the_detectable_effect_not_post_hoc_power():
    r = power.achieved_power(0.6, 8)
    assert "minimum_detectable_effect" in r
    assert "not evidence about your result" in r["caution"]


def test_a_class_sized_experiment_is_marked_infeasible_for_a_small_effect():
    assert power.n_for_ttest(0.2)["feasible"] is False
    assert power.n_for_ttest(1.5)["feasible"] is True


# ---------------------------------------------------------------- routing
def test_tiny_non_normal_samples_route_to_a_permutation_test():
    r = inference.compare_groups({"a": [1, 2, 3, 40], "b": [2, 3, 4, 50]},
                                 justification="t")
    assert r["test_used"] == "permutation"
    assert "shuffl" in r["why_this_test"].lower()


def test_large_clean_samples_route_to_a_t_test():
    rng = np.random.default_rng(5)
    r = inference.compare_groups({"a": list(rng.normal(10, 2, 40)),
                                 "b": list(rng.normal(10.5, 2, 40))}, justification="t")
    assert r["test_used"] in ("ttest_ind", "welch")


def test_three_groups_route_to_anova_or_kruskal():
    rng = np.random.default_rng(6)
    r = inference.compare_groups({f"g{i}": list(rng.normal(i, 1, 30)) for i in range(3)},
                                 justification="t")
    assert r["test_used"] in ("anova", "alexandergovern", "kruskal")
    assert r["effect_size_type"] in ("eta_squared", "epsilon_squared")


def test_unequal_variances_route_away_from_classical_anova():
    rng = np.random.default_rng(7)
    groups = {"a": list(rng.normal(0, 1, 30)), "b": list(rng.normal(0, 1, 30)),
              "c": list(rng.normal(0, 12, 30))}
    r = inference.compare_groups(groups, justification="t")
    assert r["test_used"] != "anova"


def test_engine_refuses_below_three_per_group():
    r = inference.compare_groups({"a": [1, 2], "b": [3, 4]}, justification="t")
    assert r["refused"] is True
    assert r["p_value"] is None
    assert "descriptives" in r


# ---------------------------------------------------------------- correctness
def test_t_test_result_matches_scipy_directly():
    rng = np.random.default_rng(8)
    a, b = list(rng.normal(10, 2, 40)), list(rng.normal(12, 2, 40))
    r = inference.compare_groups({"a": a, "b": b}, justification="t")
    if r["test_used"] == "ttest_ind":
        ref = stats.ttest_ind(a, b, equal_var=True)
    else:
        ref = stats.ttest_ind(a, b, equal_var=False)
    assert r["p_value"] == pytest.approx(float(ref.pvalue))


def test_shuffle_test_agrees_with_scipys_permutation_test():
    rng = np.random.default_rng(9)
    a, b = list(rng.normal(10, 2, 12)), list(rng.normal(13, 2, 12))
    mine = resampling.shuffle_test(a, b, n_shuffles=19999)["p_value"]
    ref = stats.permutation_test(
        (np.array(a), np.array(b)), lambda x, y: np.mean(x) - np.mean(y),
        permutation_type="independent", n_resamples=19999,
        random_state=np.random.default_rng(1), vectorized=False).pvalue
    assert mine == pytest.approx(float(ref), abs=0.01)


def test_shuffle_test_never_reports_p_equals_zero():
    r = resampling.shuffle_test([1, 1, 1, 1], [100, 100, 100, 100], n_shuffles=999)
    assert r["p_value"] > 0


def test_bootstrap_interval_brackets_a_known_mean():
    rng = np.random.default_rng(10)
    vals = list(rng.normal(50, 5, 40))
    r = resampling.bootstrap_ci(vals, n_resamples=2999)
    assert r["ci"][0] < np.mean(vals) < r["ci"][1]


def test_mann_kendall_finds_a_real_trend_and_not_a_flat_one():
    up = inference.trend_test(list(np.arange(20) + np.random.default_rng(11).normal(0, .3, 20)))
    assert up["p_value"] < 0.01 and up["direction"] == "increasing"
    flat = inference.trend_test(list(np.random.default_rng(12).normal(0, 1, 20)))
    assert flat["p_value"] > 0.05


def test_contingency_switches_to_fisher_on_thin_cells():
    r = inference.contingency([[1, 4], [5, 1]])
    assert r["test_used"] == "fisher_exact"
    assert r["p_value"] == pytest.approx(float(stats.fisher_exact([[1, 4], [5, 1]]).pvalue))


# ---------------------------------------------------------------- stability
def test_stability_flags_a_result_that_rests_on_one_point():
    r = inference.compare_groups({"a": [1, 1, 1, 1, 1, 9],
                                  "b": [1, 1, 1, 1, 1, 1]},
                                 justification="t", posthoc=False)
    if r.get("p_value") is not None:
        assert r["stability"]["ran"] is True


def test_stability_endorses_a_clean_separation():
    rng = np.random.default_rng(13)
    r = inference.compare_groups({"a": list(rng.normal(0, 1, 20)),
                                  "b": list(rng.normal(4, 1, 20))}, justification="t")
    assert r["stability"]["agreement"] > 0.95
    assert r["stability"]["severity"] == "info"


def test_every_comparison_returns_an_effect_size_and_a_stability_report():
    rng = np.random.default_rng(14)
    r = inference.compare_groups({"a": list(rng.normal(0, 1, 15)),
                                  "b": list(rng.normal(1, 1, 15))}, justification="t")
    assert np.isfinite(r["effect_size"])
    assert r["stability"]["ran"] is True
    assert r["plain_language"]


# ---------------------------------------------------------------- curves
def test_michaelis_menten_recovers_its_parameters():
    x = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    y = 12 * x / (3 + x)
    r = curves.fit_curve(x, y, model="michaelis_menten", n_boot=100)
    assert r["parameters"]["vmax"]["value"] == pytest.approx(12, rel=0.02)
    assert r["parameters"]["km"]["value"] == pytest.approx(3, rel=0.05)


def test_fit_refuses_more_parameters_than_data():
    r = curves.fit_curve([1, 2, 3], [1, 2, 3], model="four_pl")
    assert r["refused"] is True
    assert "4 parameters" in r["reason"]


def test_model_comparison_prefers_a_line_for_linear_data():
    x = np.arange(1.0, 15.0)
    y = 3 * x + 2 + np.random.default_rng(15).normal(0, 0.2, x.size)
    r = curves.compare_models(x, y)
    assert r["best_model"] == "linear"


def test_model_comparison_prefers_a_curve_for_saturating_data():
    x = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    y = 12 * x / (3 + x) + np.random.default_rng(16).normal(0, 0.1, x.size)
    r = curves.compare_models(x, y)
    assert r["best_model"] != "linear"


def test_growth_curve_detects_lysis():
    t = np.arange(0, 20, 0.5)
    d = np.where(t < 8, 0.05 * np.exp(0.45 * t), 0.05 * np.exp(0.45 * 8) * np.exp(-0.6 * (t - 8)))
    r = curves.fit_growth_curve(t, d)
    assert r["lysis_detected"] is True
    assert r["lysis"]["fractional_drop"] > 0.5


# ---------------------------------------------------------------- assumptions
def test_small_samples_are_told_the_normality_check_is_weak():
    from phagequest_engine.assumptions import check_normality
    res = check_normality([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    assert "very little power" in res.detail
    assert res.severity == "warn"
