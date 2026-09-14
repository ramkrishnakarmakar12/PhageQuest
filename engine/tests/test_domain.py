"""Simulation, geometry, sequence, diversity and validation tests.

The simulation tests check physics rather than output shape: mass balance,
known limits, and the specific things the specification says the block must
teach (the crash and rebound, the latent period, Poisson plaques).
"""

import numpy as np
import pytest

import phagequest_engine as pq
from phagequest_engine import diversity, geometry, sequence, validation
from phagequest_engine import simulation as sim
from phagequest_engine.transcript import TranscriptStore, set_store


@pytest.fixture(autouse=True)
def clean_store():
    set_store(TranscriptStore())


# ================================================================ simulation
def test_no_phage_means_the_bacteria_just_grow():
    r = sim.simulate("lytic", P0=0.0, S0=1e5, hours=12)
    host = np.array(r["series"]["Susceptible"])
    assert host[-1] > host[0]
    assert r["summary"]["host_collapsed"] is False


def test_no_bacteria_means_the_phage_only_decay():
    r = sim.simulate("lytic", S0=0.0, P0=1e6, hours=24,
                     params={"decay": 0.2})
    p = np.array(r["series"]["Free phage"])
    assert p[-1] < p[0]
    assert np.all(np.diff(p) <= 1e-6)


def test_populations_never_go_negative():
    for model in sim.MODELS:
        r = sim.simulate(model, hours=48)
        for name, series in r["series"].items():
            assert min(series) >= -1e-6, (model, name)


def test_a_stronger_phage_clears_the_culture_faster():
    """Compare WHEN the crash happens, not how deep it is.

    Given enough hours both adsorption rates drive the host to zero, so the
    nadir depth is identical and tells you nothing. The timing is the property
    that actually distinguishes them.
    """
    weak = sim.simulate("lytic", params={"adsorption": 1e-10}, hours=36)
    strong = sim.simulate("lytic", params={"adsorption": 1e-8}, hours=36)
    assert strong["summary"]["nadir_time"] < weak["summary"]["nadir_time"]
    assert strong["summary"]["peak_phage_time"] < weak["summary"]["peak_phage_time"]


def test_resistance_produces_the_crash_and_rebound():
    """The single thing the specification says this block exists to show."""
    r = sim.simulate("resistance", hours=72,
                     params={"adsorption": 1e-8, "resistance": 1e-6})
    s = r["summary"]
    assert s["host_collapsed"] is True
    assert s["host_rebounded"] is True
    assert s["final_resistant_fraction"] > 0.5


def test_lysogeny_leaves_survivors_the_lytic_model_does_not():
    lytic = sim.simulate("lysogenic", params={"lysogeny": 0.0}, hours=36)
    lyso = sim.simulate("lysogenic", params={"lysogeny": 0.6}, hours=36)
    assert lyso["series"]["Lysogens"][-1] > lytic["series"]["Lysogens"][-1]


def test_burst_size_scales_phage_amplification():
    small = sim.simulate("lytic", params={"burst": 20}, hours=24)
    large = sim.simulate("lytic", params={"burst": 300}, hours=24)
    assert large["summary"]["peak_phage"] > small["summary"]["peak_phage"]


def test_the_erlang_chain_makes_the_latent_period_visible():
    r = sim.one_step_growth(latent=0.5, burst=100, n_stages=20)
    one = r["runs"]["1_stages"]
    many = r["runs"]["20_stages"]
    # With one compartment the rise starts immediately; with twenty it waits.
    assert many["observed_rise_start"] > one["observed_rise_start"]


def test_parameter_sweep_in_adsorption_is_monotonic_and_downward():
    """Total amplification FALLS as adsorption rises, and that is correct.

    The intuitive expectation is the opposite, which is exactly why this sweep
    is worth putting in front of a student: a stickier phage crashes its host
    sooner, so there are fewer cells left to multiply in and the total yield
    goes down. The engine reports the direction rather than assuming it.
    """
    r = sim.parameter_sweep("lytic", "adsorption", n_values=7, hours=36)
    y = r["response"]["y"]
    assert r["rows"]
    assert r["monotonic"] is True
    assert y[-1] < y[0]


def test_sweep_output_feeds_the_curve_fitter():
    r = sim.parameter_sweep("lytic", "burst", n_values=8, hours=36)
    nxt = r["next_step"]
    assert nxt["tool"] == "fit_curve"
    assert len(nxt["x"]) == len(nxt["y"]) >= 5


def test_implausible_parameters_are_warned_about_not_blocked():
    r = sim.simulate("lytic", params={"burst": 480})
    assert r["notes"]
    assert "outside the range usually measured" in r["notes"][0]


# ---------------------------------------------------------------- stochastic
def test_identical_stochastic_runs_do_not_all_agree():
    r = sim.gillespie_lytic(S0=100, P0=3, n_runs=25, adsorption=1e-3)
    outcomes = {run["phage_extinct"] for run in r["runs"]}
    assert r["n_runs"] == 25
    # Either some died out and some did not, or the peak varied.
    peaks = [max(run["P"]) for run in r["runs"]]
    assert len(outcomes) > 1 or len(set(peaks)) > 1


def test_more_starting_phage_means_less_extinction():
    r = sim.extinction_probability(P0_values=[1, 10], n_runs=40)
    assert r["rows"][0]["extinction_probability"] >= r["rows"][1]["extinction_probability"]
    assert 0 <= r["single_phage_extinction"] <= 1


def test_poisson_plaques_matches_the_exact_probability():
    from scipy import stats
    r = sim.poisson_plaques(expected_plaques=3.0, n_plates=50)
    assert r["probability_of_empty_plate"] == pytest.approx(float(stats.poisson.pmf(0, 3.0)))
    assert r["probability_of_empty_plate"] == pytest.approx(0.0498, abs=1e-3)


# ---------------------------------------------------------------- plaque
def test_a_plaque_grows_and_stays_roughly_round():
    r = sim.plaque_growth(size=101, steps=90, spread_probability=0.6)
    f = r["final"]
    assert f["plaque_area_cells"] > 100
    assert f["radial_growth_rate"] > 0
    assert 0.1 < f["circularity"] < 1.6


def test_resistance_in_the_lawn_limits_the_plaque():
    clean = sim.plaque_growth(size=101, steps=90, spread_probability=0.6,
                              resistant_fraction=0.0)
    blocked = sim.plaque_growth(size=101, steps=90, spread_probability=0.6,
                                resistant_fraction=0.5)
    assert blocked["final"]["plaque_area_cells"] < clean["final"]["plaque_area_cells"]


def test_a_subcritical_spread_probability_fizzles_and_says_why():
    r = sim.plaque_growth(size=81, steps=60, spread_probability=0.05)
    assert r["final"]["stopped_because"]
    assert r["expected_offspring_per_lysis"] < 1.5


# ---------------------------------------------------------------- lab maths
def test_titre_arithmetic_is_correct():
    r = sim.pfu_per_ml(50, 1e6, 0.1)
    assert r["titre_pfu_per_mL"] == pytest.approx(5e8)
    assert r["ci"][0] < 5e8 < r["ci"][1]
    assert r["count_quality"] == "good"


def test_a_low_count_is_marked_unusable_with_a_wide_interval():
    r = sim.pfu_per_ml(3, 1e6)
    assert r["count_quality"] == "too few"
    assert r["ci"][1] / r["ci"][0] > 5


def test_zero_plaques_gives_an_upper_limit_not_a_titre():
    r = sim.pfu_per_ml(0, 1e6)
    assert r["count_quality"] == "upper limit only"
    assert r["ci"][0] == 0.0


def test_pooling_plates_beats_averaging_their_titres():
    """Plates at DIFFERENT dilutions are where averaging goes wrong.

    At equal dilutions the two methods coincide, so the test has to use unequal
    ones -- which is also the realistic case, since a student plates a series.
    """
    counts, dils = [8, 150], [1e7, 1e6]
    r = sim.titre_from_plates(counts, dils)
    veq = sum(0.1 / d for d in dils)
    assert r["pooled_titre_pfu_per_mL"] == pytest.approx(sum(counts) / veq)
    # The 150-plaque plate carries far more information than the 8-plaque one.
    # Pooling weights it accordingly; averaging gives them equal say and lands
    # a long way from the better-measured answer.
    best_plate = max(r["per_plate_titres"])
    pooled_err = abs(r["pooled_titre_pfu_per_mL"] - best_plate)
    naive_err = abs(r["naive_mean_of_titres"] - best_plate)
    assert pooled_err < naive_err


def test_moi_one_leaves_a_third_of_cells_untouched():
    r = sim.moi(1e8, 1e8)
    assert r["moi"] == pytest.approx(1.0)
    assert r["fraction_uninfected"] == pytest.approx(np.exp(-1), abs=1e-6)


def test_the_dilution_volume_trap_is_caught():
    r = sim.dilution_series(fold=10, volume_transferred=0.1, volume_diluent=1.0)
    assert r["fold_mismatch"] is not None
    assert r["actual_fold"] == pytest.approx(11.0)


# ================================================================ geometry
def _two_clade_distances(n_a=20, n_b=8, seed=4):
    rng = np.random.default_rng(seed)
    def genome(bias, L=20000):
        w = np.ones(4); w[bias] = 1.8; w /= w.sum()
        return "".join(rng.choice(list("ACGT"), L, p=w))
    names = [f"A{i:02d}" for i in range(n_a)] + [f"B{i:02d}" for i in range(n_b)]
    seqs = {n: genome(0 if n.startswith("A") else 2) for n in names}
    nm, X = geometry.tnf_matrix(seqs)
    D = np.array(geometry.cosine_distance_matrix(X, names=nm)["matrix"])
    return nm, D


def test_tetranucleotide_vector_has_256_entries_that_sum_to_one():
    r = geometry.tetranucleotide_vector("ACGT" * 500)
    assert len(r["frequencies"]) == 256
    assert sum(r["frequencies"]) == pytest.approx(1.0)


def test_a_genome_is_closer_to_itself_than_to_anything_else():
    nm, D = _two_clade_distances()
    assert np.allclose(np.diag(D), 0.0)
    for i in range(D.shape[0]):
        off = np.delete(D[i], i)
        assert D[i, i] < off.min()


def test_distance_matrix_is_symmetric_and_non_negative():
    _, D = _two_clade_distances()
    assert np.allclose(D, D.T, atol=1e-12)
    assert (D >= -1e-12).all()


def test_the_component_sweep_finds_the_planted_two_clade_split():
    nm, D = _two_clade_distances(n_a=20, n_b=8)
    r = geometry.component_sweep(D, names=nm, k_max=25)
    pl = r["plateau"]
    assert pl is not None
    assert pl["n_components"] == 2
    assert sorted(pl["sizes"]) == [8, 20]
    assert pl["length"] >= 3


def test_a_structureless_set_yields_no_plateau_and_says_so():
    rng = np.random.default_rng(2)
    X = rng.random((25, 256))
    D = np.array(geometry.cosine_distance_matrix(X)["matrix"])
    r = geometry.component_sweep(D, k_max=20)
    assert r["plateau"] is None or r["plateau"]["length"] <= 3


def test_curvature_is_bounded_and_computed_on_every_edge():
    _, D = _two_clade_distances(n_a=12, n_b=6)
    r = geometry.ollivier_ricci(D, k=5)
    assert r["n_edges"] > 0
    ks = [e["curvature"] for e in r["edges"]]
    assert all(-3.0 <= k <= 1.0 for k in ks)


def test_curvature_of_a_complete_graph_is_positive():
    """On a graph where everything neighbours everything, no edge is a bottleneck."""
    n = 8
    D = np.ones((n, n)) - np.eye(n)
    r = geometry.ollivier_ricci(D, k=n - 1)
    assert r["summary"]["mean"] > 0


def test_bridge_analysis_reports_both_weightings_and_names_hubs():
    nm, D = _two_clade_distances(n_a=16, n_b=6)
    r = geometry.bridge_analysis(D, names=nm, k_max=22)
    assert "unweighted" in r and "weighted" in r
    assert r["weighting_caveat"]
    assert r["unweighted"]["n_bridge_edges"] >= 1
    assert r["unweighted"]["hubs"]


def test_the_weighting_caveat_quotes_the_numbers_it_computed():
    """The caveat must never assert a difference the run did not produce."""
    nm, D = _two_clade_distances(n_a=12, n_b=6)
    r = geometry.bridge_analysis(D, names=nm, k_max=18)
    um = r["unweighted"]["overall_mean_curvature"]
    assert f"{um:+.3f}" in r["weighting_caveat"]


def test_sinkhorn_approximates_the_exact_solver():
    _, D = _two_clade_distances(n_a=10, n_b=5)
    exact = geometry.ollivier_ricci(D, k=4, method="exact")["summary"]["mean"]
    approx = geometry.ollivier_ricci(D, k=4, method="sinkhorn")["summary"]["mean"]
    assert approx == pytest.approx(exact, abs=0.1)


# ================================================================ sequence
def test_the_codon_table_is_the_standard_one():
    t = sequence.CODON_TABLE
    assert len(t) == 64
    assert t["ATG"] == "M" and t["TGG"] == "W"
    assert {t["TAA"], t["TAG"], t["TGA"]} == {"*"}
    assert t["TTT"] == "F" and t["GGG"] == "G" and t["CAT"] == "H"


def test_translation_is_correct_and_shows_its_working():
    r = sequence.translate("ATGGCTTTAGGTCATTAA")
    assert r["protein"] == "MALGH*"
    assert r["codon_detail"][0]["codon"] == "ATG"
    assert r["codon_detail"][0]["name"] == "Methionine"


def test_reverse_complement_is_an_involution():
    s = "ATGGCTTTAGGTCATTAA"
    once = sequence.reverse_complement(s)["sequence"]
    twice = sequence.reverse_complement(once)["sequence"]
    assert twice == s


def test_gc_content_is_correct():
    assert sequence.gc_content("GGCC")["gc_content"] == pytest.approx(1.0)
    assert sequence.gc_content("ATAT")["gc_content"] == pytest.approx(0.0)
    assert sequence.gc_content("ATGC")["gc_content"] == pytest.approx(0.5)


def test_coding_density_cannot_exceed_one():
    rng = np.random.default_rng(3)
    g = "".join(rng.choice(list("ACGT"), 6000))
    r = sequence.find_orfs(g, min_length_aa=30)
    assert 0.0 <= r["coding_density_estimate"] <= 1.0


def test_the_naive_caller_states_its_limitations():
    rng = np.random.default_rng(3)
    g = "".join(rng.choice(list("ACGT"), 3000))
    r = sequence.find_orfs(g)
    assert len(r["limitations"]) >= 3
    assert any("over-calls" in x for x in r["limitations"])


def test_fasta_round_trip():
    text = ">phage1 first\nATGGCT\nTTAGGT\n>phage2\nATGAAA\n"
    r = sequence.parse_fasta(text)
    assert r["n_records"] == 2
    assert r["records"][0]["sequence"] == "ATGGCTTTAGGT"
    assert r["records"][0]["id"] == "phage1"


# ================================================================ diversity
_COMMUNITIES = {
    "even": {"a": 25, "b": 25, "c": 25, "d": 25},
    "uneven": {"a": 97, "b": 1, "c": 1, "d": 1},
}


def test_an_even_community_is_more_diverse_than_a_dominated_one():
    r = diversity.alpha_diversity(_COMMUNITIES)
    assert r["samples"]["even"]["shannon"] > r["samples"]["uneven"]["shannon"]
    assert r["samples"]["even"]["pielou_evenness"] == pytest.approx(1.0, abs=1e-9)


def test_shannon_matches_its_definition():
    r = diversity.alpha_diversity({"s": {"a": 1, "b": 1, "c": 1, "d": 1}})
    assert r["samples"]["s"]["shannon"] == pytest.approx(np.log(4))


def test_clr_values_sum_to_zero_per_sample():
    r = diversity.clr_transform(_COMMUNITIES)
    for row in r["clr"]:
        assert sum(row) == pytest.approx(0.0, abs=1e-9)


def test_uneven_depth_triggers_a_warning():
    r = diversity.alpha_diversity({"shallow": {"a": 2, "b": 1},
                                   "deep": {"a": 200, "b": 150, "c": 90}})
    assert r["sequencing_depth_warning"]


def test_composition_comparison_refuses_the_t_test_route():
    rng = np.random.default_rng(21)
    samples, groups = {}, {}
    for i in range(10):
        g = "layer1" if i < 5 else "layer2"
        base = [60, 20, 20] if g == "layer1" else [20, 20, 60]
        samples[f"s{i}"] = {t: float(max(1, b + rng.normal(0, 4)))
                            for t, b in zip("abc", base)}
        groups[f"s{i}"] = g
    r = diversity.compare_composition(samples, groups)
    assert "cannot run a t-test" in r["why_not_a_t_test"]
    assert r["permanova"]["p_value"] < 0.05


def test_permanova_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(22)
    samples = {f"s{i}": {t: float(abs(rng.normal(30, 5))) for t in "abc"} for i in range(12)}
    groups = {f"s{i}": ("x" if i % 2 else "y") for i in range(12)}
    r = diversity.compare_composition(samples, groups)
    assert r["permanova"]["p_value"] > 0.05


# ================================================================ validation
def test_an_impossible_ph_is_caught_with_a_helpful_message():
    r = validation.validate_table([{"sample": "a", "pH": 19}], template="water_quality")
    assert r["valid"] is False
    msg = r["issues"][0]["message"]
    assert "pH only" in msg or "pH runs from 0 to 14" in msg
    assert "1.9" in msg


def test_text_in_a_numeric_column_is_caught_and_names_the_row():
    r = validation.validate_table(
        [{"sample": "a", "pH": 7.1}, {"sample": "b", "pH": 7.3}, {"sample": "c", "pH": "seven"}],
        template="water_quality")
    bad = [i for i in r["issues"] if i["severity"] == "block"]
    assert bad and bad[0]["row"] == 4


def test_a_missing_required_column_suggests_the_closest_name():
    r = validation.validate_table([{"sampl": "a", "pH": 7}], template="water_quality")
    msgs = " ".join(i["message"] for i in r["issues"])
    assert "sample" in msgs


def test_duplicate_rows_are_flagged_as_inflating_n():
    row = {"sample": "a", "pH": 7.0}
    r = validation.validate_table([row, dict(row), {"sample": "b", "pH": 7.5}],
                                  template="water_quality")
    assert any("identical" in i["message"] for i in r["issues"])


def test_schema_inference_recognises_a_curriculum_template():
    rows = [{"sample": "a", "dilution_factor": 1e6, "plaque_count": 42},
            {"sample": "b", "dilution_factor": 1e6, "plaque_count": 37}]
    r = validation.infer_schema(rows)
    assert r["suggested_template"] == "plaque_assay"


def test_coercion_never_invents_a_measurement():
    r = validation.coerce_table([{"n": "12"}, {"n": "TNTC"}], numeric_columns=["n"])
    assert r["n_out"] == 1
    assert r["dropped"][0]["value"] == "TNTC"
