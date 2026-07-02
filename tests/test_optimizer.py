"""
Tests for the vendored foreman_optimizer package — covers the TODO #6
checklist from foreman_optimizer/CLAUDE.md: filler idempotence,
structure-protection, skeleton stability, promotion threshold, and savings
accounting (no double-counting between filler_removal/redundancy_removal).
"""
from __future__ import annotations

from foreman_optimizer import (
    TIER_0,
    TIER_1_ELIGIBLE,
    FrequencyPromoter,
    InMemoryStore,
    ir,
    parse,
    run_tier0,
)
from foreman_optimizer.fingerprint import SQLiteStore, fingerprint, skeleton
from foreman_optimizer.rules import rule_strip_filler


class TestFillerIdempotence:
    def test_second_pass_makes_no_further_changes(self):
        text = "As a very helpful AI assistant, I would like you to please summarize this."
        once = rule_strip_filler(text)
        twice = rule_strip_filler(once)
        assert once == twice


class TestStructureProtection:
    def test_code_fence_survives_tier0(self):
        prompt = (
            "Please clean up this messy query.\n\n"
            "```sql\nSELECT   *   FROM    users WHERE  id = 1\n```\n\n"
            "Thanks so much!"
        )
        result = run_tier0(parse(prompt))
        assert "```sql\nSELECT   *   FROM    users WHERE  id = 1\n```" in result.optimized

    def test_quoted_string_survives_tier0(self):
        prompt = 'Please check if this string is valid: "  extra   spaces  , kept". Thanks!'
        result = run_tier0(parse(prompt))
        assert '"  extra   spaces  , kept"' in result.optimized


class TestSkeletonStability:
    def test_same_skeleton_across_differing_slot_values(self):
        template = 'Look up order {} and summarize the ticket "{}".'
        a = template.format("48213", "arrived broken")
        b = template.format("99001", "wrong color shipped")
        assert skeleton(a) == skeleton(b)
        assert fingerprint(a) == fingerprint(b)

    def test_different_templates_hash_differently(self):
        a = "Summarize this support ticket."
        b = "Translate this document to French."
        assert fingerprint(a) != fingerprint(b)


class TestPromotionThreshold:
    def test_promotes_at_exactly_threshold(self):
        promoter = FrequencyPromoter(InMemoryStore(), threshold=3)
        prompt = "Classify the sentiment of this ticket: order 111 arrived broken."

        obs = None
        for _ in range(3):
            obs = promoter.observe(prompt, tokens=ir.estimate_tokens(prompt))
        assert obs.count == 3
        assert obs.tier == TIER_1_ELIGIBLE

    def test_below_threshold_stays_tier0(self):
        promoter = FrequencyPromoter(InMemoryStore(), threshold=3)
        prompt = "Classify the sentiment of this ticket: order 222 arrived broken."
        obs = promoter.observe(prompt, tokens=ir.estimate_tokens(prompt))
        assert obs.count == 1
        assert obs.tier == TIER_0


class TestSavingsAccounting:
    def test_no_double_counted_savings_between_filler_and_redundancy(self):
        # Filler removal runs before dedupe, so by the time dedupe sees the
        # text the duplicated clauses are already shortened. Sum of the two
        # categories' savings should equal exactly what those two rules
        # removed — not more (which would indicate overlap/double counting).
        prompt = (
            "Please note that I want you to be thorough. "
            "Please note that I want you to be thorough. "
            "Summarize this support ticket."
        )
        result = run_tier0(parse(prompt))
        by_category = {r.category: r for r in result.results}
        filler = by_category["filler_removal"]
        dedupe = by_category["redundancy_removal"]

        # dedupe operates on filler's *output*, not the raw original text.
        assert dedupe.tokens_before == filler.tokens_after

    def test_output_shaping_books_both_legs(self):
        prompt = "Summarize this support ticket."  # no length cap / format spec
        result = run_tier0(parse(prompt))
        shaping = next(r for r in result.results if r.category == "output_shaping")
        # Appends a directive -> input tokens increase (a negative saving leg).
        assert shaping.tokens_after > shaping.tokens_before

        from foreman_optimizer.categories import savings_from_rules
        reports = savings_from_rules(result.results)
        axes = {r.category: r.axis for r in reports if r.category == "output_shaping"}
        input_legs = [r for r in reports if r.category == "output_shaping" and r.axis == "input"]
        output_legs = [r for r in reports if r.category == "output_shaping" and r.axis == "output"]
        assert len(input_legs) == 1 and input_legs[0].tokens_saved < 0
        assert len(output_legs) == 1 and output_legs[0].tokens_saved > 0
        assert axes  # sanity: both legs present


class TestSQLiteStoreThreadSafety:
    def test_usable_from_a_different_thread_than_it_was_created_on(self, tmp_path):
        # Streamlit (via st.cache_resource) can hand one cached instance to
        # whichever worker thread serves a later rerun — sqlite3 connections
        # default to single-thread-only and raise
        # "SQLite objects created in a thread can only be used in that same
        # thread" the first time that happens. Regression test for that.
        import threading

        store = SQLiteStore(str(tmp_path / "templates.db"))
        promoter = FrequencyPromoter(store)
        promoter.observe("some prompt", tokens=10)

        errors = []

        def other_thread_call():
            try:
                promoter.observe("some prompt", tokens=10)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=other_thread_call)
        t.start()
        t.join()

        assert not errors, f"cross-thread access raised: {errors}"
