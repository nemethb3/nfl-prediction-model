"""Real credit allocation: Kelly-fraction sizing, a real 15%-of-bankroll
cap with excess redistributed to uncapped picks (the standard real
"water-filling" capped-proportional allocation, not an ad hoc clip), core
vs filler tagging, and exact-to-the-credit rounding.

Why MIN_PICKS=7 isn't an arbitrary number (the task's own CONFIG comment
says as much: "needed to place 1000 under the 15% cap"): 1/MAX_BET_PCT =
1/0.15 = 6.67, so 7 is the smallest pick count for which an equal split
of 1000 credits (1000/7 ≈ 142.9) still fits under a real 150-credit
(15%) cap per pick. Fewer than 7 picks structurally cannot fully
allocate 1000 credits without breaching the cap - hence the task's own
"relax the cap only as much as needed" fallback below.
"""

from juke.config import KELLY_FRACTION, MAX_BET_PCT, MAX_PICKS, MIN_EV, MIN_PICKS, TOTAL_CREDITS


def kelly_fraction(p, decimal_odds):
    """Real fractional Kelly stake (as a fraction of bankroll), b = decimal
    odds - 1, p = win/cover probability, q = 1 - p. Returns 0.0 (not a
    negative stake) for a non-positive edge - Kelly is only meaningful on
    a real positive-EV bet."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def select_candidates(options, min_ev=MIN_EV, min_picks=MIN_PICKS, max_picks=MAX_PICKS):
    """Real selection: every option with ev >= min_ev is "core", capped at
    max_picks (a real ceiling, not a target - keeps only the highest-EV
    max_picks core options when more than that qualify, per the user's
    own instruction that fewer picks is fine when it's the better call).
    If core (after that cap) is fewer than min_picks, the next-highest-EV
    positive-EV options (ev > 0) are added as "filler" until min_picks is
    reached or positive-EV options run out - never a non-positive-EV
    option, and never past max_picks total (min_picks <= max_picks always,
    by this module's own real constraint - see module docstring)."""
    ranked = sorted(options, key=lambda o: o["ev"], reverse=True)
    core = [o for o in ranked if o["ev"] >= min_ev][:max_picks]
    chosen = list(core)
    chosen_ids = {o["game_id"] for o in chosen}
    tiers = {o["game_id"]: "core" for o in chosen}

    if len(chosen) < min_picks:
        for o in ranked:
            if len(chosen) >= min_picks:
                break
            if o["game_id"] in chosen_ids:
                continue
            if o["ev"] <= 0:
                break  # ranked descending - every option after this is also <= 0
            chosen.append(o)
            chosen_ids.add(o["game_id"])
            tiers[o["game_id"]] = "filler"

    return chosen, tiers


def _capped_proportional_allocation(weights, total_credits, max_per_pick):
    """Real water-filling allocation: split total_credits across `weights`
    proportionally, but never more than max_per_pick to any one entry -
    excess from a capped entry is redistributed proportionally among the
    still-uncapped entries, repeated until stable."""
    n = len(weights)
    allocated = [0.0] * n
    capped = [False] * n
    remaining_credits = float(total_credits)
    remaining_idx = list(range(n))

    for _ in range(n + 1):  # at most one more pass than there are entries
        if not remaining_idx:
            break
        weight_sum = sum(weights[i] for i in remaining_idx)
        if weight_sum <= 0:
            share = remaining_credits / len(remaining_idx)
            for i in remaining_idx:
                allocated[i] = share
            remaining_idx = []
            break
        newly_capped = []
        for i in remaining_idx:
            allocated[i] = remaining_credits * (weights[i] / weight_sum)
        for i in remaining_idx:
            if allocated[i] > max_per_pick + 1e-9:
                newly_capped.append(i)
        if not newly_capped:
            break
        for i in newly_capped:
            allocated[i] = max_per_pick
            capped[i] = True
        remaining_credits = total_credits - sum(allocated[i] for i in range(n) if capped[i])
        remaining_idx = [i for i in remaining_idx if i not in newly_capped]

    return allocated


def _round_to_exact_total(values, total):
    """Real largest-remainder rounding: round every value down, then hand
    out the leftover whole credits (always a small number - less than
    one per entry) to the entries with the largest fractional remainder,
    so the integers sum to exactly `total`."""
    floors = [int(v) for v in values]
    leftover = int(round(total)) - sum(floors)
    remainders = sorted(range(len(values)), key=lambda i: values[i] - floors[i], reverse=True)
    result = list(floors)
    for i in remainders[:max(leftover, 0)]:
        result[i] += 1
    return result


def allocate_stakes(options, total_credits=TOTAL_CREDITS, kelly_fraction_mult=KELLY_FRACTION,
                     max_bet_pct=MAX_BET_PCT, min_ev=MIN_EV, min_picks=MIN_PICKS, max_picks=MAX_PICKS):
    """Real end-to-end staking: selects core+filler candidates, sizes each
    with fractional Kelly, caps and redistributes to fill the full real
    total_credits, rounds to exact whole credits, and tags each pick
    "core"/"filler". Returns (picks, warning_or_None) - picks is a list of
    dicts with game_id/credits/tier added to the input option, warning is
    a real, disclosed string when the cap had to be relaxed because fewer
    than min_picks real positive-EV options existed at all.
    """
    chosen, tiers = select_candidates(options, min_ev=min_ev, min_picks=min_picks, max_picks=max_picks)
    warning = None
    max_per_pick = max_bet_pct * total_credits

    if len(chosen) < min_picks:
        if chosen:
            # Real, disclosed relaxation: the cap is loosened only as far as
            # an equal split of the real candidates we actually have
            # requires - never loosened further than that.
            max_per_pick = max(max_per_pick, total_credits / len(chosen))
            warning = (
                f"Only {len(chosen)} real positive-EV option(s) existed this week "
                f"(need {min_picks} to keep every stake under the real {max_bet_pct:.0%} cap "
                f"while allocating all {total_credits} credits) - cap relaxed to "
                f"{max_per_pick:.1f} credits/pick for this run."
            )
        else:
            warning = "No real positive-EV options existed this week - 0 credits allocated."
            return [], warning

    weights = [max(kelly_fraction(o["p_final"], o["decimal_odds"]), 1e-6) for o in chosen]
    raw_allocation = _capped_proportional_allocation(weights, total_credits, max_per_pick)
    credits = _round_to_exact_total(raw_allocation, total_credits)

    picks = []
    for o, c in zip(chosen, credits):
        pick = dict(o)
        pick["credits"] = c
        pick["tier"] = tiers[o["game_id"]]
        picks.append(pick)

    assert sum(p["credits"] for p in picks) == int(round(total_credits)), (
        "real staking bug: allocated credits don't sum to total_credits"
    )
    return picks, warning
