import numpy as np

# Universal gas constant, J/(mol*K) — matches the value used by the NASA
# Glenn thermodynamic polynomial reference (NASA/TP-2002-211556).
R = 8.314510


def calculate_cp(T, a, b, c, d, e, f, g):
    """Molar heat capacity Cp(T) in J/(mol*K) from the NASA Glenn 7-term
    polynomial for a *single* temperature interval.

    Cp°(T)/R = a*T^-2 + b*T^-1 + c + d*T + e*T^2 + f*T^3 + g*T^4

    T may be a scalar or a numpy array; a and friends are the seven
    NASA a1..a7 coefficients for the interval that T falls in. Callers
    with data spanning multiple intervals should use
    calculate_cp_piecewise instead, since coefficients differ by
    interval.
    """
    T = np.asarray(T, dtype=float)

    if np.any(T <= 0):
        raise ValueError("Temperature must be greater than 0 K.")

    Cp_over_R = a * T**-2 + b * T**-1 + c + d * T + e * T**2 + f * T**3 + g * T**4
    Cp = R * Cp_over_R

    return float(Cp) if Cp.ndim == 0 else Cp


def calculate_cp_piecewise(T, segments):
    """Evaluate Cp(T) across one or more NASA temperature-interval
    segments belonging to the same material.

    A single material (e.g. Aluminium) is stitched together from every
    phase/interval NASA fitted for it (solid, liquid, different crystal
    structures, etc.), each with its own a..g coefficients. Evaluating
    them as one continuous function is what lets a plot show the sharp
    rise/fall in Cp at a phase-transition temperature (e.g. melting
    point) instead of splitting the material into separate curves.

    T: scalar or array-like of temperatures (K)
    segments: iterable of objects/rows exposing t_min, t_max, a, b, c,
        d, e, f, g (e.g. rows of a pandas DataFrame), covering all
        phases/intervals for the material, in any order.

    Returns Cp in J/(mol*K), matching the shape of T. Temperatures that
    do not fall within any segment's [t_min, t_max] are returned as NaN
    (this shows up as a gap in the line for the handful of materials
    with a genuine data gap between fitted intervals).
    """
    scalar_input = np.isscalar(T) or np.asarray(T).ndim == 0
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    Cp = np.full(T_arr.shape, np.nan)

    # Sort by t_min so that, for the rare/tiny overlaps between adjacent
    # NASA intervals, the higher-temperature phase's coefficients win at
    # the boundary rather than the outcome depending on input order.
    for seg in sorted(segments, key=lambda s: s.t_min):
        mask = (T_arr >= seg.t_min) & (T_arr <= seg.t_max)
        if np.any(mask):
            Cp[mask] = calculate_cp(
                T_arr[mask], seg.a, seg.b, seg.c, seg.d, seg.e, seg.f, seg.g
            )

    if scalar_input:
        return float(Cp[0]) if not np.isnan(Cp[0]) else None
    return Cp


def build_material_curve(segments, t_lo, t_hi, points_per_segment=60):
    """Build a plotting-ready (T, Cp) curve for a material across all of
    its NASA phase/interval segments, restricted to [t_lo, t_hi].

    Rather than sampling one shared T grid and letting
    calculate_cp_piecewise pick a segment per point (which spreads a
    phase-transition jump across whatever grid spacing happens to
    straddle it), this samples each segment on its own dense grid and
    concatenates them in temperature order. Adjacent segments share
    their boundary temperature but are evaluated with each side's own
    coefficients, so the two points at that shared T sit at different
    Cp values -- when plotted as a connected line this draws a true
    vertical jump exactly at the phase-transition temperature (e.g. a
    melting point), instead of a slanted, smeared-out approximation.

    A handful of NASA source intervals overlap by a few K (rounding) or,
    for a couple of materials, by tens of K (e.g. SrCL2's alpha fit runs
    to 990 K while its beta fit starts at 900 K -- both are "valid" over
    that 90 K window, and beta's fit reaches a much higher Cp there than
    alpha's does). Left alone, plotting every segment's full
    [t_min, t_max] would make the line move backward in temperature
    across the overlap. To avoid that while still preserving every
    segment's own values (rather than just clipping the higher-priority
    segment's start and silently discarding its data in the overlap),
    each segment's *end* is trimmed down to the t_min of the next
    (higher t_min) overlapping segment, working from the highest t_min
    down to the lowest. That leaves each segment's own start and
    coefficients untouched -- so its natural values are never
    discarded -- and gives the higher-t_min segment priority for every
    T in the overlap, matching the convention calculate_cp_piecewise
    already uses for point queries.

    Returns (T, Cp) numpy arrays, both empty if no segment overlaps
    [t_lo, t_hi].
    """
    ordered = sorted(segments, key=lambda s: s.t_min)

    # Determine each segment's plotted upper bound: trimmed down to the
    # start of the next (higher-priority) segment if they overlap, so
    # that segment keeps its full, untouched range and always wins the
    # overlap -- worked out back-to-front so multi-segment overlap
    # chains resolve correctly.
    trimmed_hi = [seg.t_max for seg in ordered]
    for i in range(len(ordered) - 2, -1, -1):
        trimmed_hi[i] = min(trimmed_hi[i], ordered[i + 1].t_min)

    T_parts = []
    Cp_parts = []

    for seg, hi_bound in zip(ordered, trimmed_hi):
        seg_lo = max(seg.t_min, t_lo)
        seg_hi = min(hi_bound, t_hi)
        if seg_lo >= seg_hi:
            continue
        T_seg = np.linspace(seg_lo, seg_hi, points_per_segment)
        Cp_seg = calculate_cp(T_seg, seg.a, seg.b, seg.c, seg.d, seg.e, seg.f, seg.g)
        T_parts.append(T_seg)
        Cp_parts.append(Cp_seg)

    if not T_parts:
        return np.array([]), np.array([])

    return np.concatenate(T_parts), np.concatenate(Cp_parts)