from typing import Dict, Optional
import demes


def IM_symmetric_model(sampled: Dict[str, float], cfg: Optional[Dict] = None) -> demes.Graph:
    """
    Split + symmetric migration (YRI/CEU), but *no separate ancestral-only deme*.
    YRI carries the ancestral epoch pre-split; CEU branches off at T_split.

    This keeps the first deme ("YRI") extant at time 0 => pop0 is extant after msprime.from_demes().
    """

    required_keys = ["N_anc", "N_YRI", "N_CEU", "m", "T_split"]
    for k in required_keys:
        assert k in sampled, f"Missing required key: {k}"

    N0 = float(sampled["N_anc"])
    N1 = float(sampled["N_YRI"])
    N2 = float(sampled["N_CEU"])
    T  = float(sampled["T_split"])
    m  = float(sampled["m"])

    assert T > 0, "T_split must be > 0"

    b = demes.Builder(time_units="generations", generation_time=1)

    # Root extant deme YRI:
    #   - from present back to T: size N1
    #   - older than T: ancestral size N0
    b.add_deme(
        "YRI",
        epochs=[
            dict(start_size=N0, end_time=T),  # older epoch (ancestral)
            dict(start_size=N1, end_time=0),  # recent epoch (modern)
        ],
    )

    # CEU branches off at split time
    b.add_deme(
        "CEU",
        ancestors=["YRI"],
        start_time=T,
        epochs=[dict(start_size=N2, end_time=0)],
    )

    # Symmetric migration AFTER split (times when both exist: [0, T])
    if m > 0:
        b.add_migration(source="YRI", dest="CEU", rate=m, start_time=T, end_time=0)
        b.add_migration(source="CEU", dest="YRI", rate=m, start_time=T, end_time=0)

    return b.resolve()
