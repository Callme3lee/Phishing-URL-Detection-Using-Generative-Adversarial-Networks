from phishing_gans_v3.tiered_url_inference import assign_tier


TIERS=[
    {"name":"allow","lower_inclusive":None,"upper_exclusive":.2},
    {"name":"review","lower_inclusive":.2,"upper_exclusive":.7},
    {"name":"block","lower_inclusive":.7,"upper_exclusive":None},
]


def test_tier_boundaries_are_lower_inclusive():
    assert assign_tier(0.0,TIERS)=="allow"
    assert assign_tier(.199999,TIERS)=="allow"
    assert assign_tier(.2,TIERS)=="review"
    assert assign_tier(.699999,TIERS)=="review"
    assert assign_tier(.7,TIERS)=="block"
    assert assign_tier(1.0,TIERS)=="block"
