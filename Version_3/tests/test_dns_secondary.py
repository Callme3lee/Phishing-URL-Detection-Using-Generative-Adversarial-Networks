import numpy as np

from phishing_gans_v3.train_dns_secondary import FEATURES,select_combined_threshold,vector


def test_dns_vector_excludes_query_time_and_primary_score():
    row={name:"1" for name in FEATURES}
    row["host_a_query_ms"]="9999";row["ensemble_score"]="0.99"
    values=vector(row)
    assert len(values)==len(FEATURES)
    assert all("query_ms" not in name for name in FEATURES)
    assert "ensemble_score" not in FEATURES


def test_combined_threshold_obeys_fpr_cap_and_recovers_routed_phishing():
    labels=np.asarray([0,0,1,1],dtype=np.uint8)
    primary=np.zeros(4,dtype=np.float32);routed=np.arange(4);dns=np.asarray([.1,.8,.9,.7])
    threshold,prediction=select_combined_threshold(labels,primary,routed,dns,1.1,.5)
    assert threshold==.7
    assert prediction.tolist()==[False,True,True,True]
