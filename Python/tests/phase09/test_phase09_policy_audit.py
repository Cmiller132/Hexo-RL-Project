from tools.refactor.phase09_policy_audit import run_audit


def test_phase09_policy_audit_is_green():
    report = run_audit()
    assert report["ok"], report["findings"]

