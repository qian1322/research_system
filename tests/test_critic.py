from research_system.nodes.critic import should_revise


def test_should_revise_when_not_approved():
    assert should_revise({"quality_approved": False}) == "revise"


def test_should_revise_when_approved():
    assert should_revise({"quality_approved": True}) == "approved"


def test_should_revise_default_missing_key():
    assert should_revise({}) == "revise"
