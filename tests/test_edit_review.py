from research_system.nodes.edit_review import should_continue_editing


def test_should_continue_editing_when_instruction_given():
    assert should_continue_editing({"edit_instructions": "add a section on X"}) == "edit"


def test_should_continue_editing_when_empty():
    assert should_continue_editing({"edit_instructions": ""}) == "done"


def test_should_continue_editing_default_missing_key():
    assert should_continue_editing({}) == "done"
