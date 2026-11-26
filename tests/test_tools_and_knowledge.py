def test_lookup_existing_and_missing(toolbox):
    assert toolbox.lookup_order("ord-1001")["order"]["customer_id"] == "C001"
    assert toolbox.lookup_order("ORD-9999")["error"] == "not_found"


def test_policy_search_finds_refund_limit(toolbox):
    passages = toolbox.search_policy("how much can be refunded automatically")["passages"]
    assert passages and passages[0]["id"] == "REF-2"


def test_policy_search_returns_nothing_for_nonsense(toolbox):
    assert toolbox.search_policy("zzzz qqqq")["passages"] == []


def test_escalation_creates_ticket(toolbox):
    assert toolbox.execute("escalate_to_human", {"summary": "late"}, "C001")["status"] == "ticket_created"


def test_decide_refund_flow(toolbox):
    pending = toolbox.request_refund_approval("ORD-1002", 149.0, "broken")
    assert toolbox.decide_refund(pending["approval_id"], "approve", "alice")["status"] == "issued"
    try:
        toolbox.decide_refund(pending["approval_id"], "approve", "bob")
    except ValueError:
        pass
    else:
        raise AssertionError("second decision must fail")
