from customs_ai_agent.runner import answer_question, run_review
from customs_ai_agent.schemas.models import ReviewRunRequest


def test_chat_answers_only_from_case_context():
    state = run_review(ReviewRunRequest(scenario_id="chat_evidence_bound"))
    answer = answer_question(state, "Why does this case need review?")
    assert answer.grounded is True
    assert "INV_TOTAL_MISMATCH" in answer.answer


def test_chat_refuses_unrelated_question():
    state = run_review(ReviewRunRequest(scenario_id="happy_path"))
    answer = answer_question(state, "What is the weather tomorrow?")
    assert answer.grounded is False
    assert answer.evidence_ids == []
