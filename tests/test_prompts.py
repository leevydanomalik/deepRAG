from rag.core.prompts import (
    ANSWER_PROMPT,
    KG_EXTRACTION_PROMPT,
    LOOP_PLAN_PROMPT,
)


def test_prompts_have_required_placeholders():
    assert "{question}" in ANSWER_PROMPT
    assert "{context}" in ANSWER_PROMPT
    assert "{text}" in KG_EXTRACTION_PROMPT
    assert "{question}" in LOOP_PLAN_PROMPT
