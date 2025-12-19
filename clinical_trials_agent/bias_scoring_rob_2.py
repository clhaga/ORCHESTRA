from typing import Dict


def score_randomization_process(answers: Dict[str, str]) -> str:
    """
    Domain 1: Risk of bias arising from the randomization process.
    Logic:
      - If Q1.2 = N/PN → High
      - If Q1.2 = Y/PY:
          → If Q1.1 = N/PN → High
          → If Q1.1 = Y/PY/NI → Q1.3: if Y/PY → Some concerns; if N/PN/NI → Low
      - If Q1.2 = NI → Q1.3: if Y/PY → High; if N/PN/NI → Some concerns
    """
    q1_1 = answers.get("Q1.1")
    q1_2 = answers.get("Q1.2")
    q1_3 = answers.get("Q1.3")

    if not all([q1_1, q1_2, q1_3]):
        return "Not Scored"

    if q1_2 in ["No", "Probably No"]:
        return "High"
    elif q1_2 in ["Yes", "Probably Yes"]:
        if q1_1 in ["No", "Probably No"]:
            return "High"
        else:  # q1_1 in ["Yes", "Probably Yes", "No Information"]
            if q1_3 in ["Yes", "Probably Yes"]:
                return "Some concerns"
            else:  # q1_3 in ["No", "Probably No", "No Information"]
                return "Low"
    else:  # q1_2 == "No Information"
        if q1_3 in ["Yes", "Probably Yes"]:
            return "High"
        else:  # q1_3 in ["No", "Probably No", "No Information"]
            return "Some concerns"


def score_deviations_from_intended_interventions(answers: Dict[str, str]) -> str:
    """
    Domain 2: Risk of bias due to deviations from intended interventions.
    Two parts:
      Part A: Blinding & protocol deviations
      Part B: Analysis method (ITT)
    Final = worst of two.
    """
    # Required questions
    required = ["Q2.1", "Q2.2", "Q2.3", "Q2.4", "Q2.5", "Q2.6", "Q2.7"]
    if not all(answers.get(q) is not None for q in required):
        return "Not Scored"

    q2_1 = answers["Q2.1"]
    q2_2 = answers["Q2.2"]
    q2_3 = answers["Q2.3"]
    q2_4 = answers["Q2.4"]
    q2_5 = answers["Q2.5"]
    q2_6 = answers["Q2.6"]
    q2_7 = answers["Q2.7"]

    # Part 2A: Blinding & Deviations
    if q2_1 in ["No", "Probably No"] and q2_2 in ["No", "Probably No"]:
        judgment_2a = "Low"
    else:
        if q2_3 in ["No", "Probably No"]:
            judgment_2a = "Low"
        elif q2_3 == "No Information":
            judgment_2a = "Some concerns"
        elif q2_4 in ["No", "Probably No"]:
            judgment_2a = "Some concerns"
        elif q2_5 in ["Yes", "Probably Yes"]:
            judgment_2a = "Some concerns"
        else:
            judgment_2a = "High"

    # Part 2B: Analysis Method (ITT)
    if q2_6 in ["Yes", "Probably Yes"]:
        judgment_2b = "Low"
    elif q2_7 in ["No", "Probably No"]:
        judgment_2b = "Some concerns"
    else:
        judgment_2b = "High"

    # Final = worst of 2A and 2B
    score_map = {"Low": 1, "Some concerns": 2, "High": 3}
    try:
        max_score = max(score_map[judgment_2a], score_map[judgment_2b])
        return {1: "Low", 2: "Some concerns", 3: "High"}[max_score]
    except KeyError:
        return "Not Scored"


def score_missing_outcome_data(answers: Dict[str, str]) -> str:
    """
    Domain 3: Risk of bias due to missing outcome data.
    Logic:
      - If Q3.1 = Y/PY → Low
      - If Q3.1 = N/PN/NI → Q3.2:
          → If Q3.2 = Y/PY → Low
          → If Q3.2 = N/PN → Q3.3:
              → If Q3.3 = N/PN → Low
              → If Q3.3 = Y/PY/NI → Q3.4:
                  → If Q3.4 = N/PN → Some concerns
                  → If Q3.4 = Y/PY/NI → High
    """
    q3_1 = answers.get("Q3.1")
    q3_2 = answers.get("Q3.2")
    q3_3 = answers.get("Q3.3")
    q3_4 = answers.get("Q3.4")

    if not all([q3_1, q3_2, q3_3, q3_4]):
        return "Not Scored"

    if q3_1 in ["Yes", "Probably Yes"]:
        return "Low"
    else:  # q3_1 in ["No", "Probably No", "No Information"]
        if q3_2 in ["Yes", "Probably Yes"]:
            return "Low"
        else:  # q3_2 in ["No", "Probably No", "No Information"]
            if q3_3 in ["No", "Probably No"]:
                return "Low"
            else:  # q3_3 in ["Yes", "Probably Yes", "No Information"]
                if q3_4 in ["No", "Probably No"]:
                    return "Some concerns"
                else:  # q3_4 in ["Yes", "Probably Yes", "No Information"]
                    return "High"


def score_measurement_of_outcome(answers: Dict[str, str]) -> str:
    """
    Domain 4: Risk of bias in measurement of the outcome.
    Logic:
      - If Q4.1 or Q4.2 = Y/PY → High
      - Else:
          → If Q4.2 = NI → Q4.3:
              → If Q4.3 = N/PN → Some concerns
              → If Q4.3 = Y/PY/NI → Q4.4:
                  → If Q4.4 = N/PN → Some concerns
                  → If Q4.4 = Y/PY/NI → Q4.5:
                      → If Q4.5 = N/PN → Some concerns
                      → If Q4.5 = Y/PY/NI → High
          → If Q4.2 = N/PN → Q4.3:
              → If Q4.3 = N/PN → Low
              → If Q4.3 = Y/PY/NI → Q4.4:
                  → If Q4.4 = N/PN → Low
                  → If Q4.4 = Y/PY/NI → Q4.5:
                      → If Q4.5 = N/PN → Some concerns
                      → If Q4.5 = Y/PY/NI → High
    """
    q4_1 = answers.get("Q4.1")
    q4_2 = answers.get("Q4.2")
    q4_3 = answers.get("Q4.3")
    q4_4 = answers.get("Q4.4")
    q4_5 = answers.get("Q4.5")

    if not all([q4_1, q4_2, q4_3, q4_4, q4_5]):
        return "Not Scored"

    if q4_1 in ["Yes", "Probably Yes"] or q4_2 in ["Yes", "Probably Yes"]:
        return "High"
    else:
        if q4_2 == "No Information":
            if q4_3 in ["No", "Probably No"]:
                return "Some concerns"
            else:
                if q4_4 in ["No", "Probably No"]:
                    return "Some concerns"
                else:
                    if q4_5 in ["No", "Probably No"]:
                        return "Some concerns"
                    else:
                        return "High"
        elif q4_2 in ["No", "Probably No"]:
            if q4_3 in ["No", "Probably No"]:
                return "Low"
            else:
                if q4_4 in ["No", "Probably No"]:
                    return "Low"
                else:
                    if q4_5 in ["No", "Probably No"]:
                        return "Some concerns"
                    else:
                        return "High"
        else:
            return "Not Scored"


def score_selection_of_reported_result(answers: Dict[str, str]) -> str:
    """
    Domain 5: Risk of bias in selection of the reported result.
    Logic:
      - If Q5.2 or Q5.3 = Y/PY → High
      - If Q5.2 or Q5.3 = NI (but neither = Y/PY) → Some concerns
      - If both Q5.2 and Q5.3 = N/PN → Q5.1:
          → If Q5.1 = Y/PY → Low
          → If Q5.1 = N/PN/NI → Some concerns
    """
    q5_1 = answers.get("Q5.1")
    q5_2 = answers.get("Q5.2")
    q5_3 = answers.get("Q5.3")

    if not all([q5_1, q5_2, q5_3]):
        return "Not Scored"

    if q5_2 in ["Yes", "Probably Yes"] or q5_3 in ["Yes", "Probably Yes"]:
        return "High"
    elif q5_2 == "No Information" or q5_3 == "No Information":
        return "Some concerns"
    else:
        if q5_1 in ["Yes", "Probably Yes"]:
            return "Low"
        else:
            return "Some concerns"


def score_rob2_domain_scores(signaling_answers: Dict[str, str]) -> Dict[str, str]:
    """
    Apply RoB 2 rules to return domain-level judgments.
    Returns 'Not Scored' for any domain that fails.
    """
    return {
        "Randomization Process": score_randomization_process(signaling_answers),
        "Deviations from Intended Interventions": score_deviations_from_intended_interventions(signaling_answers),
        "Missing Outcome Data": score_missing_outcome_data(signaling_answers),
        "Measurement of the Outcome": score_measurement_of_outcome(signaling_answers),
        "Selection of the Reported Result": score_selection_of_reported_result(signaling_answers)
    }


def get_overall_rob2_judgment(domain_scores: Dict[str, str]) -> str:
    """
    Return the overall risk of bias judgment.
    If any domain is 'Not Scored', return 'Not Scored'.
    Otherwise, take worst-case judgment.
    """
    score_map = {
        "Low": 1,
        "Some concerns": 2,
        "High": 3,
        "Not Scored": 4
    }

    if "Not Scored" in domain_scores.values():
        return "Not Scored"

    try:
        max_score = max(score_map[score] for score in domain_scores.values())
        return {1: "Low", 2: "Some concerns", 3: "High"}[max_score]
    except Exception as e:
        print(f"Error computing overall RoB 2 judgment: {e}")
        return "Not Scored"