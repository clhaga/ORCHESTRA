from typing import Dict, Any


def score_domain_1_confounding(answers: Dict[str, str]) -> str:
    """
    Domain 1: Risk of bias due to confounding.
    Logic based on ROBINS-I V2 flowchart.
    Normalizes SY/SN to Y/N for consistent logic.
    """
    q1_1 = answers.get("Q1.1")
    q1_2 = answers.get("Q1.2")
    q1_3 = answers.get("Q1.3")
    q1_4 = answers.get("Q1.4")

    if not all([q1_1, q1_2, q1_3, q1_4]):
        return "Not Scored"

    def normalize(q):
        if q == "SY":
            return "Y"
        elif q == "SN":
            return "N"
        else:
            return q

    q1_1 = normalize(q1_1)
    q1_2 = normalize(q1_2)
    q1_3 = normalize(q1_3)
    q1_4 = normalize(q1_4)

    # Original logic follows
    if q1_1 in ["Y", "PY"]:
        if q1_3 in ["N", "PN", "NI"]:
            if q1_2 in ["Y", "PY"]:
                if q1_4 in ["N", "PN"]:
                    return "Low"
                else:  # Y/PY
                    return "Serious"
            elif q1_2 in ["WN"]:
                if q1_4 in ["N", "PN"]:
                    return "Moderate"
                else:
                    return "Serious"
            else:  # SN/NI (normalized to N/NI)
                return "Serious"
        elif q1_3 in ["Y", "PY"]:
            if q1_4 in ["Y", "PY"]:
                return "Critical"
            elif q1_4 in ["N", "PN"]:
                if q1_2 in ["Y", "PY"]:
                    return "Serious"
                else:
                    return "Critical"
            else:  # NI
                return "Critical"

    elif q1_1 in ["WN"]:
        if q1_3 in ["Y", "PY"]:
            return "Critical"
        elif q1_3 in ["N", "PN", "NI"]:
            if q1_2 in ["Y", "PY", "WN"]:
                if q1_4 in ["N", "PN"]:
                    return "Moderate"
                else:
                    return "Serious"
            else:
                return "Serious"

    elif q1_1 in ["SN", "NI"]:
        if q1_4 in ["Y", "PY"]:
            return "Critical"
        elif q1_4 in ["N", "PN"]:
            return "Serious"
        else:  # NI
            if q1_2 in ["Y", "PY"]:
                return "Serious"
            else:
                return "Critical"

    return "Not Scored"

def score_domain_2_classification(answers: Dict[str, str]) -> str:
    """
    Domain 2: Risk of bias in classification of interventions.
    Follows ROBINS-I V2 official logic.
    """
    q2_1 = answers.get("Q2.1")
    q2_2 = answers.get("Q2.2")
    q2_3 = answers.get("Q2.3")
    q2_4 = answers.get("Q2.4")
    q2_5 = answers.get("Q2.5")

    if not all([q2_1, q2_2, q2_3, q2_4, q2_5]):
        return "Not Scored"

    # === Q2.1 = N / PN ===
    if q2_1 in ["N", "PN"]:
        # Go to Q2.3
        if q2_3 in ["Y", "PY"]:
            # Go to Q2.4
            if q2_4 == "SY":
                return "Serious"
            elif q2_4 in ["WY", "NI"]:
                # Go to Q2.5
                if q2_5 in ["Y", "PY"]:
                    return "Low"
                elif q2_5 == "SN":
                    return "Serious"
                else:  # WN / NI
                    return "Moderate"
        
        elif q2_3 in ["N", "PN", "NI"]:
            # Go to Q2.4
            if q2_4 == "SY":
                return "Critical"
            elif q2_4 in ["WY", "NI"]:
                # Go to Q2.5
                if q2_5 in ["Y", "PY"]:
                    return "Moderate"
                else:  # WN / SN / NI
                    return "Serious"
            elif q2_4 in ["N", "PN"]:
                # Go to Q2.5
                if q2_5 in ["Y", "PY"]:
                    return "Moderate"
                else:  # SN / WN / NI
                    return "Serious"

    # === Q2.1 = Y / PY / NI ===
    elif q2_1 in ["Y", "PY", "NI"]:
        # Go to Q2.2
        if q2_2 == "SY":
            return "Critical"
        elif q2_2 in ["Y", "PY"]:
            return "Critical"
        elif q2_2 in ["N", "PN", "WY", "NI"]:
            # Go to Q2.3 (this was missing!)
            if q2_3 in ["Y", "PY"]:
                # Go to Q2.4
                if q2_4 == "SY":
                    return "Serious"
                elif q2_4 in ["WY", "NI"]:
                    # Go to Q2.5
                    if q2_5 in ["Y", "PY"]:
                        return "Low"
                    elif q2_5 == "SN":
                        return "Serious"
                    else:  # WN / NI
                        return "Moderate"
                elif q2_4 in ["N", "PN"]:
                    # Go to Q2.5
                    if q2_5 in ["Y", "PY"]:
                        return "Low"
                    elif q2_5 == "SN":
                        return "Serious"
                    else:  # WN / NI
                        return "Moderate"
            elif q2_3 in ["N", "PN", "NI"]:
                # Go to Q2.4
                if q2_4 == "SY":
                    return "Critical"
                elif q2_4 in ["WY", "NI"]:
                    # Go to Q2.5
                    if q2_5 in ["Y", "PY"]:
                        return "Moderate"
                    else:  # WN / SN / NI
                        return "Serious"
                elif q2_4 in ["N", "PN"]:
                    # Go to Q2.5
                    if q2_5 in ["Y", "PY"]:
                        return "Moderate"
                    else:  # SN / WN / NI
                        return "Serious"

    return "Not Scored"



# def score_domain_2_classification(answers: Dict[str, str]) -> str:
#     """
#     Domain 2: Risk of bias in classification of interventions.
#     Follows ROBINS-I V2 official logic.
#     """
#     q2_1 = answers.get("Q2.1")
#     q2_2 = answers.get("Q2.2")
#     q2_3 = answers.get("Q2.3")
#     q2_4 = answers.get("Q2.4")
#     q2_5 = answers.get("Q2.5")

#     if not all([q2_1, q2_2, q2_3, q2_4, q2_5]):
#         return "Not Scored"

#     # === Q2.1 = N / PN ===
#     if q2_1 in ["N", "PN"]:
#         if q2_3 in ["Y", "PY"]:
#             if q2_4 == "SY":
#                 return "Serious"
#             elif q2_4 in ["WY", "NI"]:
#                 if q2_5 in ["Y", "PY"]:
#                     return "Low"
#                 elif q2_5 == "SN":
#                     return "Serious"
#                 else:  # WN / NI
#                     return "Moderate"
#             # Note: Q2.4 = N/PN not handled in this branch per official logic
#         elif q2_3 in ["N", "PN", "NI"]:
#             if q2_4 == "SY":
#                 return "Critical"
#             elif q2_4 in ["WY", "NI"]:
#                 if q2_5 in ["Y", "PY"]:
#                     return "Moderate"
#                 else:  # WN / SN / NI
#                     return "Serious"
#             elif q2_4 in ["N", "PN"]:
#                 if q2_5 in ["Y", "PY"]:
#                     return "Moderate"
#                 else:  # WN / SN / NI
#                     return "Serious"

#     # === Q2.1 = Y / PY / NI ===
#     elif q2_1 in ["Y", "PY", "NI"]:
#         if q2_2 == "SY":
#             return "Critical"
#         elif q2_2 in ["N", "PN", "WY", "NI"]:
#             if q2_4 == "SY":
#                 return "Critical"
#             else:  # N / PN / WY / NI
#                 return "Serious"
#         else:
#             return "Serious"  

#     return "Not Scored"  


def score_domain_3_selection(answers: Dict[str, str]) -> str:
    """
    Domain 3: Risk of bias in selection of participants into the study.
    Three parts: A (immortal time), B (prevalent user), C (other selection).
    Final judgment uses Q3.8–Q3.10 for serious risks.
    """
    required = [f"Q3.{i}" for i in range(1, 11)]
    if not all(answers.get(q) is not None for q in required):
        return "Not Scored"

    # Part A
    if answers["Q3.1"] in ["N", "PN"]:
        part_a = "Low"
    elif answers["Q3.1"] == "NI":
        part_a = "Moderate"
    else:  # Y/PY
        if answers["Q3.2"] in ["N", "PN"]:
            part_a = "Low"
        elif answers["Q3.2"] in ["Y", "PY"]:
            part_a = "Serious"
        else:
            part_a = "Moderate"

    # Part B
    if answers["Q3.3"] in ["Y", "PY"]:
        part_b = "Low"
    elif answers["Q3.3"] == "NI":
        part_b = "Moderate"
    else:  # N/PN
        if answers["Q3.4"] in ["Y", "PY", "NI"]:
            part_b = "Moderate"
        else:
            part_b = "Serious"

    # Part C
    if answers["Q3.5"] in ["N", "PN"]:
        part_c = "Low"
    elif answers["Q3.5"] == "NI":
        part_c = "Moderate"
    else:  # Y/PY
        if answers["Q3.6"] in ["N", "PN"]:
            part_c = "Low"
        elif answers["Q3.6"] == "NI":
            part_c = "Moderate"
        else:  # Y/PY
            if answers["Q3.7"] in ["N", "PN", "NI"]:
                part_c = "Moderate"
            else:
                part_c = "Serious"

    max_part = max([part_a, part_b, part_c], key=["Low", "Moderate", "Serious", "Critical"].index)
    if max_part == "Low":
        return "Low"
    elif max_part == "Moderate":
        return "Moderate"
    else:
        if answers["Q3.8"] in ["Y", "PY"]:
            return "Moderate"
        elif answers["Q3.9"] in ["Y", "PY"]:
            return "Moderate"
        else:
            if answers["Q3.10"] in ["Y", "PY"]:
                return "Critical"
            else:
                return "Serious"

    return "Not Scored"


def score_domain_4_deviations(answers: Dict[str, str]) -> str:
    """
    Domain 4: Risk of bias due to deviations from intended interventions.
    Two parts: A (non-adherence), B (outcome-driven discontinuation).
    """
    q4_1 = answers.get("Q4.1")
    q4_2 = answers.get("Q4.2")
    q4_3 = answers.get("Q4.3")
    q4_4 = answers.get("Q4.4")
    q4_5 = answers.get("Q4.5")

    if not all([q4_1, q4_2, q4_3, q4_4, q4_5]):
        return "Not Scored"

    # Part A
    if q4_1 in ["N", "PN"]:
        part_a = "Low"
    elif q4_1 == "NI":
        part_a = "Moderate"
    else:  # Y/PY
        # Check if BOTH 4.2 AND 4.3 are N/PN/NI
        if q4_2 in ["N", "PN", "NI"] and q4_3 in ["N", "PN", "NI"]:
            part_a = "Low"
        # Check if EITHER 4.2 OR 4.3 is Y/PY
        elif q4_2 in ["Y", "PY"] or q4_3 in ["Y", "PY"]:
            # Go to 4.4
            if q4_4 in ["N", "PN"]:
                part_a = "Moderate"
            else:  # Y/PY/NI
                part_a = "Serious"
        else:
            # This should not happen if the logic is followed correctly
            part_a = "Not Scored"

    # Part B
    if q4_5 in ["Y", "PY"]:
        part_b = "Low"
    elif q4_5 in ["WN", "NI"]:
        part_b = "Moderate"
    else:  # SN
        part_b = "Serious"

    if part_a == "Serious" and part_b == "Serious":
        return "Critical"
    elif part_a == "Serious" or part_b == "Serious":
        return "Serious"
    elif part_a == "Moderate" or part_b == "Moderate":
        return "Moderate"
    else:
        return "Low"


def score_domain_5_missing_data(answers: Dict[str, str]) -> str:
    """
    Domain 5: Risk of bias due to missing data.
    Normalizes WY → Y, SN → N for consistent logic.
    """
    required = [f"Q5.{i}" for i in range(1, 12)]
    if not all(answers.get(q) is not None for q in required):
        return "Not Scored"

    # Normalize WY → Y, SN → N for logic consistency
    def normalize(q):
        if q == "WY":
            return "Y"
        elif q == "SN":
            return "N"
        else:
            return q

    q5_1 = normalize(answers["Q5.1"])
    q5_2 = normalize(answers["Q5.2"])
    q5_3 = normalize(answers["Q5.3"])
    q5_4 = normalize(answers["Q5.4"])
    q5_5 = normalize(answers["Q5.5"])
    q5_6 = normalize(answers["Q5.6"])
    q5_7 = normalize(answers["Q5.7"])
    q5_8 = normalize(answers["Q5.8"])
    q5_9 = normalize(answers["Q5.9"])
    q5_10 = normalize(answers["Q5.10"])
    q5_11 = normalize(answers["Q5.11"])

    # Path 1: All Q5.1, Q5.2, Q5.3 = Y/PY → Low
    if all(q in ["Y", "PY"] for q in [q5_1, q5_2, q5_3]):
        return "Low"

    # Path 2: Any of Q5.1, Q5.2, Q5.3 != Y/PY → go to Q5.4
    if q5_4 in ["Y", "PY", "NI"]:  # Complete case
        if q5_5 in ["N", "PN"]:
            return "Low"
        elif q5_6 == "Y":  # Y includes normalized WY
            return "Low" if q5_11 in ["Y", "PY"] else "Moderate"
        elif q5_6 in ["WY", "NI"]:  
            return "Moderate" if q5_11 in ["Y", "PY"] else "Serious"
        elif q5_6 == "SN":  # Normalized to N
            return "Serious" if q5_11 in ["Y", "PY"] else "Critical"
        else:
            return "Not Scored"  # Should not reach here

    else:  # q5_4 in ["N", "PN"] → Imputation or other method
        if q5_7 in ["Y", "PY"]:  # Imputed
            if q5_8 in ["N", "PN", "NI"]:
                return "Serious"
            elif q5_9 in ["Y", "PY"]:
                return "Low"
            elif q5_9 in ["WN", "NI"]:
                return "Moderate" if q5_11 in ["Y", "PY"] else "Serious"
            elif q5_9 == "SN":
                return "Serious" if q5_11 in ["Y", "PY"] else "Critical"
        else:  # Alternative method (q5_7 in ["N", "PN", "NI"])
            if q5_10 in ["Y", "PY"]:
                return "Low"
            elif q5_10 in ["WN", "NI"]:
                return "Moderate" if q5_11 in ["Y", "PY"] else "Serious"
            elif q5_10 == "SN":
                return "Serious" if q5_11 in ["Y", "PY"] else "Critical"

    return "Not Scored"


def score_domain_6_measurement(answers: Dict[str, str]) -> str:
    """
    Domain 6: Risk of bias arising from measurement of the outcome.
    """
    q6_1 = answers.get("Q6.1")
    q6_2 = answers.get("Q6.2")
    q6_3 = answers.get("Q6.3")

    if not all([q6_1, q6_2, q6_3]):
        return "Not Scored"

    if q6_1 in ["Y", "PY"]:
        return "Serious"
    elif q6_1 in ["N", "PN"]:
        if q6_2 in ["N", "PN"]:
            return "Low"
        elif q6_3 in ["N", "PN"]:
            return "Low"
        elif q6_3 in ["WY", "NI"]:
            return "Moderate"
        else:  # SY
            return "Serious"
    else:  # q6_1 == "NI"
        if q6_2 in ["N", "PN"]:
            return "Moderate"
        else:
            if q6_3 in ["WY", "N", "PN", "NI"]:
                return "Moderate"
            else:  # SY
                return "Serious"

    return "Not Scored"


def score_domain_7_selection(answers: Dict[str, str]) -> str:
    """
    Domain 7: Risk of bias in selection of the reported result.
    """
    q7_1 = answers.get("Q7.1")
    q7_2 = answers.get("Q7.2")
    q7_3 = answers.get("Q7.3")
    q7_4 = answers.get("Q7.4")

    if not all([q7_1, q7_2, q7_3, q7_4]):
        return "Not Scored"

    if q7_1 in ["Y", "PY"]:
        return "Low"
    if all(q in ["N", "PN"] for q in [q7_2, q7_3, q7_4]):
        return "Low"
    if any(q == "NI" for q in [q7_2, q7_3, q7_4]) and not any(q in ["Y", "PY"] for q in [q7_2, q7_3, q7_4]):
        return "Moderate"
    num_y_py = sum(1 for q in [q7_2, q7_3, q7_4] if q in ["Y", "PY"])
    if num_y_py >= 2:
        return "Critical"
    return "Serious"


def score_robins_i_domain_scores(signaling_answers: Dict[str, str]) -> Dict[str, str]:
    """
    Apply ROBINS-I V2 rules to return domain-level judgments.
    Returns 'Not Scored' for any domain that fails.
    """
    return {
        "Confounding": score_domain_1_confounding(signaling_answers),
        "Classification of Interventions": score_domain_2_classification(signaling_answers),
        "Selection of Participants into the Study": score_domain_3_selection(signaling_answers),
        "Deviations from Intended Interventions": score_domain_4_deviations(signaling_answers),
        "Missing Outcome Data": score_domain_5_missing_data(signaling_answers),
        "Measurement of the Outcome": score_domain_6_measurement(signaling_answers),
        "Selection of the Reported Result": score_domain_7_selection(signaling_answers)
    }


def get_overall_robins_i_judgment(domain_scores: Dict[str, str]) -> str:
    """
    Return overall risk of bias judgment.
    Ignores 'Not Scored' domains and returns the worst-case judgment among valid domains.
    If ALL domains are 'Not Scored', returns 'Not Scored'.
    """
    score_map = {
        "Low": 1,
        "Moderate": 2,
        "Serious": 3,
        "Critical": 4,
    }

    # Extract only valid risk judgments (exclude "Not Scored")
    valid_scores = [score for score in domain_scores.values() if score in score_map]

    # If no domains were scored, return "Not Scored"
    if not valid_scores:
        return "Not Scored"

    # Return the worst (highest) risk level among valid domains
    max_score = max(score_map[score] for score in valid_scores)
    return {1: "Low", 2: "Moderate", 3: "Serious", 4: "Critical"}[max_score]