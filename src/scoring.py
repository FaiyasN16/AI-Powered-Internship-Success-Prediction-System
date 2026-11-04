"""
Profile Scoring and Domain Readiness Module.

Implements transparent, documented heuristics for:
  - Overall Profile Score (0 - 100)
  - Domain Readiness Score (0 - 100)
  - Domain Suitability Classification

Notice: These heuristic scores are calculated via rule-based rubrics to provide
holistic academic guidance. They are distinct from the statistical Random Forest
selection probability estimate.
"""

from typing import Dict, Any, Tuple


def calculate_overall_profile_score(profile: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Calculate an Overall Profile Score (0 - 100) based on weighted rubric components.

    Rubric Breakdown (Total = 100 points):
      - Academic Performance (CGPA): 25 pts
      - Project Volume (Projects): 20 pts
      - Project Depth (Complexity): 15 pts
      - Practical Experience (Internship): 15 pts
      - Industry Credentials (Certifications): 10 pts
      - Technical Breadth (Technical Skills): 10 pts
      - Language Repertoire (Programming Languages): 5 pts

    Parameters
    ----------
    profile : dict
        Validated student profile dictionary.

    Returns
    -------
    (float, dict)
        Overall score (0 - 100) and component breakdown.
    """
    cgpa = float(profile.get("CGPA", 0.0))
    projects = int(profile.get("Number of Relevant Projects", 0))
    complexity = int(profile.get("Average Project Complexity", 1))
    has_internship = str(profile.get("Previous Internship Experience", "No")).strip().lower() == "yes"
    certs = int(profile.get("Relevant Certifications", 0))
    langs = int(profile.get("Number of Programming Languages", 0))
    skills = int(profile.get("Number of Technical Skills", 0))

    # 1. CGPA points (0 - 25)
    # Scaled from 5.0 base (CGPA below 5 gives 0-5, 10 gives 25)
    cgpa_score = min(max((cgpa / 10.0) * 25.0, 0.0), 25.0)

    # 2. Projects score (0 - 20) -> Target: 4+ projects
    projects_score = min((projects / 4.0) * 20.0, 20.0)

    # 3. Project Complexity score (0 - 15) -> Scale 1 to 5
    complexity_score = ((complexity - 1) / 4.0) * 15.0

    # 4. Previous Internship (0 or 15)
    internship_score = 15.0 if has_internship else 0.0

    # 5. Certifications score (0 - 10) -> Target: 2+ certs
    certs_score = min((certs / 2.0) * 10.0, 10.0)

    # 6. Technical Skills score (0 - 10) -> Target: 8+ skills
    skills_score = min((skills / 8.0) * 10.0, 10.0)

    # 7. Programming Languages score (0 - 5) -> Target: 3+ languages
    langs_score = min((langs / 3.0) * 5.0, 5.0)

    total_score = round(
        cgpa_score + projects_score + complexity_score + internship_score + certs_score + skills_score + langs_score,
        1
    )
    total_score = float(min(max(total_score, 0.0), 100.0))

    breakdown = {
        "Academic Standing (CGPA)": round(cgpa_score, 1),
        "Project Portfolio": round(projects_score, 1),
        "Project Architecture & Complexity": round(complexity_score, 1),
        "Prior Internship Experience": round(internship_score, 1),
        "Industry Certifications": round(certs_score, 1),
        "Technical Skills Mastery": round(skills_score, 1),
        "Programming Languages": round(langs_score, 1),
    }

    return total_score, breakdown


def calculate_domain_readiness(profile: Dict[str, Any]) -> Tuple[float, str, str]:
    """
    Calculate a Domain Readiness Score (0 - 100) and Suitability Category tailored
    to the specific internship domain applied.

    Parameters
    ----------
    profile : dict
        Validated student profile dictionary.

    Returns
    -------
    (float, str, str)
        Tuple of (readiness_score, suitability_level, explanation).
    """
    domain = str(profile.get("Internship Domain Applied", "Software Engineering")).strip()
    cgpa = float(profile.get("CGPA", 0.0))
    projects = int(profile.get("Number of Relevant Projects", 0))
    complexity = int(profile.get("Average Project Complexity", 1))
    certs = int(profile.get("Relevant Certifications", 0))
    langs = int(profile.get("Number of Programming Languages", 0))
    skills = int(profile.get("Number of Technical Skills", 0))

    score = 0.0

    if domain in ["Artificial Intelligence and Machine Learning", "Data Science and Analytics"]:
        # AI/ML & Data Science emphasize project complexity, math/skills, and academic rigor
        comp_pts = min(((complexity - 1) / 4.0) * 35.0, 35.0)
        skills_pts = min((skills / 8.0) * 25.0, 25.0)
        cgpa_pts = min((cgpa / 10.0) * 20.0, 20.0)
        proj_pts = min((projects / 3.0) * 20.0, 20.0)
        score = comp_pts + skills_pts + cgpa_pts + proj_pts

    elif domain in ["Cloud Computing", "Cybersecurity"]:
        # Cloud & Cyber emphasize certifications, architecture complexity, and hands-on tools
        cert_pts = min((certs / 3.0) * 35.0, 35.0)
        comp_pts = min(((complexity - 1) / 4.0) * 25.0, 25.0)
        skills_pts = min((skills / 7.0) * 20.0, 20.0)
        proj_pts = min((projects / 3.0) * 20.0, 20.0)
        score = cert_pts + comp_pts + skills_pts + proj_pts

    elif domain in ["Web Development", "Mobile App Development"]:
        # Web & Mobile prioritize portfolio depth, practical frameworks, and languages
        proj_pts = min((projects / 4.0) * 35.0, 35.0)
        comp_pts = min(((complexity - 1) / 4.0) * 25.0, 25.0)
        skills_pts = min((skills / 6.0) * 25.0, 25.0)
        lang_pts = min((langs / 3.0) * 15.0, 15.0)
        score = proj_pts + comp_pts + skills_pts + lang_pts

    else:  # Software Engineering
        proj_pts = min((projects / 4.0) * 25.0, 25.0)
        comp_pts = min(((complexity - 1) / 4.0) * 25.0, 25.0)
        lang_pts = min((langs / 4.0) * 25.0, 25.0)
        skills_pts = min((skills / 7.0) * 25.0, 25.0)
        score = proj_pts + comp_pts + lang_pts + skills_pts

    score = round(min(max(score, 0.0), 100.0), 1)

    if score >= 80.0:
        suitability = "High Suitability"
        summary = (
            f"Strong technical alignment for {domain}. Profile exhibits the core technical depth "
            "and project evidence expected by internship evaluators."
        )
    elif score >= 60.0:
        suitability = "Moderate Suitability"
        summary = (
            f"Competitive baseline for {domain}. Demonstrates foundational competencies with potential "
            "to strengthen specialized domain portfolio items."
        )
    elif score >= 40.0:
        suitability = "Developing Suitability"
        summary = (
            f"Early-stage profile for {domain}. Meets basic prerequisites but requires targeted hands-on "
            "projects and relevant skills to reach competitive readiness."
        )
    else:
        suitability = "Foundational Preparation"
        summary = (
            f"Introductory profile for {domain}. Recommended to build core fundamentals and complete guided "
            "domain-specific projects before applying to competitive openings."
        )

    return score, suitability, summary

