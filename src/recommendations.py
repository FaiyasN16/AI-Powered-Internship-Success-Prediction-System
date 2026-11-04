"""
Recommendation and Profile Comparison Engine.

Generates personalized strengths, targeted improvement areas, prioritized
action items, a 4-phase short-term roadmap, and candidate vs. ideal target
profile benchmark comparisons.
"""

from typing import Dict, Any, List
import pandas as pd


# Reference Ideal Benchmarks (Configurable target reference goals for student guidance)
IDEAL_PROFILE_TARGETS = {
    "CGPA": 8.5,
    "Number of Relevant Projects": 4,
    "Average Project Complexity": 4,  # Scale 1-5 (Advanced)
    "Previous Internship Experience": "Yes",
    "Relevant Certifications": 2,
    "Number of Programming Languages": 3,
    "Number of Technical Skills": 8,
}


def generate_recommendations(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate domain-aware, respectful, actionable recommendations based on profile metrics.

    Parameters
    ----------
    profile : dict
        Validated student profile dictionary.

    Returns
    -------
    dict
        Dictionary containing strengths, improvement areas, prioritized actions,
        and a 4-phase improvement roadmap.
    """
    domain = str(profile.get("Internship Domain Applied", "Software Engineering")).strip()
    cgpa = float(profile.get("CGPA", 0.0))
    projects = int(profile.get("Number of Relevant Projects", 0))
    complexity = int(profile.get("Average Project Complexity", 1))
    has_internship = str(profile.get("Previous Internship Experience", "No")).strip().lower() == "yes"
    certs = int(profile.get("Relevant Certifications", 0))
    langs = int(profile.get("Number of Programming Languages", 0))
    skills = int(profile.get("Number of Technical Skills", 0))

    strengths: List[str] = []
    improvement_areas: List[str] = []
    action_items: List[str] = []

    # 1. Evaluate Academic Standing (CGPA)
    if cgpa >= 8.5:
        strengths.append(f"Strong Academic Excellence: CGPA of {cgpa:.2f}/10 reflects solid conceptual mastery.")
    elif cgpa >= 7.5:
        strengths.append(f"Healthy Academic Foundation: CGPA of {cgpa:.2f}/10 meets screening thresholds.")
    else:
        improvement_areas.append(
            f"Academic Rigor: Current CGPA ({cgpa:.2f}) is below standard shortlist benchmarks (7.5+)."
        )
        action_items.append(
            "Strengthen core subject grades (Data Structures, Algorithms, OS) and highlight high scores in major domain courses."
        )

    # 2. Evaluate Projects
    if projects >= 4:
        strengths.append(f"Robust Project Portfolio: Completed {projects} projects demonstrating practical initiative.")
    elif projects >= 2:
        strengths.append(f"Practical Exposure: {projects} projects provide a good starting baseline.")
    else:
        improvement_areas.append(
            f"Limited Practical Projects: Only {projects} project(s) recorded. Recruiters look for 3+ demonstrated builds."
        )
        action_items.append(
            f"Build and publish 2 domain-focused projects tailored to {domain} with live demo URLs and clean GitHub repositories."
        )

    # 3. Project Complexity
    if complexity >= 4:
        strengths.append(f"High Architectural Complexity (Level {complexity}/5): Evidence of full-lifecycle implementation.")
    elif complexity == 3:
        pass  # Neutral
    else:
        improvement_areas.append(
            f"Project Architecture Depth: Project complexity ({complexity}/5) suggests beginner-to-intermediate scope."
        )
        action_items.append(
            "Elevate an existing project by integrating production features: database indexing, authentication, CI/CD pipelines, and unit tests."
        )

    # 4. Previous Internship
    if has_internship:
        strengths.append("Prior Professional Experience: Industry internship experience provides a significant competitive edge.")
    else:
        improvement_areas.append("Zero Formal Internship Experience: Lack of prior workplace technical exposure.")
        action_items.append(
            "Bridge the industry experience gap by contributing to open-source repositories, participating in hackathons, or completing guided simulations."
        )

    # 5. Certifications
    if certs >= 2:
        strengths.append(f"Verified Industry Credentials: {certs} relevant certifications validate skill standards.")
    else:
        improvement_areas.append(f"Credential Validation: Only {certs} certification(s) logged.")
        if domain in ["Cloud Computing", "Cybersecurity"]:
            action_items.append(
                f"Earn a vendor certification in {domain} (e.g., AWS Cloud Practitioner / Solutions Architect or CompTIA Security+)."
            )
        else:
            action_items.append(
                f"Obtain a recognized skill certificate in {domain} (e.g., DeepLearning.AI, Meta Frontend/Backend, or Google Data Analytics)."
            )

    # 6. Technical Skills & Languages
    if skills >= 8:
        strengths.append(f"Comprehensive Technical Repertoire: {skills} verified technical tools and skills.")
    elif skills < 5:
        improvement_areas.append(f"Skill Breadth: Currently cataloging {skills} technical skills. Aim for 7-10 domain tools.")
        action_items.append(
            f"Expand domain-relevant stack for {domain} (e.g., Docker, REST APIs, Git, PostgreSQL, automated testing frameworks)."
        )

    if langs >= 3:
        strengths.append(f"Polyglot Adaptability: Familiarity with {langs} programming languages.")
    elif langs == 1:
        action_items.append(
            "Focus deeply on mastering one primary language for problem-solving (e.g., Python/Java/C++) and one scripting/domain language."
        )

    # Ensure at least 3 to 5 action items
    fallback_best_practices = [
        "Prepare a portfolio website showcasing architecture diagrams, problem statements, and quantitative project outcomes.",
        "Conduct timed mock technical interviews focusing on system design, data structures, and domain-specific questions.",
        f"Contribute to open-source software repositories relevant to {domain} to demonstrate collaborative git workflows.",
        "Publish a technical writeup or case study summarizing architectural lessons learned from your capstone project.",
        "Network with alumni and engineering professionals in your target domain for informational advice and referral guidance.",
    ]
    for practice in fallback_best_practices:
        if len(action_items) >= 4:
            break
        if practice not in action_items:
            action_items.append(practice)

    # Domain-specific roadmap
    roadmap = [
        {
            "phase": "Phase 1 (Weeks 1-3): Fundamentals & Core Stack",
            "focus": f"Strengthen computer science fundamentals and master the primary programming language for {domain}.",
            "deliverable": "Complete 30 core algorithmic problems and establish clean coding habits.",
        },
        {
            "phase": "Phase 2 (Weeks 4-7): Capstone Production Project",
            "focus": f"Design and implement a high-complexity full-stack / end-to-end {domain} application.",
            "deliverable": "Deployed project with GitHub documentation, clean architecture, and test suites.",
        },
        {
            "phase": "Phase 3 (Weeks 8-10): Industry Credentials & Portfolio",
            "focus": "Obtain an accredited domain certification and polish technical portfolio and resume.",
            "deliverable": "Updated resume featuring quantitative metrics and certified credential badges.",
        },
        {
            "phase": "Phase 4 (Weeks 11-12): Application & Interview Drills",
            "focus": "Targeted internship applications, cold networking with alumni, and behavioral/technical interview mocks.",
            "deliverable": "Submit 20 tailored applications with customized cover notes.",
        },
    ]

    return {
        "strengths": strengths if strengths else ["Profile meets introductory criteria."],
        "improvement_areas": improvement_areas if improvement_areas else ["Maintain steady project updates and technical practice."],
        "action_items": action_items[:5],
        "roadmap": roadmap,
    }


def generate_ideal_profile_comparison(profile: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a candidate profile vs. ideal target profile benchmark comparison table.

    Parameters
    ----------
    profile : dict
        Validated student profile dictionary.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    rows = []

    # 1. CGPA
    student_cgpa = float(profile.get("CGPA", 0.0))
    target_cgpa = IDEAL_PROFILE_TARGETS["CGPA"]
    cgpa_diff = student_cgpa - target_cgpa
    rows.append({
        "Profile Feature": "CGPA (0 - 10)",
        "Student Value": f"{student_cgpa:.2f}",
        "Target Benchmark": f"{target_cgpa:.2f}",
        "Status / Gap": f"{cgpa_diff:+.2f}" if cgpa_diff < 0 else "Benchmark Met (0.0)",
        "Suggested Action": "Strengthen academic focus in core semester exams." if cgpa_diff < 0 else "Maintain academic consistency.",
    })

    # 2. Number of Relevant Projects
    student_proj = int(profile.get("Number of Relevant Projects", 0))
    target_proj = IDEAL_PROFILE_TARGETS["Number of Relevant Projects"]
    proj_diff = student_proj - target_proj
    rows.append({
        "Profile Feature": "Relevant Projects Count",
        "Student Value": str(student_proj),
        "Target Benchmark": f"{target_proj}+",
        "Status / Gap": f"{proj_diff} projects" if proj_diff < 0 else "Target Met",
        "Suggested Action": f"Build {abs(proj_diff)} more full-scale domain project(s)." if proj_diff < 0 else "Highlight complexity in current projects.",
    })

    # 3. Average Project Complexity
    complexity_names = {1: "1 (Basic)", 2: "2 (Intermediate)", 3: "3 (Moderate)", 4: "4 (Advanced)", 5: "5 (Production)"}
    student_comp = int(profile.get("Average Project Complexity", 1))
    target_comp = IDEAL_PROFILE_TARGETS["Average Project Complexity"]
    comp_diff = student_comp - target_comp
    rows.append({
        "Profile Feature": "Project Complexity (1 - 5)",
        "Student Value": complexity_names.get(student_comp, str(student_comp)),
        "Target Benchmark": complexity_names.get(target_comp, str(target_comp)),
        "Status / Gap": f"{comp_diff} level" if comp_diff < 0 else "Target Met",
        "Suggested Action": "Refactor codebase with cloud deployment, logging, and unit tests." if comp_diff < 0 else "Document architectural choices in README.",
    })

    # 4. Previous Internship Experience
    student_intern = str(profile.get("Previous Internship Experience", "No")).strip()
    target_intern = IDEAL_PROFILE_TARGETS["Previous Internship Experience"]
    intern_met = student_intern.lower() == "yes"
    rows.append({
        "Profile Feature": "Previous Internship Experience",
        "Student Value": student_intern,
        "Target Benchmark": target_intern,
        "Status / Gap": "Present" if intern_met else "Missing (-1)",
        "Suggested Action": "Leverage past manager recommendations." if intern_met else "Substitute with open source contributions or research projects.",
    })

    # 5. Relevant Certifications
    student_certs = int(profile.get("Relevant Certifications", 0))
    target_certs = IDEAL_PROFILE_TARGETS["Relevant Certifications"]
    certs_diff = student_certs - target_certs
    rows.append({
        "Profile Feature": "Relevant Certifications",
        "Student Value": str(student_certs),
        "Target Benchmark": f"{target_certs}+",
        "Status / Gap": f"{certs_diff} certs" if certs_diff < 0 else "Target Met",
        "Suggested Action": "Pursue an industry-standard credential (e.g., AWS, Azure, Google)." if certs_diff < 0 else "Keep certifications updated on LinkedIn.",
    })

    # 6. Programming Languages
    student_langs = int(profile.get("Number of Programming Languages", 0))
    target_langs = IDEAL_PROFILE_TARGETS["Number of Programming Languages"]
    langs_diff = student_langs - target_langs
    rows.append({
        "Profile Feature": "Programming Languages",
        "Student Value": str(student_langs),
        "Target Benchmark": f"{target_langs}",
        "Status / Gap": f"{langs_diff} languages" if langs_diff < 0 else "Target Met",
        "Suggested Action": "Deepen mastery in a primary OOP and a secondary scripting language." if langs_diff < 0 else "Focus on depth of mastery over breadth.",
    })

    # 7. Number of Technical Skills
    student_skills = int(profile.get("Number of Technical Skills", 0))
    target_skills = IDEAL_PROFILE_TARGETS["Number of Technical Skills"]
    skills_diff = student_skills - target_skills
    rows.append({
        "Profile Feature": "Technical Skills Count",
        "Student Value": str(student_skills),
        "Target Benchmark": f"{target_skills}+",
        "Status / Gap": f"{skills_diff} skills" if skills_diff < 0 else "Target Met",
        "Suggested Action": f"Learn {abs(skills_diff)} more modern developer tools (Git, Docker, CI/CD, SQL)." if skills_diff < 0 else "Showcase practical skill applications.",
    })

    return pd.DataFrame(rows)
