"""
Dataset Generation Script for AI-Powered Internship Success Prediction System.

Generates a realistic, reproducible synthetic dataset of student internship applicants.
Target Column: 'Internship Outcome' (Selected / Not Selected)
Features:
  1. CGPA (0.0 - 10.0)
  2. Number of Relevant Projects (0+)
  3. Average Project Complexity (1 - 5)
  4. Previous Internship Experience (Yes / No)
  5. Relevant Certifications (0+)
  6. Number of Programming Languages (0+)
  7. Number of Technical Skills (0+)
  8. Internship Domain Applied (Domain Category)

Synthetic Data Assumptions & Logic:
- A latent selection propensity score is generated using realistic feature weights reflecting
  college internship recruitment criteria (academic standing, practical project depth, prior
  exposure, and domain fit).
- Controlled Gaussian noise is added to reflect unmeasured factors (interview communication,
  portfolio presentation, cultural fit, hiring manager subjectivity), ensuring the outcome
  is neither trivial nor completely deterministic.
- Realistic feature correlations are maintained (e.g., higher project complexity aligns with
  more technical skills).
- No target leakage: target is computed solely via latent function and never leaked back into features.
"""

import os
import numpy as np
import pandas as pd

DOMAINS = [
    "Web Development",
    "Artificial Intelligence and Machine Learning",
    "Data Science and Analytics",
    "Cybersecurity",
    "Cloud Computing",
    "Mobile App Development",
    "Software Engineering",
]

DOMAIN_WEIGHTS = [0.22, 0.18, 0.16, 0.12, 0.12, 0.10, 0.10]


def generate_internship_dataset(
    n_samples: int = 1000,
    random_state: int = 42,
    output_path: str = "data/internship_dataset.csv"
) -> pd.DataFrame:
    """
    Generate synthetic internship applicant dataset and save to CSV.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of student records to generate.
    random_state : int, default=42
        Seed for reproducibility.
    output_path : str, default='data/internship_dataset.csv'
        File destination path.

    Returns
    -------
    pd.DataFrame
        Generated dataset with student profiles and outcomes.
    """
    rng = np.random.default_rng(random_state)

    # 1. CGPA: Truncated normal distribution centered around 7.5 (min 5.0, max 9.9)
    cgpa_raw = rng.normal(loc=7.6, scale=1.0, size=n_samples)
    cgpa = np.clip(np.round(cgpa_raw, 2), 5.0, 9.95)

    # 2. Number of Relevant Projects: Poisson distribution (mean 3.2, clipped [0, 10])
    projects = np.clip(rng.poisson(lam=3.2, size=n_samples), 0, 10)

    # 3. Average Project Complexity: 1 to 5, correlated with projects & CGPA
    complexity_base = 1 + (projects / 3.0) * 0.8 + (cgpa / 10.0) * 1.5 + rng.normal(0, 0.7, size=n_samples)
    complexity = np.clip(np.round(complexity_base).astype(int), 1, 5)

    # 4. Previous Internship Experience: Bernoulli trial with ~32% base probability,
    # slightly boosted by higher CGPA and more projects
    prev_intern_prob = 0.15 + (cgpa - 5.0) * 0.04 + (projects * 0.03)
    prev_intern_prob = np.clip(prev_intern_prob, 0.05, 0.75)
    prev_intern_binary = rng.binomial(1, prev_intern_prob)
    prev_internship = np.where(prev_intern_binary == 1, "Yes", "No")

    # 5. Relevant Certifications: Poisson distribution (mean 1.6, clipped [0, 8])
    certifications = np.clip(rng.poisson(lam=1.6, size=n_samples), 0, 8)

    # 6. Number of Programming Languages: Poisson distribution (mean 3.1, clipped [1, 8])
    prog_languages = np.clip(rng.poisson(lam=3.1, size=n_samples), 1, 8)

    # 7. Number of Technical Skills: correlated with languages, projects, and certifications
    skills_base = 2 + prog_languages * 0.7 + projects * 0.5 + certifications * 0.4 + rng.normal(0, 1.2, size=n_samples)
    tech_skills = np.clip(np.round(skills_base).astype(int), 1, 15)

    # 8. Internship Domain Applied
    domain_applied = rng.choice(DOMAINS, size=n_samples, p=DOMAIN_WEIGHTS)

    # Latent Propensity Calculation for Selection:
    # Feature weights based on realistic internship hiring criteria:
    # - CGPA weight: 0.28
    # - Relevant Projects: 0.24
    # - Project Complexity: 0.22
    # - Previous Internship: 0.26
    # - Certifications: 0.12
    # - Programming Languages: 0.08
    # - Technical Skills: 0.16

    norm_cgpa = (cgpa - 7.0) / 1.2
    norm_projects = (projects - 3.0) / 1.8
    norm_complexity = (complexity - 3.0) / 1.1
    norm_intern = (prev_intern_binary - 0.3) / 0.45
    norm_certs = (certifications - 1.5) / 1.2
    norm_langs = (prog_languages - 3.0) / 1.3
    norm_skills = (tech_skills - 5.5) / 2.5

    # Domain specific synergy adjustments
    domain_synergy = np.zeros(n_samples)
    for i, d in enumerate(domain_applied):
        if d in ["Artificial Intelligence and Machine Learning", "Data Science and Analytics"]:
            domain_synergy[i] = 0.2 * norm_complexity[i] + 0.15 * norm_skills[i]
        elif d in ["Cloud Computing", "Cybersecurity"]:
            domain_synergy[i] = 0.25 * norm_certs[i] + 0.15 * norm_complexity[i]
        elif d in ["Web Development", "Mobile App Development"]:
            domain_synergy[i] = 0.25 * norm_projects[i] + 0.15 * norm_skills[i]
        else: # Software Engineering
            domain_synergy[i] = 0.15 * norm_cgpa[i] + 0.15 * norm_projects[i] + 0.15 * norm_langs[i]

    latent_score = (
        0.28 * norm_cgpa
        + 0.24 * norm_projects
        + 0.22 * norm_complexity
        + 0.26 * norm_intern
        + 0.12 * norm_certs
        + 0.08 * norm_langs
        + 0.16 * norm_skills
        + domain_synergy
    )

    # Add controlled Gaussian noise to simulate unobserved factors (interview communication, soft skills, fit)
    noise = rng.normal(loc=0.0, scale=0.65, size=n_samples)
    final_score = latent_score + noise

    # Logistic transformation to obtain continuous selection probability
    prob_selected = 1.0 / (1.0 + np.exp(-(final_score - 0.10)))

    # Generate binary outcome with realistic selection rate (~42-45%)
    selected_mask = prob_selected >= 0.50
    outcomes = np.where(selected_mask, "Selected", "Not Selected")

    df = pd.DataFrame({
        "CGPA": cgpa,
        "Number of Relevant Projects": projects,
        "Average Project Complexity": complexity,
        "Previous Internship Experience": prev_internship,
        "Relevant Certifications": certifications,
        "Number of Programming Languages": prog_languages,
        "Number of Technical Skills": tech_skills,
        "Internship Domain Applied": domain_applied,
        "Internship Outcome": outcomes,
    })

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Synthetic dataset with {len(df)} records saved to '{output_path}'.")
        print(f"Outcome distribution:\n{df['Internship Outcome'].value_counts(normalize=True)}")

    return df


if __name__ == "__main__":
    generate_internship_dataset()

