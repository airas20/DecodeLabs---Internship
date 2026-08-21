"""
Tech Stack Recommender - Project 3 (DecodeLabs AI Internship)

Author: Airas Sultan
"""

import csv
import math
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "raw_skills.csv")
TOP_N = 3


def load_jobs(path):
    jobs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["role"].strip()
            skills = [s.strip().lower() for s in row["skills"].split(";") if s.strip()]
            jobs.append({"role": role, "skills": skills})
    return jobs


def build_vocab(jobs, query_terms):
    vocab = set()
    for job in jobs:
        vocab.update(job["skills"])
    vocab.update(query_terms)
    return sorted(vocab)


def term_freq(terms, vocab):
    
    vec = [0.0] * len(vocab)
    if not terms:
        return vec
    for i, word in enumerate(vocab):
        count = terms.count(word)
        vec[i] = count / len(terms)
    return vec


def inverse_doc_freq(jobs, vocab):
    n_docs = len(jobs)
    idf = [0.0] * len(vocab)
    for i, word in enumerate(vocab):
        docs_with_word = sum(1 for job in jobs if word in job["skills"])
       
        idf[i] = math.log(n_docs / (1 + docs_with_word)) + 1
    return idf


def tfidf(terms, vocab, idf):
    tf = term_freq(terms, vocab)
    return [tf[i] * idf[i] for i in range(len(vocab))]


def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def get_user_skills():
    print("Tell me a few of your skills or interests (at least 3), separated by commas.")
    print("e.g. python, cloud computing, automation\n")
    raw = input("Your skills: ")
    skills = [s.strip().lower() for s in raw.split(",") if s.strip()]

    while len(skills) < 3:
        print(f"\nThat's only {len(skills)} skill(s), I need at least 3 to get a good match.")
        raw = input("Add a few more, comma separated: ")
        skills += [s.strip().lower() for s in raw.split(",") if s.strip()]

    return skills


def recommend(user_skills, jobs, top_n=TOP_N):
    vocab = build_vocab(jobs, user_skills)
    idf = inverse_doc_freq(jobs, vocab)

    user_vec = tfidf(user_skills, vocab, idf)

    scored = []
    for job in jobs:
        job_vec = tfidf(job["skills"], vocab, idf)
        score = cosine_similarity(user_vec, job_vec)
        scored.append((job["role"], score, job["skills"]))

    scored.sort(key=lambda x: x[1], reverse=True)

    
    if scored[0][1] == 0:
        print("\nCouldn't find a close match for those exact skills, showing popular roles instead.\n")
        return scored[:top_n]

    return scored[:top_n]


def main():
    jobs = load_jobs(DATA_FILE)
    user_skills = get_user_skills()
    results = recommend(user_skills, jobs)

    print("\n--- Top Recommended Career Paths ---\n")
    for rank, (role, score, skills) in enumerate(results, start=1):
        match_pct = round(score * 100, 1)
        print(f"{rank}. {role}  ({match_pct}% match)")
        print(f"   Key skills: {', '.join(skills)}\n")


if __name__ == "__main__":
    main()
