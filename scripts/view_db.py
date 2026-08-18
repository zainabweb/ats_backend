"""
Quick terminal viewer for what's actually stored in ChromaDB.

Run from the project root (after activating your venv):
    python scripts/view_db.py
    python scripts/view_db.py --jobs-only
    python scripts/view_db.py --resumes-only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import chroma_service  # noqa: E402


def print_jobs():
    jobs = chroma_service.list_jobs()
    print(f"\n=== JOB DESCRIPTIONS ({len(jobs)}) ===")
    if not jobs:
        print("  (none saved yet)")
    for j in jobs:
        print(f"\n  job_id: {j['job_id']}")
        print(f"  title:  {j['job_title']}")
        print(f"  skills: {', '.join(j['required_skills'])}")
        print(f"  min_experience: {j['minimum_experience']} yrs | education: {j['education_requirement']}")
        print(f"  status: {j['status']}")


def print_resumes():
    resumes = chroma_service.list_resumes()
    print(f"\n=== CANDIDATE RESUMES ({len(resumes)}) ===")
    if not resumes:
        print("  (none screened yet)")
    for r in resumes:
        print(f"\n  resume_id: {r['resume_id']}")
        print(f"  name:  {r['full_name']}  |  email: {r['email']}  |  phone: {r['phone']}")
        print(f"  experience: {r['total_experience']} yrs ({r['experience_source']})")
        print(f"  education: {r['education']}")
        print(f"  skills: {', '.join(r['skills'])}")
        print(f"  file: {r['resume_file_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-only", action="store_true")
    parser.add_argument("--resumes-only", action="store_true")
    parser.add_argument("--raw", action="store_true", help="dump full JSON instead of the formatted view")
    args = parser.parse_args()

    if args.raw:
        data = {}
        if not args.resumes_only:
            data["jobs"] = chroma_service.list_jobs()
        if not args.jobs_only:
            data["resumes"] = chroma_service.list_resumes()
        print(json.dumps(data, indent=2, default=str))
    else:
        if not args.resumes_only:
            print_jobs()
        if not args.jobs_only:
            print_resumes()
        print()
