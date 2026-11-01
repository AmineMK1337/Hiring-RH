from schema import ResumeData, HRFormData, Justification
from mongodb import get_collection
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain.output_parsers import PydanticOutputParser
from llm import with_schema

# The LLM occasionally returns JSON that doesn't match the schema; a retry usually fixes it
RETRY_ON_BAD_JSON = dict(retry_if_exception_type=(OutputParserException,), stop_after_attempt=3)


MAX_JOB_DESCRIPTION_CHARS = 4000

JUSTIFICATION_SYSTEM = (
    "You are a careful, fair technical recruiter. You judge how well ONE candidate fits ONE job, "
    "using only the evidence in the candidate profile. Judge relevance to this specific job, not "
    "general quality: a strong profile from another field is a poor fit. Ignore anything unrelated "
    "to the job (name, gender, age, nationality, photo, address)."
)

JUSTIFICATION_HUMAN = """JOB
Title: {job_title}
Required skills: {required_skills}
Minimum experience: {min_experience} years
Additional criteria: {additional_criteria}
Full job description:
<<<
{job_description}
>>>

CANDIDATE
Current title: {current_title}
Experience:
{experience}
Skills: {skills}
Education:
{education}
Certifications: {certifications}

HOW TO SCORE
First write the text fields, citing evidence from the profile. Then give three scores from 0 to 10,
each against THIS job:
- skills_score: share of the job's key skills and tools the candidate shows.
  0 = none, 3 = a few, 5 = about half, 8 = most, 10 = all key skills.
- experience_score: years and level in the same kind of work as the job.
  0 = no related experience, 3 = some related work far below the requirement,
  5 = related work somewhat below the requirement, 8 = meets it, 10 = clearly exceeds it in closely related roles.
  Experience in an unrelated field does not count.
- domain_score: how close the candidate's field and past roles are to the job's field.
  0 = unrelated field, 3 = different field with transferable skills, 5 = adjacent field,
  8 = same field, different role type, 10 = same field and same role type.
Use the whole range and in-between values (2, 4, 6, 7, 9). Do not give 0 unless there is truly no evidence.

{format_instructions}"""


def _profile_lines(items, fmt: str) -> str:
    return "\n".join(fmt.format(i=i) for i in items) or "- none listed"


# Function to Generate LLM Justification prompt
def generate_llm_justification(resume: ResumeData, hr_input: HRFormData):
    """Chat messages asking the LLM to justify and score one candidate for one job.

    The candidate's name is deliberately left out: it is not job evidence.
    """
    parser = PydanticOutputParser(pydantic_object=Justification)
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUSTIFICATION_SYSTEM),
        ("human", JUSTIFICATION_HUMAN),
    ])
    job_description = (hr_input.job_description or "not provided")[:MAX_JOB_DESCRIPTION_CHARS]

    return prompt.format_messages(
        job_title=hr_input.job_title,
        required_skills=", ".join(hr_input.required_skills) or "not specified",
        min_experience=hr_input.min_experience,
        additional_criteria=hr_input.additional_criteria or "none",
        job_description=job_description,
        current_title=resume.job_title,
        experience=_profile_lines(resume.experience, "- {i.title} at {i.company} ({i.year})"),
        skills=", ".join(resume.skills) or "none listed",
        education=_profile_lines(resume.education, "- {i.degree}, {i.university} ({i.year})"),
        certifications=", ".join(c.name for c in resume.certifications) or "none",
        format_instructions=parser.get_format_instructions(),
    )


# Ask the LLM to evaluate one candidate against the job criteria
def justify_candidate(resume: ResumeData, hr_input: HRFormData, llm) -> Justification:
    parser = PydanticOutputParser(pydantic_object=Justification)
    chain = (with_schema(llm, Justification) | parser).with_retry(**RETRY_ON_BAD_JSON)
    return chain.invoke(generate_llm_justification(resume, hr_input))


# Rank candidates by the LLM's fit score (ties broken by number of skills)
def rank_candidates(candidates):
    return sorted(candidates,
                  key=lambda x: (x["justification"].fit_score, len(x["resume"].skills)),
                  reverse=True)


# Fetch and Rank Candidates
def find_best_candidates(hr_input: HRFormData, llm):
    collection = get_collection()
    resumes = list(collection.find())  # Load resumes from MongoDB
    candidates = []

    for resume_data in resumes:
        resume = ResumeData(**resume_data)
        try:
            justification = justify_candidate(resume, hr_input, llm)
        except Exception as e:
            # Skip this candidate instead of failing the whole ranking
            print(f"❌ Justification failed for {resume.name}: {e}")
            continue
        candidates.append({"resume": resume, "justification": justification})
        print(f"📄 Justification Done for {resume.name}")

    candidates = rank_candidates(candidates)

    # Select top N candidates
    return candidates[: hr_input.num_candidates], candidates[hr_input.num_candidates : hr_input.num_candidates + 2]
