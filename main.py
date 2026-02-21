import os
import pypdfium2 as pdfium
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM

class CandidateScore(BaseModel):
    score: int
    key_matching_skills: list[str]
    missing_skills: list[str]
    justification: str

def extract_text_from_pdf(pdf_path: str) -> str:
    pdf = pdfium.PdfDocument(pdf_path)
    text = "\n".join([page.get_textpage().get_text_range() for page in pdf])
    return text

def calculate_similarity(jd_text: str, resume_text: str) -> float:

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    jd_vector = model.encode(jd_text, convert_to_tensor=True)
    resume_vector = model.encode(resume_text, convert_to_tensor=True)
    
    cosine_score = util.cos_sim(jd_vector, resume_vector).item()
    return round(cosine_score * 100, 2)

def run_hr_crew(candidate_name: str, jd_text: str, resume_text: str, math_score: float):
    
    local_qwen = LLM(model="ollama/qwen2.5:14b", base_url="http://localhost:11434")


    tech_evaluator = Agent(
        role='Senior Technical Screener',
        goal='Analyze resumes against job descriptions with ruthless accuracy.',
        backstory='You are a veteran IT recruiter who specializes in AI, PyTorch, and systems engineering. You strictly stick to facts.',
        llm=local_qwen,
        verbose=True
    )

    
    hr_manager = Agent(
        role='HR Hiring Manager',
        goal='Synthesize technical feedback into a final structured JSON decision.',
        backstory='You take raw technical analysis and convert it into a final score and justification for the leadership dashboard.',
        llm=local_qwen,
        verbose=True
    )


    analysis_task = Task(
        description=f"Analyze {candidate_name}'s Resume: {resume_text}\nAgainst JD: {jd_text}\nMath Score: {math_score}/100.",
        expected_output="Bulleted list of matching and missing skills.",
        agent=tech_evaluator
    )

    decision_task = Task(
        description="Take the technical analysis and generate the final candidate score and justification.",
        expected_output="Structured JSON evaluation.",
        agent=hr_manager,
        output_pydantic=CandidateScore # Forces the Crew to output your strict schema
    )

    hr_crew = Crew(
        agents=[tech_evaluator, hr_manager],
        tasks=[analysis_task, decision_task],
        process=Process.sequential,
        verbose=True
    )
    
    return hr_crew.kickoff().pydantic.model_dump()


if __name__ == "__main__":
    job_description = "Looking for a GenAI Systems Engineer with PyTorch, LangGraph, and industrial manufacturing experience."
    resume_folder = "resumes"
    results = []
    
    for filename in os.listdir(resume_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(resume_folder, filename)
            candidate_name = filename.replace(".pdf", "")
            
            # Step-by-Step processing
            resume_text = extract_text_from_pdf(filepath)
            similarity = calculate_similarity(job_description, resume_text)
            evaluation = run_hr_crew(candidate_name, job_description, resume_text, similarity)
            
            evaluation['Candidate'] = candidate_name
            evaluation['Vector_Match_%'] = similarity
            results.append(evaluation)
            
    
    df = pd.DataFrame(results)
    df = df[['Candidate', 'Vector_Match_%', 'score', 'key_matching_skills', 'missing_skills', 'justification']]
    print("\n✅ Final Ranked Dashboard:")
    print(df.to_markdown(index=False))
