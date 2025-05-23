import os
import tempfile
import re
import logging
from datetime import datetime
import uuid
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnableSequence

class JobAssistant:
   
    def __init__(self, model_name="llama3:8b"):

        self._setup_logging()

        # Initialize LLM model with better parameters
        self.model = ChatOllama(
            model=model_name,
            temperature=0.3,  # More focused responses
            top_p=0.9,        # Better response quality
            repeat_penalty=1.1  # Reduce repetition
        )

        # Improved text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=400,
            separators=["\n\n", "\n", "(?<=\\. )", " ", ""],
            length_function=len
        )
       
        # Document stores
        self.resume_store = None
        self.job_posting_store = None
        self.skills_store = None
       
        # Retrievers
        self.resume_retriever = None
        self.job_posting_retriever = None
        self.skills_retriever = None
       
        # Initialize document types
        self.document_types = {
            "resume": False,
            "job_posting": False,
            "skills": False
        }
       
        # Initialize prompts
        self._initialize_prompts()

    def _setup_logging(self):
      
        self.logger = logging.getLogger("JobAssistant")
        self.logger.setLevel(logging.DEBUG)
    
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
    
        # Create file handler which logs even debug messages
        log_file = f"logs/job_assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
    
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
    
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
    
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
        self.logger.info("JobAssistant initialized")

    
    def _initialize_prompts(self):
        """Initialize various prompt templates for different tasks."""
        # Enhanced general QA prompt
        self.qa_prompt = PromptTemplate.from_template(
    """As a professional career advisor, provide a response structured as follows:
    
    ### Summary
    [1-3 sentence overview]
    
    ### Detailed Analysis
    [Several paragraphs or bullet points]
    
    ### Recommendations
    - Clear, actionable items
    - Prioritized when possible
    
    ### Additional Resources (if applicable)
    [Suggestions for further reading/learning]
    
    CONTEXT: {context}
    
    QUESTION: {question}
    
    Respond using professional but approachable tone, with markdown formatting."""
)
       
        # Enhanced resume improvement prompt
        self.resume_improvement_prompt = PromptTemplate.from_template(
    """Role: Act as an expert executive resume writer with 10+ years of experience. 
    Critically analyze the provided resume against the job requirements and deliver concise, high-impact recommendations to optimize the resume.

    Response Format (Follow Exactly):

    [Brief Summary]
    *(1-3 sentences summarizing the biggest gaps and opportunities for improvement.)*

    1. Missing Requirements *(2-7 critical items from the job description that are absent or underdeveloped.)*

[Specific skill, certification, or experience]

[Another missing element]

2. Experiences to Emphasize *(2-7 existing resume items that should be strengthened or repositioned to better match the role.)*

[Relevant experience to highlight]

[Another experience to expand]

3. Quantifiable Achievements *(2-5 measurable results to add—think $, %, time, efficiency gains.)*

"Increased revenue by X%" → "Boosted revenue by 27% in 6 months by…"

[Another metric suggestion]

4. Key Terms to Include *(3-8 exact keywords/phrases from the job description for ATS optimization.)*

[Keyword 1]

[Keyword 2]

5. Technical Skills to Highlight *(2-7 hard skills/tech tools to feature more prominently.)*

[Skill 1]

[Skill 2]

6. Section Reorganization *(2-7 structural changes—e.g., move "Leadership" before "Education," merge redundant sections.)*

[Change 1]

[Change 2]

    JOB REQUIREMENTS:
    {job_posting_context}

    CURRENT RESUME:
    {resume_context}

    USER REQUEST:
    {question}

    Rules:

Be brutally honest—omit fluff and focus on actionable fixes.

Prioritize specificity (avoid vague advice like "improve clarity").

Use bullet points only (no paragraphs).

Do not rewrite the resume—only provide recommendations.
"""
)
 

        # Missing skills prompt
        self.missing_skills_prompt = PromptTemplate.from_template(
    """As a career expert, analyze these documents to identify skill gaps. Structure your response EXACTLY as follows:

    ### Skill Gap Analysis Summary
    [1-3 sentence overview of main gaps]

    ### Critical Missing Skills (Hard Skills)
    - [Skill 1]: Explanation why this is important for the role
    - [Skill 2]: Explanation why this is important for the role
    (List 1-7 most critical technical skills missing)

    ### Skills Needing Development (Soft Skills)
    - [Skill 1]: How to develop this
    - [Skill 2]: How to develop this
    (List 1-5 soft skills needing work)

    ### Recommended Learning Resources
    - [Resource 1]: For [specific skill]
    - [Resource 2]: For [specific skill]
    (List 1-5 Learning Resources)

    DATA PROVIDED:
    - Resume Content: {resume_context}
    - Job Requirements: {job_posting_context}
    - Additional Skills: {skills_context}

    QUESTION: {question}

    Provide only the structured response above. If information is missing, say "I need more information about [specific missing data]."""
)
       
        # Cover letter prompt
        self.cover_letter_prompt = PromptTemplate.from_template(
            """You are an expert cover letter writer. Create a customized cover letter based on the candidate's
            resume and the job posting.
           
            Resume: {resume_context}
           
            Job Posting: {job_posting_context}
           
            Skills Profile: {skills_context}
           
            Question: {question}
           
            Create a professional, personalized cover letter that:
            1. Has a proper heading and salutation
            2. Opens with an engaging introduction
            3. Highlights relevant skills and experiences from the resume that match the job posting
            4. Explains why the candidate is interested in the position and company
            5. Includes a call to action and professional closing
            6. Is approximately 250-350 words in length
           
            Answer:"""
        )
       
        # Interview preparation prompt
        self.interview_prep_prompt = PromptTemplate.from_template(
            """You are an interview preparation coach. Based on the resume and job posting, provide advice on how
            the candidate should prepare for an interview for this position.
           
            Resume: {resume_context}
           
            Job Posting: {job_posting_context}
           
            Skills Profile: {skills_context}
           
            Question: {question}
           
            Provide comprehensive interview preparation advice, including:
            1. Potential technical questions based on the skills required and how to answer them
            2. Behavioral questions likely to be asked and how to structure responses using the STAR method
            3. Questions the candidate should ask the interviewer
            4. Key experiences from the resume to highlight during the interview
            5. Research to conduct about the company before the interview
            6. Tips for showcasing skills that might be missing or underdeveloped
           
            Answer:"""
        )
   


    def ingest_resume(self, file_path: str) -> None:
        """
        Enhanced resume ingestion with comprehensive validation, preprocessing, and error handling.
    
        Args:
            file_path: Path to the resume file (PDF)
        
        Raises:
            ValueError: If file is invalid, empty, or unprocessable
            RuntimeError: If processing fails unexpectedly
        """
        self.logger.info(f"Starting resume ingestion from: {file_path}")
    
        try:
            # ======================
            # 1. File Validation
            # ======================
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            if not file_path.lower().endswith('.pdf'):
                raise ValueError("Only PDF resumes are currently supported")
            
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError("Resume file too large (max 10MB)")
            if file_size < 1024:  # 1KB minimum
                raise ValueError("Resume file appears too small to be valid")

            # ======================
            # 2. Document Loading
            # ======================
            loader = PyPDFLoader(
                file_path=file_path,
                extract_images=False,  # Disable image extraction for performance
                headers={"User-Agent": "Resume Parser/1.0"}  # Some PDFs require this
            )
        
            try:
                docs = loader.load()
            except Exception as load_error:
                raise ValueError(f"Failed to parse PDF: {str(load_error)}") from load_error

            # ======================
            # 3. Content Validation
            # ======================
            if not docs:
                raise ValueError("The resume appears to be empty or unreadable")
            
            total_content = " ".join(doc.page_content for doc in docs)
            if len(total_content.strip()) < 200:  # Minimum reasonable resume length
                raise ValueError("The resume content appears too short to be valid")
            
            if not any(word in total_content.lower() for word in ["experience", "education", "skills"]):
                self.logger.warning("Resume missing common sections - may be low quality")

            # ======================
            # 4. Enhanced Preprocessing
            # ======================
            processed_docs = []
            for doc in docs:
                # Clean and sanitize content
                doc.page_content = self._clean_text(doc.page_content)
                doc.page_content = self._sanitize_content(doc.page_content)
            
                # Extract metadata from document structure
                lines = doc.page_content.split('\n')
                if len(lines) > 3:
                    # Detect section headers (all caps in first line)
                    if (lines[0].isupper() and 
                        10 < len(lines[0]) < 50 and 
                        not lines[0].isdigit()):
                        doc.metadata['section'] = lines[0].title()
                        doc.page_content = '\n'.join(lines[1:])  # Remove header from content
                
                    # Detect contact info in first 3 lines
                    if doc.metadata.get('page', 1) == 1 and len(processed_docs) == 0:
                        contact_info = self._extract_contact_info('\n'.join(lines[:3]))
                        if contact_info:
                            doc.metadata.update(contact_info)

                processed_docs.append(doc)

            # ======================
            # 5. Chunking Strategy
            # ======================
            chunks = self.text_splitter.split_documents(processed_docs)
            chunks = filter_complex_metadata(chunks)
        
            if not chunks:
                raise ValueError("No valid content chunks could be extracted from resume")

            self.logger.debug(f"Created {len(chunks)} chunks from resume")

            # ======================
            # 6. Enhanced Metadata
            # ======================
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "document_type": "resume",
                    "source": os.path.basename(file_path),
                    "chunk_id": f"res_{uuid.uuid4().hex[:8]}",
                    "chunk_num": i,
                    "total_chunks": len(chunks),
                    "processing_time": datetime.now().isoformat(),
                    "content_length": len(chunk.page_content)
                })

            # ======================
            # 7. Vector Store Creation
            # ======================
            try:
                self.resume_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=FastEmbedEmbeddings(),
                    collection_name=f"resume_{os.path.basename(file_path)}_{datetime.now().timestamp()}",
                    collection_metadata={
                        "hnsw:space": "cosine",
                        "description": "Resume document chunks",
                        "source_file": file_path
                    }
                )
            
                self.resume_retriever = self.resume_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 6,
                        "score_threshold": 0.25,
                        "fetch_k": 20,
                        "lambda_mult": 0.5  # Diversity parameter
                    }
                )
            
                self.document_types["resume"] = True
                self.logger.info(f"Successfully ingested resume with {len(chunks)} chunks")
            
            except Exception as e:
                self.logger.error(f"Failed to create vector store: {str(e)}", exc_info=True)
                raise RuntimeError("Failed to process resume due to system error") from e

        except ValueError as ve:
            self.logger.warning(f"Resume validation failed: {str(ve)}")
            raise  # Re-raise for UI handling
        except Exception as e:
            self.logger.error(f"Unexpected error ingesting resume: {str(e)}", exc_info=True)
            raise RuntimeError("An unexpected error occurred while processing your resume") from e


    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning for better processing"""
        # Normalize various bullet points and special characters
        text = re.sub(r'[•‣⁃→]', '-', text)
        # Remove header/footer artifacts
        text = re.sub(r'\n\d+\s*[A-Za-z]+\s*\d+', '\n', text)
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r'(?<!\n)\s+', ' ', text)
        # Clean up line breaks
        text = re.sub(r'(\s*\n){3,}', '\n\n', text)
        return text.strip()
   
    def ingest_job_posting(self, file_path: str) -> None:
        """
        Ingest a job posting PDF or text document and create a vector store.
       
        Args:
            file_path: Path to the job posting file
        """
        if file_path.lower().endswith('.pdf'):
            docs = PyPDFLoader(file_path=file_path).load()
        else:
            docs = TextLoader(file_path=file_path).load()
           
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
       
        # Add document type metadata
        for chunk in chunks:
            if "metadata" not in chunk.__dict__:
                chunk.metadata = {}
            chunk.metadata["document_type"] = "job_posting"
       
        self.job_posting_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings()
        )
       
        self.job_posting_retriever = self.job_posting_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.25,
            },
        )
       
        self.document_types["job_posting"] = True
   
    def ingest_skills_profile(self, file_path: str) -> None:
        """
        Ingest a text file containing skills information and create a vector store.
       
        Args:
            file_path: Path to the skills text file
        """
        docs = TextLoader(file_path=file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
       
        # Add document type metadata
        for chunk in chunks:
            if "metadata" not in chunk.__dict__:
                chunk.metadata = {}
            chunk.metadata["document_type"] = "skills"
       
        self.skills_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings()
        )
       
        self.skills_retriever = self.skills_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.25,
            },
        )
       
        self.document_types["skills"] = True
   
    def get_document_content(self, doc_type: str, query: str = "") -> str:
        """Enhanced retrieval with better error handling"""
        try:
            if doc_type == "resume" and self.resume_retriever:
                docs = self.resume_retriever.invoke(query)
                return "\n\n".join(self._clean_chunk(d) for d in docs[:3])
                
            elif doc_type == "job_posting" and self.job_posting_retriever:
                docs = self.job_posting_retriever.invoke(query)
                return "\n\n".join(self._clean_chunk(d) for d in docs[:3])
                
            elif doc_type == "skills" and self.skills_retriever:
                docs = self.skills_retriever.invoke(query)
                return "\n\n".join(self._clean_chunk(d) for d in docs[:3])
                
            return f"No {doc_type.replace('_', ' ')} content available"
            
        except Exception as e:
            self.logger.error(f"Error retrieving {doc_type} content: {str(e)}")
            return ""
  
            

        
    def _clean_chunk(self, doc) -> str:
        """Enhanced chunk cleaning with better metadata handling"""
        content = doc.page_content
        if hasattr(doc, 'metadata'):
            if 'section' in doc.metadata:
                content = f"### {doc.metadata['section']}\n{content}"
            if 'page' in doc.metadata:
                content = f"(Page {doc.metadata['page']})\n{content}"
        return self._clean_text(content)
  
    def improve_resume(self, query: str) -> str:
        """
        Provide advice on how to improve the resume for the specific job.
       
        Args:
            query: User's question about resume improvement
           
        Returns:
            Advice on improving the resume
        """
        if not all([self.document_types["resume"], self.document_types["job_posting"]]):
            missing = []
            if not self.document_types["resume"]:
                missing.append("resume")
            if not self.document_types["job_posting"]:
                missing.append("job posting")
            return f"Please upload the missing documents: {', '.join(missing)}."
       
        # Get document contents
        resume_content = self.get_document_content("resume")
        job_posting_content = self.get_document_content("job_posting")
        skills_content = self.get_document_content("skills") if self.document_types["skills"] else "No skills profile provided."
       
        # Create chain
        chain = self.resume_improvement_prompt | self.model | StrOutputParser()
       
        # Execute chain
        response = chain.invoke({
            "resume_context": resume_content,
            "job_posting_context": job_posting_content,
            "skills_context": skills_content,
            "question": query
        })
       
        return response
   
    def identify_missing_skills(self, query: str) -> str:
        """
        Identify skills that are missing based on the job posting.
       
        Args:
            query: User's question about missing skills
           
        Returns:
            Analysis of missing skills
        """
        if not all([self.document_types["resume"], self.document_types["job_posting"]]):
            missing = []
            if not self.document_types["resume"]:
                missing.append("resume")
            if not self.document_types["job_posting"]:
                missing.append("job posting")
            return f"Please upload the missing documents: {', '.join(missing)}."
       
        # Get document contents
        resume_content = self.get_document_content("resume")
        job_posting_content = self.get_document_content("job_posting")
        skills_content = self.get_document_content("skills") if self.document_types["skills"] else "No skills profile provided."
       
        # Create chain
        chain = self.missing_skills_prompt | self.model | StrOutputParser()
    
       
        # Execute chain
        response = chain.invoke({
            "resume_context": resume_content,
            "job_posting_context": job_posting_content,
            "skills_context": skills_content,
            "question": query
        })
       
        return response
   
    def create_cover_letter(self, query: str) -> str:
        """
        Create a customized cover letter based on the resume and job posting.
       
        Args:
            query: User's request for a cover letter
           
        Returns:
        except Exception as e:
            self.logger.error(f"Error retrieving {doc_type} content: {str(e)}", exc_info=True)
            return ""
            A customized cover letter
        """
        if not all([self.document_types["resume"], self.document_types["job_posting"]]):
            missing = []
            if not self.document_types["resume"]:
                missing.append("resume")
            if not self.document_types["job_posting"]:
                missing.append("job posting")
            return f"Please upload the missing documents: {', '.join(missing)}."
       
        # Get document contents
        resume_content = self.get_document_content("resume")
        job_posting_content = self.get_document_content("job_posting")
        skills_content = self.get_document_content("skills") if self.document_types["skills"] else "No skills profile provided."
       
        # Create chain
        chain = self.cover_letter_prompt | self.model | StrOutputParser()
        
       
        # Execute chain
        response = chain.invoke({
            "resume_context": resume_content,
            "job_posting_context": job_posting_content,
            "skills_context": skills_content,
            "question": query
        })
       
        return response
   
    def prepare_for_interview(self, query: str) -> str:
        """
        Provide advice on how to prepare for an interview for this job.
       
        Args:
            query: User's question about interview preparation
           
        Returns:
            Interview preparation advice
        """
        if not all([self.document_types["resume"], self.document_types["job_posting"]]):
            missing = []
            if not self.document_types["resume"]:
                missing.append("resume")
            if not self.document_types["job_posting"]:
                missing.append("job posting")
            return f"Please upload the missing documents: {', '.join(missing)}."
       
        # Get document contents
        resume_content = self.get_document_content("resume")
        job_posting_content = self.get_document_content("job_posting")
        skills_content = self.get_document_content("skills") if self.document_types["skills"] else "No skills profile provided."
       
        # Create chain
        chain = self.interview_prep_prompt | self.model | StrOutputParser()
       
        # Execute chain
        response = chain.invoke({
            "resume_context": resume_content,
            "job_posting_context": job_posting_content,
            "skills_context": skills_content,
            "question": query
        })
       
        return response
  

    def ask(self, query: str, retries: int = 2) -> str:
        """Enhanced query handling with timeouts"""
        try:
            # Validate input
            query = query.strip()
            if not query or len(query) < 5:
                return "Please provide a more detailed question."
                
            # Check documents
            if not any(self.document_types.values()):
                return ("Please upload relevant documents first. I can help with:\n"
                    "- Resume analysis (upload your resume)\n"
                    "- Job matching (upload resume and job description)\n"
                    "- Interview prep (upload job description)")
                
            # Process with timeout
            response = None
            for attempt in range(retries + 1):
                try:
                    response = self._process_query_attempt(query)
                    if response and len(response.split()) > 10:  # Basic validation
                        return response
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt} failed: {str(e)}")
                    if attempt == retries:
                        raise
                    
            return response or "I couldn't generate a response. Please try again."
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {str(e)}")
            return "An error occurred while processing your request. Please try again."

    def log_retrieval_stats(self, query: str, docs: list, doc_type: str):
        """Log detailed retrieval statistics."""
        if not docs:
            self.logger.warning(f"No documents retrieved for {doc_type} with query: {query}")
            return
    
        self.logger.debug(f"Retrieved {len(docs)} documents for {doc_type}")
    
        # Log scores if available
        if hasattr(docs[0], 'metadata') and 'score' in docs[0].metadata:
            scores = [doc.metadata.get('score', 'N/A') for doc in docs]
            self.logger.debug(f"Retrieval scores for {doc_type}: {scores}")
            self.logger.debug(f"Average score: {sum(scores)/len(scores):.2f}")
            self.logger.debug(f"Max score: {max(scores):.2f}")
            self.logger.debug(f"Min score: {min(scores):.2f}")
    
        # Log document lengths
        lengths = [len(doc.page_content) for doc in docs]
        self.logger.debug(f"Document lengths (chars): {lengths}")

    def _expand_query(self, query: str) -> str:
        """
        Enhance the query with alternative phrasings using LLM for better retrieval.
        
        Args:
            query: The original user query
            
        Returns:
            Expanded query with alternative phrasings if helpful,
            otherwise returns original query
        """
        # Don't expand very short or empty queries
        query = query.strip()
        if len(query.split()) < 3 or len(query) < 10:
            return query
        
        try:
            # Create a more robust prompt template
            expansion_prompt = PromptTemplate.from_template(
                """You are a query expansion assistant. Given the original query below, generate
                2-3 alternative phrasings that might help retrieve more relevant document chunks.
                Focus on:
                - Synonym substitution
                - Professional terminology variations
                - Broader/narrower interpretations
                - Common alternative phrasings in recruitment contexts
                
                ORIGINAL QUERY: {query}
                
                ALTERNATIVE PHRASINGS:
                1. """
            )
            
            # Create and execute the chain
            chain = (
                expansion_prompt 
                | self.model 
                | StrOutputParser()
            )
            
            # Get expansions with timeout safety
            expansions = chain.invoke({"query": query})
            
            # Parse the response safely
            if not expansions:
                return query
                
            # Extract the first alternative phrasing
            alternatives = []
            for line in expansions.split('\n'):
                line = line.strip()
                if line and re.match(r'^\d+\.', line):
                    alt = re.sub(r'^\d+\.\s*', '', line)
                    if alt and alt not in alternatives:
                        alternatives.append(alt)
                        if len(alternatives) >= 2:  # Limit to top 2 alternatives
                            break
            
            # Format the expanded query
            if alternatives:
                return f"{query} [or: {', '.join(alternatives)}]"
            return query
            
        except Exception as e:
            self.logger.debug(f"Query expansion failed for '{query}': {str(e)}")
            return query  # Fallback to original query
    
    def _post_process_response(self, response: str) -> str:
        """Enhanced response cleaning and structuring"""
        
        # Ensure consistent section headers
        response = re.sub(r'(?i)(\n\s*)(analysis|recommendations?|summary|key points)(\s*:\s*)', 
                        lambda m: f"\n\n### {m.group(2).title()}\n", response)
        
        # Normalize bullet points
        response = re.sub(r'(?<!\n)[•‣⁃*]\s*', '\n- ', response)
        response = re.sub(r'(?<!\n)(\d+\.\s)', '\n\\1', response)
        
        # Ensure proper spacing around sections
        response = re.sub(r'(\n\s*){3,}', '\n\n', response.strip())
        
        # Capitalize headings
        response = re.sub(r'^(#+\s*\w+)', 
                        lambda m: m.group(1).title(), 
                        response, flags=re.MULTILINE)
        
        # Remove empty sections
        response = re.sub(r'### [^\n]+\n\n+(?=###)', '', response)
        
        return response.strip()

    

    def _process_query_attempt(self, query: str) -> str:
        """Single attempt at processing a query"""
        # Get relevant context from all documents
        context_parts = []
        
        if self.document_types["resume"]:
            resume_content = self.get_document_content("resume", query)
            if resume_content:
                context_parts.append(f"### RESUME CONTENT\n{resume_content}")
        
        if self.document_types["job_posting"]:
            job_content = self.get_document_content("job_posting", query)
            if job_content:
                context_parts.append(f"### JOB DESCRIPTION\n{job_content}")
        
        if self.document_types["skills"]:
            skills_content = self.get_document_content("skills", query)
            if skills_content:
                context_parts.append(f"### SKILLS PROFILE\n{skills_content}")
        
        full_context = "\n\n".join(context_parts) if context_parts else "No relevant context found"
        
        # Classify query type
        query_type = self._classify_query(query)
        self.logger.debug(f"Classified query as: {query_type}")
        
        # Route to appropriate handler
        handlers = {
            "resume_improvement": self.improve_resume,
            "missing_skills": self.identify_missing_skills,
            "cover_letter": self.create_cover_letter,
            "interview_prep": self.prepare_for_interview
        }
        
        if query_type in handlers:
            return handlers[query_type](query)
        
        # Default to general QA
        chain = self.qa_prompt | self.model | StrOutputParser()
        return chain.invoke({"context": full_context, "question": query})
    


    def _classify_query(self, query: str) -> str:
        """Better query classification with structure expectations"""
        classifier_prompt = """Analyze this career question and classify its type, then specify the expected response structure:
        
        Question: {query}
        
        Types and Structures:
        - resume_improvement: Should use ### Content Alignment, ### Keyword Optimization, ### Structural Improvements sections
        - missing_skills: Should list skills in categories (Technical, Soft, etc.) with ### Critical Missing Skills section
        - cover_letter: Should return properly formatted letter with [Date], [Address] blocks
        - interview_prep: Should use ### Technical Questions, ### Behavioral Questions sections
        - general: Should use ### Summary, ### Analysis, ### Recommendations sections
        
        Return only the type label:"""
        
        try:
            chain = PromptTemplate.from_template(classifier_prompt) | self.model | StrOutputParser()
            return chain.invoke({"query": query}).strip().lower()
        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            return "general"

    def _validate_response(self, response: str) -> bool:
        """More lenient validation for skills queries"""
        if not response:
            return False
            
        # Skills-specific validation
        if "skill" in response.lower():
            required_phrases = [
                "missing",
                "recommend",
                "develop",
                "skill"
            ]
            if not all(phrase in response.lower() for phrase in required_phrases):
                self.logger.debug("Rejected: Missing key skill analysis components")
                return False
                
            return True
        
        # Default validation for other query types
        return len(response.split()) > 25 and "\n" in response

    # Add to job_assistant.py
    def _sanitize_content(self, text: str) -> str:
        """Remove potentially sensitive information"""
        # Remove phone numbers
        text = re.sub(r'(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}', '[PHONE]', text)
        # Remove email addresses
        text = re.sub(r'[\w\.-]+@[\w\.-]+', '[EMAIL]', text)

        # Remove physical addresses (simple pattern)
        text = re.sub(
        r'\d{1,5}\s[\w\s]{3,}\s(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln)', 
        '[ADDRESS]', 
        text, 
        flags=re.IGNORECASE)
        return text
    
    def _extract_contact_info(self, text: str) -> dict:
        """Extract contact information from resume header"""
        contact_info = {}
    
        # Email extraction
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
        if emails:
            contact_info['email'] = emails[0]
    
        # Phone extraction
        phones = re.findall(
        r'(\+?\d{1,3}[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}', 
        text)

        if phones:
            contact_info['phone'] = phones[0]
    
        # LinkedIn/profile URL extraction
        urls = re.findall(
        r'(https?:\/\/(?:www\.)?linkedin\.com\/in\/[a-zA-Z0-9-]+)|(https?:\/\/(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\/[a-zA-Z0-9-]*)', 
        text)

        if urls:
            contact_info['profile_url'] = urls[0][0] or urls[0][1]
    
        return contact_info

    def _default_retrieval(self, doc_type: str, query: str) -> str:
        """Default retrieval method for document content"""
        try:
            if doc_type == "resume" and self.resume_retriever:
                docs = self.resume_retriever.invoke(query)
                self.log_retrieval_stats(query, docs, doc_type)
                return "\n\n".join(self._clean_chunk(d) for d in docs[:3])
                
            elif doc_type == "job_posting" and self.job_posting_retriever:
                docs = self.job_posting_retriever.invoke(query)
                self.log_retrieval_stats(query, docs, doc_type)
                return "\n\n".join(self._clean_chunk(d) for d in docs[:3])
                
            elif doc_type == "skills" and self.skills_retriever:
                docs = self.skills_retriever.invoke(query)
                self.log_retrieval_stats(query, docs, doc_type)
                return "\n\n".join(self._clean_chunk(d) for d in docs[:3])
                
            return f"No content available for {doc_type}"
            
        except Exception as e:
            self.logger.error(f"Error retrieving {doc_type} content: {str(e)}", exc_info=True)
            return ""