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
   
    def __init__(self, model_name="mistral"):

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
            """As a professional career advisor with 10+ years experience, analyze the following 
            context and provide a detailed, structured response to the question below.

            CONTEXT:
            {context}

            QUESTION: 
            {question}

            RESPONSE GUIDELINES:
            1. Be specific, concrete, and actionable
            2. Use professional tone but remain approachable
            3. Structure response with clear headings if longer than 3 sentences
            4. Provide examples where relevant
            5. If suggesting changes, explain the benefits
            6. Maintain focus on practical career advice

            WELL-STRUCTURED RESPONSE:"""
        )
       
        # Enhanced resume improvement prompt
        self.resume_improvement_prompt = PromptTemplate.from_template(
            """As an executive resume writer (20+ years experience), analyze this resume against 
            the job requirements and provide specific, actionable recommendations.

            JOB REQUIREMENTS (Key Priorities):
            {job_posting_context}

            CURRENT RESUME CONTENT:
            {resume_context}

            ADDITIONAL SKILLS:
            {skills_context}

            USER REQUEST:
            {question}

            Provide recommendations covering these areas (use headings):
            
            [Content Alignment]
            - Missing requirements to add
            - Existing experiences to emphasize
            - Quantifiable achievements to highlight
            
            [Keyword Optimization]
            - Specific terms from job description to include
            - Technical skills to feature more prominently
            
            [Structural Improvements]
            - Sections to reorganize/restructure
            - Visual presentation suggestions
            
            [Additional Notes]
            - Any other observations"""
        )
 

        # Missing skills prompt
        self.missing_skills_prompt = PromptTemplate.from_template(
            """You are an expert career advisor. Analyze the resume and job posting below to identify skills
            the candidate is missing or should develop further for this particular job.
           
            Resume: {resume_context}
           
            Job Posting: {job_posting_context}
           
            Skills Profile: {skills_context}
           
            Question: {question}
           
            Provide a detailed analysis of:
            1. Critical skills mentioned in the job posting that are missing from the resume
            2. Skills that are present but need more emphasis or development
            3. Technical skills gap assessment
            4. Soft skills gap assessment
            5. Recommendations for skill acquisition or improvement
           
            Answer:"""
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
                "score_threshold": 0.5,
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
                "score_threshold": 0.5,
            },
        )
       
        self.document_types["skills"] = True
   
    @lru_cache(maxsize=3)
    def get_document_content(self, doc_type: str, query: str = "") -> str:
        """Enhanced retrieval with multiple fallback strategies"""
        retriever = getattr(self, f"{doc_type}_retriever", None)
        if not retriever:
            self.logger.warning(f"No retriever available for {doc_type}")
            return ""
    
        try:
            strategies = [
                lambda: retriever.invoke(query),  # Original query
                lambda: retriever.invoke(self._expand_query(query)),  # Expanded query
                lambda: retriever.invoke("key qualifications and requirements"),  # Generic
                lambda: getattr(self, f"{doc_type}_store").similarity_search(" ", k=4)  # Fallback
            ]
            
            docs = []
            for strategy in strategies:
                if len(docs) < 2:  # Minimum number of chunks we want
                    try:
                        result = strategy()
                        if result:
                            docs.extend(result if isinstance(result, list) else [result])
                    except Exception as e:
                        self.logger.debug(f"Retrieval strategy failed: {str(e)}")
            
            self.log_retrieval_stats(query, docs, doc_type)
            
            if not docs:
                self.logger.warning(f"No documents retrieved for {doc_type} after all strategies")
                return ""
            
            # Process and prioritize chunks
            cleaned_chunks = []
            for doc in docs[:8]:  # Limit to top 8 chunks
                try:
                    cleaned = self._clean_chunk(doc)
                    if cleaned:
                        cleaned_chunks.append(cleaned)
                except Exception as e:
                    self.logger.debug(f"Error cleaning chunk: {str(e)}")
            
            return "\n\n---\n\n".join(cleaned_chunks) if cleaned_chunks else ""
            
        except Exception as e:
            self.logger.error(f"Error retrieving {doc_type} content: {str(e)}", exc_info=True)
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
        """Enhanced query handling with retries and validation"""
        self.logger.info(f"Processing query: '{query}'")
        
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
        
        # Try multiple times if needed
        for attempt in range(retries + 1):
            try:
                response = self._process_query_attempt(query)
                if self._validate_response(response):
                    return self._post_process_response(response)
                self.logger.warning(f"Poor response quality on attempt {attempt}")
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {str(e)}", exc_info=True)
        
        return ("I'm having trouble providing a quality response. "
                "Please try:\n1. Rephrasing your question\n"
                "2. Uploading more complete documents\n"
                "3. Asking a more specific question")
    

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
        """Use LLM to expand the query for better retrieval"""
        if len(query.split()) < 4:  # Only expand short queries
            expansion_prompt = f"""Original query: {query}
            
            Generate 3 alternative phrasings that might help retrieve relevant document chunks:
            1. """
            
            try:
                chain = PromptTemplate.from_template("{query}") | self.model | StrOutputParser()
                expansions = chain.invoke({"query": expansion_prompt})
                return f"{query} ({expansions.split('\n')[0]})"
            except:
                return query
        return query
    
    def _post_process_response(self, response: str) -> str:
        """Clean and enhance the final response"""
        # Fix common formatting issues
        response = re.sub(r'(\n\s*){3,}', '\n\n', response)
        response = re.sub(r'(?<!\n)\s*([\-•*])\s*', '\n\\1 ', response)
        
        # Ensure proper capitalization
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentences = [s[0].upper() + s[1:] if s and s[0].islower() else s for s in sentences]
        response = ' '.join(sentences)
        
        # Remove hallucinated references
        response = re.sub(r'\b(?:Note|Reference):.*$', '', response, flags=re.MULTILINE|re.IGNORECASE)
        
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
        """Better query classification using LLM"""
        classifier_prompt = """Analyze this career-related question and classify its type:
        
        Question: {query}
        
        Possible types:
        - resume_improvement (requests about modifying/improving resume)
        - missing_skills (asks about skills gaps)
        - cover_letter (requests for cover letter help)
        - interview_prep (asks about interview preparation)
        - general (other career-related questions)
        
        Return just the type label (no explanation):"""
        
        try:
            chain = PromptTemplate.from_template(classifier_prompt) | self.model | StrOutputParser()
            return chain.invoke({"query": query}).strip().lower()
        except:
            return "general"

    def _validate_response(self, response: str) -> bool:
        """Check if response meets quality standards"""
        if not response:
            return False
        if len(response.split()) < 25:  # Too short
            return False
        if "I don't know" in response or "not provided" in response.lower():
            return False
        if response.count('\n') < 2 and len(response) > 300:  # Needs more structure
            return False
        return True

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

