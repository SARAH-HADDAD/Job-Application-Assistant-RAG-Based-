import os
import tempfile
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

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
    """
    An AI assistant that analyzes your resume, job postings, and skills to provide
    personalized career advice.
    """
   
    def __init__(self, model_name="mistral"):

        self._setup_logging()

        # Initialize LLM model
        self.model = ChatOllama(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
       
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
        """Configure logging for the JobAssistant."""
        self.logger = logging.getLogger("JobAssistant")
        self.logger.setLevel(logging.INFO)
    
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
       
        # General QA prompt for document retrieval
        self.qa_prompt = PromptTemplate.from_template(
            """You are an AI career assistant. Use the following context to answer the question.
            If you don't know the answer, say you don't know. Be concise but informative.
           
            Question: {question}
            Context: {context}
           
            Answer:"""
        )
       
        # Resume improvement prompt
        self.resume_improvement_prompt = PromptTemplate.from_template(
            """You are an expert resume consultant. Analyze the resume and job posting below to suggest specific
            improvements to make the resume more appealing for this particular job.
           
            Resume: {resume_context}
           
            Job Posting: {job_posting_context}
           
            Skills Profile: {skills_context}
           
            Question: {question}
           
            Provide specific, actionable recommendations on how to modify the resume to better match this job posting.
            Focus on:
            1. Skills alignment
            2. Experience highlighting
            3. Resume formatting and structure
            4. Keywords to include
            5. Sections to emphasize or add
           
            Answer:"""
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
        """Ingest a resume PDF document and create a vector store."""
        self.logger.info(f"Ingesting resume from: {file_path}")
    
        try:
            docs = PyPDFLoader(file_path=file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
            self.logger.debug(f"Created {len(chunks)} chunks from resume")
        
            # Add document type metadata
            for chunk in chunks:
                if "metadata" not in chunk.__dict__:
                    chunk.metadata = {}
                chunk.metadata["document_type"] = "resume"
        
            self.resume_store = Chroma.from_documents(
                documents=chunks,
                embedding=FastEmbedEmbeddings())
            
            self.logger.debug("Created Chroma vector store for resume")
        
            self.resume_retriever = self.resume_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.3,},)
            self.logger.info("Resume ingestion completed successfully")
        
            self.document_types["resume"] = True
        except Exception as e:
            self.logger.error(f"Error ingesting resume: {str(e)}")
            raise
   
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
   
    def get_document_content(self, doc_type: str) -> str:
        """
        Get the full content of a document type.
       
        Args:
            doc_type: Type of document ("resume", "job_posting", or "skills")
           
        Returns:
            String containing the full document content
        """
        retriever = None
        if doc_type == "resume" and self.resume_retriever:
            retriever = self.resume_retriever
        elif doc_type == "job_posting" and self.job_posting_retriever:
            retriever = self.job_posting_retriever
        elif doc_type == "skills" and self.skills_retriever:
            retriever = self.skills_retriever
       
        if not retriever:
            self.logger.warning(f"No retriever available for document type: {doc_type}")
            return "Document not available"
        
        try:
            # Get all documents from the retriever
            docs = retriever.invoke("")
            self.logger.debug(f"Retrieved {len(docs)} documents for {doc_type}")

            if docs:
                if hasattr(docs[0], 'metadata') and 'score' in docs[0].metadata:
                    scores = [doc.metadata.get('score', 'N/A') for doc in docs]
                    self.logger.debug(f"Retrieval scores for {doc_type}: {scores}")
            
            return "\n\n".join([doc.page_content for doc in docs])
        
        except Exception as e:
            self.logger.error(f"Error retrieving {doc_type} content: {str(e)}")
            return f"Error retrieving document content: {str(e)}"
        
   
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
   
    def ask(self, query: str) -> str:
        """Process a query and route it to the appropriate function."""
        self.logger.info(f"Processing query: {query}")
    
        # Check if any documents have been uploaded
        if not any(self.document_types.values()):
            self.logger.warning("No documents uploaded - returning prompt to upload documents")
            return "Please upload at least one document (resume, job posting, or skills profile)."
    
        # Log document status
        self.logger.debug(f"Document status - Resume: {self.document_types['resume']}, "
                     f"Job Posting: {self.document_types['job_posting']}, "
                     f"Skills: {self.document_types['skills']}")
    
        # Determine query type and route to appropriate function
        query_lower = query.lower()
    
        try:
            if any(phrase in query_lower for phrase in ["modify cv", "modify resume", "improve resume"]):
                self.logger.info("Routing to resume improvement")
                return self.improve_resume(query)
        
            elif any(phrase in query_lower for phrase in ["missing skills", "skills gap"]):
                self.logger.info("Routing to missing skills analysis")
                return self.identify_missing_skills(query)
        
            elif any(phrase in query_lower for phrase in ["cover letter", "write a letter"]):
                self.logger.info("Routing to cover letter creation")
                return self.create_cover_letter(query)
        
            elif any(phrase in query_lower for phrase in ["interview", "prepare for interview"]):
                self.logger.info("Routing to interview preparation")
                return self.prepare_for_interview(query)
        
            else:
                self.logger.info("Routing to general question handling")
                # General question - use a simpler QA approach
                context = ""
            
                if self.resume_retriever:
                    self.logger.debug("Attempting resume retrieval")
                    resume_docs = self.resume_retriever.invoke(query)
                    if resume_docs:
                        self.logger.debug(f"Retrieved {len(resume_docs)} resume docs")
                        if hasattr(resume_docs[0], 'metadata') and 'score' in resume_docs[0].metadata:
                            scores = [doc.metadata.get('score', 'N/A') for doc in resume_docs]
                            self.logger.debug(f"Resume retrieval scores: {scores}")
                        context += "Resume information:\n" + "\n".join([doc.page_content for doc in resume_docs]) + "\n\n"
                    else:
                        self.logger.debug("No resume docs retrieved")
            
                # Similar logging for job_posting and skills retrievers...
            
                chain = self.qa_prompt | self.model | StrOutputParser()
                self.logger.debug("Executing QA chain")
                return chain.invoke({"context": context, "question": query})
    
        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {str(e)}")
            return f"An error occurred while processing your request: {str(e)}"
   
    def clear(self) -> None:
        """Clear all document stores and reset the assistant."""
        self.resume_store = None
        self.job_posting_store = None
        self.skills_store = None
       
        self.resume_retriever = None
        self.job_posting_retriever = None
        self.skills_retriever = None
       
        self.document_types = {
            "resume": False,
            "job_posting": False,
            "skills": False
        }

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