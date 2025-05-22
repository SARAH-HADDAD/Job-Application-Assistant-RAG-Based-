import os
import tempfile
import streamlit as st
from streamlit_chat import message
from job_assistant import JobAssistant

# Page configuration
st.set_page_config(
    page_title="AI Job Assistant",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
   
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = JobAssistant()
   
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = {
            "resume": None,
            "job_posting": None,
            "skills": None
        }

# Display chat messages
def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))

# Process user input
def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.spinner("Thinking..."):
            agent_text = st.session_state["assistant"].ask(user_text)
       
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))
        st.session_state["user_input"] = ""

# Handle file uploads
def handle_resume_upload():
    try:
        if st.session_state["resume_uploader"]:
            file = st.session_state["resume_uploader"]
            if file.size > 5*1024*1024:  # 5MB limit
                raise ValueError("File size too large (max 5MB)")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tf.write(file.getbuffer())
                file_path = tf.name
            
            with st.spinner(f"Analyzing resume: {file.name}"):
                st.session_state["assistant"].ingest_resume(file_path)
            
            os.remove(file_path)
            st.session_state["uploaded_files"]["resume"] = file.name
            st.toast(f"Resume uploaded: {file.name}", icon="✅")
    except Exception as e:
        st.error(f"Failed to process resume: {str(e)}")
        st.session_state["uploaded_files"]["resume"] = None

def handle_job_posting_upload():
    if st.session_state["job_posting_uploader"]:
        file = st.session_state["job_posting_uploader"]
        ext = os.path.splitext(file.name)[1].lower()
       
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
       
        with st.spinner(f"Analyzing job posting: {file.name}"):
            st.session_state["assistant"].ingest_job_posting(file_path)
       
        os.remove(file_path)
        st.session_state["uploaded_files"]["job_posting"] = file.name
        st.success(f"Job posting uploaded: {file.name}")

def handle_skills_upload():
    if st.session_state["skills_uploader"]:
        file = st.session_state["skills_uploader"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
       
        with st.spinner(f"Analyzing skills profile: {file.name}"):
            st.session_state["assistant"].ingest_skills_profile(file_path)
       
        os.remove(file_path)
        st.session_state["uploaded_files"]["skills"] = file.name
        st.success(f"Skills profile uploaded: {file.name}")

# Clear conversation history
def clear_conversation():
    st.session_state["messages"] = []

# Reset everything
def reset_everything():
    st.session_state["messages"] = []
    st.session_state["assistant"] = JobAssistant()
    st.session_state["uploaded_files"] = {
        "resume": None,
        "job_posting": None,
        "skills": None
    }

# Main function to render the UI
def main():
    initialize_session_state()
   
    st.title("💼 AI Job Application Assistant")
   
    # Example questions
    with st.expander("💡 Example Questions"):
        examples = [
            "How can I improve my resume for this job?",
            "What skills am I missing for this position?",
            "Generate a cover letter for this job application",
            "Help me prepare for an interview for this role"
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state["user_input"] = ex
                process_input()
                
    # Create sidebar for file uploads
    with st.sidebar:
        st.header("Upload Documents")
       
        st.subheader("Resume (PDF)")
        st.file_uploader(
            "Upload your resume",
            type=["pdf"],
            key="resume_uploader",
            on_change=handle_resume_upload,
            label_visibility="collapsed"
        )
        if st.session_state["uploaded_files"]["resume"]:
            st.success(f"✅ {st.session_state['uploaded_files']['resume']}")
       
        st.subheader("Job Posting (PDF or TXT)")
        st.file_uploader(
            "Upload job posting",
            type=["pdf", "txt"],
            key="job_posting_uploader",
            on_change=handle_job_posting_upload,
            label_visibility="collapsed"
        )
        if st.session_state["uploaded_files"]["job_posting"]:
            st.success(f"✅ {st.session_state['uploaded_files']['job_posting']}")
       
        st.subheader("Skills Profile (TXT)")
        st.file_uploader(
            "Upload skills profile",
            type=["txt"],
            key="skills_uploader",
            on_change=handle_skills_upload,
            label_visibility="collapsed"
        )
        if st.session_state["uploaded_files"]["skills"]:
            st.success(f"✅ {st.session_state['uploaded_files']['skills']}")
       
        st.divider()
       
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                clear_conversation()
        with col2:
            if st.button("Reset All"):
                reset_everything()
   
    # Chat area
    st.subheader("Chat with your Job Assistant")
   
    # Display chat messages
    display_messages()
   
    # User input
    user_input = st.text_input(
        "Ask a question about your job application...",
        key="user_input",
        on_change=process_input,
        value=st.session_state.get("user_input", "")
    )
   
    # Upload status
    col1, col2, col3 = st.columns(3)
    with col1:
        resume_status = "✅" if st.session_state["uploaded_files"]["resume"] else "❌"
        st.write(f"{resume_status} Resume")
    with col2:
        job_status = "✅" if st.session_state["uploaded_files"]["job_posting"] else "❌"
        st.write(f"{job_status} Job Posting")
    with col3:
        skills_status = "✅" if st.session_state["uploaded_files"]["skills"] else "❌"
        st.write(f"{skills_status} Skills Profile")

if __name__ == "__main__":
    main()