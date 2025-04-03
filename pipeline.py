import os
import logging
import time
import random
import sqlite3
from typing import List, Tuple
from dotenv import load_dotenv
import streamlit as st
import pymupdf4llm
import textgrad as tg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_api_key():
    if 'api_keys' in st.secrets and 'GROQ_API_KEY' in st.secrets['api_keys']:
        return st.secrets['api_keys']['GROQ_API_KEY']
    return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_api_key()
assert GROQ_API_KEY, "Please set the GROQ_API_KEY in environment variables or Streamlit secrets"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Database setup
def init_db():
    conn = sqlite3.connect("summaries.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_summary(filename: str, summary: str):
    conn = sqlite3.connect("summaries.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO summary_history (filename, summary) VALUES (?, ?)", (filename, summary))
    conn.commit()
    conn.close()

def get_summary_history() -> List[Tuple[int, str, str, str]]:
    conn = sqlite3.connect("summaries.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, summary, timestamp FROM summary_history ORDER BY timestamp DESC")
    history = cursor.fetchall()
    conn.close()
    return history

class PDFProcessor:
    @staticmethod
    def read_pdf(file_path: str) -> List[str]:
        try:
            llama_reader = pymupdf4llm.LlamaMarkdownReader()
            doc = llama_reader.load_data(file_path)
            return [page.text for page in doc]
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

class SummarizationPipeline:
    def __init__(self):
        self.model_no = 0 
        self.models = [
            "deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.1-8b-instant",
            "llama-3.2-3b-preview", "llama-3.3-70b-versatile", "mistral-saba-24b"
        ]

    def initialize_model(self):
        llm = tg.get_engine(f"groq-{self.models[self.model_no]}")
        tg.set_backward_engine(llm, override=True)
        return tg.BlackboxLLM(llm)

    def retry_with_backoff(self, func, *args, **kwargs):
        backoff_time = 5
        max_backoff_time = 60
        while True:
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.warning(f"Rate limit hit, retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time = min(max_backoff_time, backoff_time * 2 + random.uniform(0, 1))

    def process_batch(self, batch_text: str) -> tg.Variable:
        system_prompt = tg.Variable(
            value=f"Summarize this financial document: \n{batch_text}",
            requires_grad=True,
            role_description="system_prompt",
        )
        answer = self.retry_with_backoff(self.initialize_model(), system_prompt)
        self.optimize_answer(answer)
        return answer

    def optimize_answer(self, answer: tg.Variable):
        optimizer = tg.TGD(parameters=[answer])
        loss_fn = tg.TextLoss("Summarization optimization")
        loss = loss_fn(answer)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def process(self, file_path: str) -> str:
        try:
            with st.spinner("Reading and processing PDF..."):
                pages = PDFProcessor.read_pdf(file_path)
                batch_size = 3
                batches = [" ".join(pages[i:i + batch_size]) for i in range(0, len(pages), batch_size)]

                progress_bar = st.progress(0)
                batch_summaries = []
                for i, batch in enumerate(batches):
                    batch_summaries.append(self.process_batch(batch))
                    progress_bar.progress((i + 1) / len(batches))
                    self.model_no = (self.model_no + 1) % len(self.models)

                combined_text = " ".join([batch.value for batch in batch_summaries])
                final_summary = self.summarize_document(combined_text)
                return final_summary.value
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            logger.error(f"Error processing PDF: {e}")
            raise

    def summarize_document(self, text: str) -> tg.Variable:
        system_prompt = tg.Variable(
            value=f"Provide a concise summary: \n{text}",
            requires_grad=True,
            role_description="system_prompt",
        )
        final_answer = self.retry_with_backoff(self.initialize_model(), system_prompt)
        self.optimize_answer(final_answer)
        return final_answer

def main():
    st.title("PDF Summarization Tool")
    st.write("Upload a PDF file to generate a summary.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            pipeline = SummarizationPipeline()
            summary = pipeline.process(temp_path)
            st.subheader("Summary")
            st.write(summary)
            save_summary(uploaded_file.name, summary)
        finally:
            os.remove(temp_path)

    st.subheader("Summary History")
    history = get_summary_history()
    for record in history:
        st.write(f"**{record[1]}** ({record[3]})")
        st.write(record[2])
        st.markdown("---")

if __name__ == "__main__":
    main()
