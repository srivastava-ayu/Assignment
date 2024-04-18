import re
import os
import pickle
import PyPDF2
import nltk
from openai import OpenAI
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# nltk.download('punkt') # Download the 'punkt' package for tokenization

load_dotenv()  # Load environment variables from .env file

GPT_35_TURBO = 'gpt-3.5-turbo'
GPT_4 = 'gpt-4'

# Prompt
PROMPT_TEMPLATE = """
            Use the following pieces of context to answer the question at the end.
            Work like a Resume ATS. 
            Always reply as a person. NEVER BREAK THE CHARACTER!
            Don't try to make up an answer. 
            {context}
            Question: {question}
            chat_history: {chat_history}
        """

QA_CHAIN_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Resume_Matcher:
    def __init__(self):
        # self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        # sentence-transformers/all-MiniLM-L6-v2
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embedding(self, text):
        response = client.embeddings.create(input=text,
        model="text-embedding-ada-002")
        embedding = response.data[0].embedding
        return np.array(embedding)

    def get_embedding_st(self, text):
        return self.model.encode(text)


    def preprocess_text(self, text):
        """Preprocesses text by lowercasing, removing punctuation, and tokenizing."""
        text = text.lower()  # Lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = nltk.word_tokenize(text)  # Tokenize
        return tokens

    def extract_text_from_pdf(self, pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

    def calculate_similarity(self, embedding1, embedding2):
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity

    def calculate_similarity_st(self, embedding1, embedding2):
        # Use cosine similarity from SentenceTransformers util
        return util.cos_sim(embedding1, embedding2).item()
        # return scores

    def euclidean_distance(self, embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)

    def get_match_score_and_explanation(self, resume_text, job_description_text):
        # Preprocess text
        # resume_tokens = self.preprocess_text(resume_text)
        # job_description_tokens = self.preprocess_text(job_description_text)

        # Generate embeddings
        resume_embedding = self.get_embedding(resume_text)

        job_description_embedding = self.get_embedding(job_description_text)

        # Calculate similarity
        similarity_score = self.calculate_similarity(resume_embedding, job_description_embedding)

        excellent_match_threshold = 0.8
        good_match_threshold = 0.6
        fair_match_threshold = 0.4

        # Determine match level and generate explanation
        if similarity_score >= excellent_match_threshold or similarity_score >= good_match_threshold:
            match_level = "Excellent Match"
            explanation = "The candidate possesses a strong alignment with the required skills and experience."
        elif similarity_score >= good_match_threshold and similarity_score <= excellent_match_threshold:
            match_level = "Good Match"
            explanation = "The candidate demonstrates a good fit for the role, with some areas for potential development."
        elif similarity_score >= fair_match_threshold and similarity_score <=good_match_threshold:
            match_level = "Fair Match"
            explanation = "The candidate shows some relevant qualifications, but may require additional training or experience."
        else:
            match_level = "Poor Match"
            explanation = "The candidate's skills and experience do not significantly align with the job requirements."

        return match_level, explanation, similarity_score

def main():
    st.title("Resume-Job Description Matcher")
    matcher = Resume_Matcher()  # Create an instance of the class
    keywords="No Matching Skills"
    # Upload Resume and Job Description
    resume_file = st.sidebar.file_uploader("Upload Resume's (PDF)", accept_multiple_files=True, type=["pdf"])
    jd_file = st.sidebar.file_uploader("Upload JD's (PDF)", accept_multiple_files=True, type=["pdf"])

    if resume_file and jd_file:
        # Extract text from PDFs
        st.subheader("Job Description")
        for i, (resume, jd) in enumerate(zip(resume_file, jd_file)):  # Loop through each resume_file and jd file:
            resume_text = matcher.extract_text_from_pdf(resume) 
            jd_text = matcher.extract_text_from_pdf(jd)        
        # text_resume = matcher.__preprocess(resume_text) 
        # text_jd = matcher.__preprocess(jd_text)
            with st.expander("View Job Description"):
                st.write(jd_text)

        # Calculate match score and explanation
            match_level, explanation, similarity_score = matcher.get_match_score_and_explanation(resume_text, jd_text)

            if similarity_score >=0.7:
                text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len
            )

        # Generate the chunks
                chunks = text_splitter.split_text(text=resume_text)
                store_name = f"faiss_store_{i}"  # Change the store name to include the resume index

                if os.path.exists(store_name):
                    vector_store = FAISS.load_local(store_name, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
                else:
                    embeddings = OpenAIEmbeddings()
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    vector_store.save_local(store_name)

            # We are ready with the embeddings and the knowledge-base - We are basically ready to accept the questions
            # We want to make sure that these answers are only generated from the text from my resume
            # this can be done by giving it prompt to work with

                llm = ChatOpenAI(model_name=GPT_4, temperature=0)

                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                # create a chain to answer questions
                # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                qa = ConversationalRetrievalChain.from_llm(
                    llm,
                    retriever,
                    verbose=True,
                    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
                    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                    # memory=memory
                )

                question= f"given the {jd_text} find ALL the Matching skills, Missing Skills, Social Skills and experience FOR EACH RESUME and tell if Resume 1 or Resume 2 is a better match and assign a Score based on these in the format:  Matching Skills: <Number of matched skills>, Missing Skills: <Number of missing skills>, Social Skills: <Number of social skills>, Experience: <>, Score: <Bases on these>"
                chat_history = []  # or a list of previous conversations
                result = qa({"question": question, "chat_history": chat_history})
                # chat_history = [*chat_history, (question, result["answer"])]
                keywords = result["answer"]

        # Display results
        st.subheader("Match Results")
        st.write(f"**Match Level:** {match_level}")
        st.write(f"**Explanation:** {explanation}")
        st.write(f"**Similarity Score:** {similarity_score:.4f}")
        st.write(f"**Matching Skills:** {keywords}")

if __name__ == "__main__":
    main()