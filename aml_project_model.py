# -*- coding: utf-8 -*-
"""AML_project_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18SohuHrXSJoSnmCDjDo4rC9uiv4hWvbr

# EE782: AML Project
# IITB SmartBot

Dhatri Mehta 210070027</br>
Devesh Soni 21D070025</br>
Shivansh Gupta 21D070067

## Packages and Libraries
"""

## This was just to check the status and usage of compute unit of GPU
!nvidia-smi

# This code installs specific versions of various Python packages using pip, suppressing progress bars for a cleaner output

!pip install -Uqqq pip --progress-bar off
!pip install -qqq torch==2.0.1 --progress-bar off
!pip install -qqq transformers==4.31.0 --progress-bar off
!pip install -qqq langchain==0.0.266 --progress-bar off
!pip install -qqq chromadb==0.4.5 --progress-bar off
!pip install -qqq pypdf==3.15.0 --progress-bar off
!pip install -qqq xformers==0.0.20 --progress-bar off
!pip install -qqq sentence_transformers==2.2.2 --progress-bar off
!pip install -qqq InstructorEmbedding==1.0.1 --progress-bar off
!pip install -qqq pdf2image==1.16.3 --progress-bar off

"""Binary wheel files in Python are pre-built distributions of a package that can be easily installed on a compatible system without the need for compilation. They play a crucial role in simplifying the installation process and improving the efficiency of package distribution. For our model we are downloading a AUTOGPTQ BWF file package"""

#using the wget command to download a binary wheel file
#The file being downloaded is "auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl,"  related to a AutoGPTQ model

!wget -q https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.1/auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl

!pip install -qqq auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl --progress-bar off

!apt-get update

!apt-get install poppler-utils

!sudo apt-get install poppler-utils

## importing various libraries required for our LLM
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

"""## Data

### The below code show how our pdf looks like
"""

scraped_data_images = convert_from_path("/content/Scraped_data/Final_data.pdf", dpi=88)
scraped_data_images[0]

!rm -rf "db"

## loading the pdf file
loader = PyPDFDirectoryLoader("/content/Scraped_data")
docs = loader.load()
len(docs)                        # number of pages in the pdf

##  this code is setting up an embedding generator using the Hugging Face "instructor-large" model
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
)

"""This code is using a recursive character-based text splitter to divide a collection of documents into smaller chunks of text
One chunk contains 1024 words and each chunk has an overlap of 64 words with successive and predecessor chunk. This help to form a proper connection between differnet chunks helping the model to learn efficiently.
"""

# this code is using a recursive character-based text splitter to divide a collection of documents into smaller chunks of text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(docs)
len(texts)

"""**Chroma** is an open-source embedding database that is used to store and manage embeddings. </br>
This code creates a Chroma object by calling the from_documents method. The method takes texts (a list of text chunks created above), embeddings (hkunlp/HuggingFace), and persist_directory="db"
"""

db = Chroma.from_documents(texts, embeddings, persist_directory="db")

"""## Llama 2 13B

This code snippet is initializing a language model using the Hugging Face Transformers library</br>
The pre-trained language model we are using is **TheBloke/Llama-2**-13B-chat-GPTQ.</br>
The model variable now holds an instance of the **GPTQ language model** ready for use for our Q&A tasks.
"""

model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "model"

#  initializes a tokenizer using the AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# This initializes the GPTQ (Quantized GPT) language model using the AutoGPTQForCausalLM class
# The from_quantized method is used to load a quantized version of the model, which we have downloaded above as a binary wheel file
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    revision="gptq-4bit-128g-actorder_True",
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    device=DEVICE,
    quantize_config=None,
)

## This was just to check the status and usage of compute unit of GPU
!nvidia-smi

"""This is **System Prompt** structure which we have designed for use in a context to guide the behavior of our language model with specific instructions while taking into account user prompts."""

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()

"""The **TextStreamer** class is a utility class from the transformers library that is used to stream the output of a language model during generation. This can be useful for applications where you want to display the generated text as it is being produced, or for collecting the output of the model without waiting for the entire generation to finish."""

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

## This code is creating a text generation pipeline
text_pipeline = pipeline(
    "text-generation",
    model=model,                                                        # pre-trained language model to be used for text generation
    tokenizer=tokenizer,                                                # tokenizer defined above
    max_new_tokens=1024,                                                # Sets the maximum number of tokens that can be generated as output
    temperature=0,                                                      # Temperature is a hyperparameter that controls the randomness of the generated text.
                                                                           # A temperature of 0 results in deterministic output, meaning the most likely token is always chosen.
    top_p=0.95,                                                         # controls the diversity of the generated output
    repetition_penalty=1.15,                                            # Repetition penalty discourages the model from repeating the same tokens in its output
    streamer=streamer,
)

# creating an instance of the HuggingFacePipeline class, for a pipeline for text generation (text_pipeline) created above
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# This code is creating a RetrievalQA object named qa_chain using the question-answering pipeline (llm)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

"""## Chat with our SmartBOT"""

result = qa_chain("hello how are you")

result = qa_chain("I have some questions, can I ask you?")

result = qa_chain("What is SMP?")

result = qa_chain("How can I book a room in guest house?")

result = qa_chain("What are the different scholarships provided at iit Bomaby?")

result = qa_chain("How can I calculate CPI and SPI?")

result = qa_chain("what is Mood Indigo and what all should I need to know to attend it?")

result = qa_chain("When is Mood Indigo?")

result = qa_chain(
    "what is Semester Excahnge?"
)

result = qa_chain("Who is GSHA and can you provide his contact?")

result = qa_chain("I'm hungry, Where can I get a pizza?")

"""## References

https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ </br>
https://huggingface.co/hkunlp/instructor-large </br>
https://www.mlexpert.io/prompt-engineering/private-gpt4all#create-chain
"""