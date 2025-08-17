# LLM_chatbot
This is my project in which I have made a question answering chatbot for questions related to my college IIT Bomaby. It is based on retrieval augmented generation(RAG), by integrating Retrieval-based techniques with a LLM model
<br>
1. `Web_scraping.ipynb` and `SMP_scraping-.ipynb` files contains the data scraped from various websites of our college. All the scraped data was stored in a pdf file, which I haven't include in this repo.
2. `AML_project_model.ipynb` file is the main file that contains the process of loading and combining various models/techniques, using the scraped data, which was finally integrated into a Q&A pipeline, which is able to satisfactorially answer questions (some examples included at end).
<br>

### Newly Added Work

1. Extended into an **Agentic RAG** chatbot using **LangGraph** with dynamic routing for more accurate answers.
2. Used ChromaDB for semantic retrieval and **LLaMA-2-13B-GPTQ** for efficient generation.
3. Integrated **LangSmith** for evaluation, tracing, and moderation.
4. Built a scalable scraping pipeline (20+ IITB sites) and added lightweight output moderation.
5. Provided a simple CLI (crawl, ingest, chat) for easy updates and usage.
The code for this given in the `Agentic_RAG.py` file

> **NOTE** The data I had scraped and used for training this chatbot belongs to IIT Bombay and its various organisations and bodies 
