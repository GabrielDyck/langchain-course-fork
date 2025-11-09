import os

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pinecone in machine learning?"
    #chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    # Pulls a prompt template for retrieval QA chat from LangChain Hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # Creates a chain to combine documents using the LLM and the pulled prompt
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # Creates a retrieval chain using the retriever and the document combining chain
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    # Invokes the retrieval chain with the user's query and gets the result
    # The retrival_chain is a LangChain pipeline that first uses the retriever to fetch relevant documents from the vectorstore.
    # The retriever converts the query into an embedding and searches the Pinecone index for similar document embeddings.
    # The combine_docs_chain then takes these documents and uses the LLM (ChatOpenAI) with the prompt to synthesize an answer.
    result = retrival_chain.invoke(input={"input": query})

    # Prints the result from the retrieval chain
    # The result is typically a dictionary containing the answer and possibly source documents.
    print(result)

    # Defines a custom prompt template for RAG (Retrieval Augmented Generation)
    # This template instructs the LLM to use the provided context to answer the question, with specific constraints on length and style.
    # The {context} placeholder will be filled with retrieved document content, and {question} with the user's query.
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say \"thanks for asking!\" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""

    # Creates a RAG chain using the retriever, formatting function, and custom prompt
    # The RAG chain is a pipeline that:
    # 1. Uses the retriever to fetch relevant documents from the vectorstore based on the query.
    # 2. Applies format_docs to join the content of these documents into a single string for context.
    # 3. Passes the context and question to the custom prompt template (custom_rag_prompt), which formats the input for the LLM.
    # 4. The LLM (llm) generates a concise answer using the prompt and context.
    rag_chain = (
            {"context": vectorstore.as_retriever() |
                        format_docs,  # Retrieves and formats documents as context
             "question":
                 RunnablePassthrough()}  # Passes the query directly
            | custom_rag_prompt  # Formats the prompt for the LLM
            | llm  # Generates the answer
    )

    # Invokes the RAG chain with the user's query and gets the response
    # The query is processed through the chain, retrieving context, formatting the prompt, and generating the answer.
    res = rag_chain.invoke(query)
    # Prints the response from the RAG chain
    # The response is the LLM's answer, typically a string following the template's instructions.
    print(res)