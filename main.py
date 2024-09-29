import asyncio
from documents import read_pdf, split_text
from graph import create_graph
from dotenv import load_dotenv

load_dotenv()
async def main():
    # Read text from the PDF file
    pdf_path = "paper.pdf"
    docs = read_pdf(pdf_path)

    # Split the text into smaller chunks
    split_docs = split_text(docs)

    # Construct the StateGraph and run it
    graph = create_graph()
    app = graph.compile()

    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):
        print(list(step.keys()))
    print(step)

if __name__ == "__main__":
    asyncio.run(main())
