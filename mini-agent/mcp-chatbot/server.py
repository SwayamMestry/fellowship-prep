import arxiv
import os
import json
from dotenv import load_dotenv
from fastmcp import FastMCP

mcp = FastMCP('research') 

_ = load_dotenv()
PAPER_DIR = 'papers'

@mcp.tool()
def search_papers(topic:str, max_results:int=5) -> list:

    '''Search for papers on arXiv based on the topic and save the results'''

    client = arxiv.Client()
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )
    papers = client.results(search)

    path = os.path.join(PAPER_DIR, topic.lower().replace(' ', '_'))
    os.makedirs(path,exist_ok=True)

    file_path = os.path.join(path,'papers_info.json')

    try:
        with open(file_path,'r') as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError,json.JSONDecodeError):
        papers_info = {}
    
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title' : paper.title,
            'authors' : [author.name for author in paper.authors],
            'summary' : paper.summary,
            'pdf_url' : paper.pdf_url,
            'published' : str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

    with open(file_path,'w') as json_file:
        json.dump(papers_info, json_file, indent = 2)
    
    print(f'results saved at {file_path}')

    return paper_ids

@mcp.tool() 
def extract_info(paper_id:str)->str:

    '''search for information about a specific paper in all topic directories'''

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR,item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path,'papers_info.json')
            if os.path.isfile(file_path):
                try:
                    with open(file_path,'r') as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id],indent = 2)
                except (FileNotFoundError,json.JSONDecodeError) as e:
                    print(f'Error loading {file_path} : {str(e)}')
                    continue
    return f'Paper {paper_id} does not exist'

if __name__ == '__main__':
    mcp.run(transport='stdio')