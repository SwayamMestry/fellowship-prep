import arxiv
import os
import json
from dotenv import load_dotenv
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import faiss

mcp = FastMCP('research') 

_ = load_dotenv()
PAPER_DIR = 'papers'

@mcp.tool()
def search_papers(topic:str, max_results:int=5) -> list:

    '''Search for papers on arXiv based on the topic and save the results'''
    pool = max(max_results*4,20)
    client = arxiv.Client()
    search = arxiv.Search(
        query = topic,
        max_results = pool,
        sort_by = arxiv.SortCriterion.Relevance
    )
    papers = list(client.results(search))

    path = os.path.join(PAPER_DIR, topic.lower().replace(' ', '_'))
    os.makedirs(path,exist_ok=True)

    file_path = os.path.join(path,'papers_info.json')

    try:
        with open(file_path,'r') as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError,json.JSONDecodeError):
        papers_info = {}
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query = model.encode([topic])
    abstract = model.encode([paper.summary for paper in papers])
    index = faiss.IndexFlatL2(abstract.shape[1])
    index.add(abstract)
    _,indices = index.search(query,max_results)
    final = []
    for i in indices[0]:
        final.append(papers[i])

    paper_ids = []
    for paper in final:
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

@mcp.resource('papers://folders')
def get_folder_names()->str:
    '''
    gives a simple list of all topic folders
    '''
    folders = []
    if os.path.exists(PAPER_DIR):
        for topic_dir in os.listdir(PAPER_DIR):
            topic_path = os.path.join(PAPER_DIR,topic_dir)
            if os.path.isdir(topic_path):
                file = os.path.join(topic_path,'papers_info.json')
                if os.path.exists(file):
                    folders.append(topic_dir)
    content = 'topics:\n'
    if folders:
        for folder in folders:
            content += f'\n->{folder}'
            content += f'Use @{folder} to access papers on that topic'
    else:
        content = '\nno topics found'
    return content

@mcp.resource('papers://{topic}')
def get_topic_papers(topic:str)->str:
    '''
    get info on papers on a specific topic
    args:
        topic: the topic to get papers on
    '''
    topic_path = topic.replace(' ','_').lower()
    file = os.path.join(PAPER_DIR,topic_path,'papers_info.json')
    if not os.path.exists(file):
        return 'no papers found on topic'
    try:
        with open(file,'r') as f:
            data = json.load(f)
        content = f"# Papers on {topic.replace('_', ' ').title()}\n\n"
        content += f"Total papers: {len(data)}\n\n"
            
        for paper_id, paper_info in data.items():
            content += f"## {paper_info['title']}\n"
            content += f"- **Paper ID**: {paper_id}\n"
            content += f"- **Authors**: {', '.join(paper_info['authors'])}\n"
            content += f"- **Published**: {paper_info['published']}\n"
            content += f"- **PDF URL**: [{paper_info['pdf_url']}]({paper_info['pdf_url']})\n\n"
            content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
            content += "---\n\n"
        return content
    except json.JSONDecodeError:
        return 'error reading json data'

@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5, detail: bool = False) -> str:
    """Generate a prompt for Claude to find and discuss academic papers on a specific topic."""
    if detail:
        return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. Follow these instructions:
        1. First, search for papers using search_papers(topic='{topic}', max_results={num_papers}).
        2. For each paper found, extract and organize the following information:
        - Paper title
        - Authors
        - Publication date
        - Brief summary of the key findings
        - Main contributions or innovations
        - Methodologies used
        - Relevance to the topic '{topic}'
        
        3. Provide a comprehensive summary that includes:
        - Overview of the current state of research in '{topic}'
        - Common themes and trends across the papers
        - Key research gaps or areas for future investigation
        - Most impactful or influential papers in this area
        
        4. Organize your findings in a clear, structured format with headings and bullet points for easy readability.
        
        Please present both detailed information about each paper and a high-level synthesis of the research landscape in {topic}."""
    else:
        return f"""Search for {num_papers} academic papers about '{topic}' using the search_papers tool. Follow these instructions: Search for papers using search_papers(topic='{topic}', max_results={num_papers}). Return a simple list of the papers found"""
if __name__ == '__main__':
    mcp.run(transport='stdio')