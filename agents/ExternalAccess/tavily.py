from config import TAVILY_API_KEY
from langchain_tavily import TavilySearch


def web_search(query: str):
    search_tool = TavilySearch(k=3, tavily_api_key=TAVILY_API_KEY)
    return search_tool.run(query)

# print(web_search("What is AIRA 2")['results'][0]['url'])
# print(web_search("What is AIRA 2")['results'][0]['content'])
# print(web_search("What is AIRA 2")['results'][0]['score'])
print(web_search("What is AIRA 2")['results'])


# python -m agents.ExternalAccess.tavily