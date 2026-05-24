from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client=MultiServerMCPClient({
        "math":{
            "command":"python",
            "args":["math_server.py"], ## ensure correct absolute path
            "transport":"stdio"
        },
        "weather":{
            "url":"http://localhost:8000/mcp",
            "transport":"streamable_http"
        }
    })

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools=await client.get_tools()
    model =ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0
    )

    agent=create_react_agent(model, tools, prompt="You are a helpful assistant")

    math_response=await agent.ainvoke({"messages": [{"role":"user", "content": "what's (3+5)*7?"}]})
    print("\n\nMath Response: \n\n", math_response['messages'][-1].content)

    weather_response=await agent.ainvoke({"messages": [{"role":"user", "content": "what is the weather in San Francisco?"}]})
    print("\n\nWeather Response: \n\n", weather_response['messages'][-1].content)

asyncio.run(main())
