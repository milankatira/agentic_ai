from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Weather")

@mcp.tool()

async def get_weather(location:str)->str:
    """ Get weather for a location """
    return f"Weather in {location} is always sunny"

if __name__=="__main__":
    mcp.run(transport="streamable-http")
