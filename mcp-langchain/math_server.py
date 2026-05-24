from mcp.server.fastmcp import FastMCP

## tool name
mcp=FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """ Add two numbers """
    return a+b



@mcp.tool()
def multiply(a:int,b:int)->int:
    """ Multiply two numbers """
    return a*b


# This block ensures that the server only runs when the script is executed directly (not when imported).
# It starts the MCP server using the 'stdio' transport, which means it will communicate with clients
# by reading from standard input and writing to standard output. This is typical for local integrations.
if __name__=="__main__":
    mcp.run(transport="stdio")

