import asyncio

import mcp_server.tools.lookup
import mcp_server.tools.query_chillers
import mcp_server.tools.query_chips
import mcp_server.tools.query_cooling_towers
import mcp_server.tools.query_data_centers
import mcp_server.tools.query_installed_chips
import mcp_server.tools.query_timeline  # noqa: F401
from evals.run_eval import main

asyncio.run(main())