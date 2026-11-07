import mcp_server.tools.lookup
import mcp_server.tools.query_chillers
import mcp_server.tools.query_chips
import mcp_server.tools.query_cooling_towers
import mcp_server.tools.query_data_centers
import mcp_server.tools.query_installed_chips
import mcp_server.tools.query_timeline
from mcp_server.server import mcp
from observability.otel import configure_telemetry

configure_telemetry("epoch-ai-datacenters")

mcp.run()