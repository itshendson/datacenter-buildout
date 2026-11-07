from agent.cli import main
from observability.otel import configure_telemetry

configure_telemetry("datacenter-agent")

main()