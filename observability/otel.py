"""Configure OpenTelemetry for this process."""

import logging
import os

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_telemetry(service_name: str) -> None:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    resource = Resource.create({
        "service.name": service_name,
    })

    # Tracer
    tracer_provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter()
    processor = BatchSpanProcessor(exporter)

    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    # Logging
    log_exporter = OTLPLogExporter()
    log_processor = BatchLogRecordProcessor(log_exporter)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(log_processor)
    set_logger_provider(logger_provider)

    logging.getLogger().addHandler(
        LoggingHandler(logger_provider=logger_provider)
    )
