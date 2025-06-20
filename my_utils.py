import os
from my_config import MyConfig
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


my_config = MyConfig()

def setup_langfuse_tracer():
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = my_config.OTEL_EXPORTER_OTLP_ENDPOINT
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {my_config.LANGFUSE_AUTH}"
    # your Hugging Face token
    os.environ["HF_TOKEN"] = my_config.HF_TOKEN


    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

def hf_login():
    from huggingface_hub import login
    login(token=my_config.HF_TOKEN)