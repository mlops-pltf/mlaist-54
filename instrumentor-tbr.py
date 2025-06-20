from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

def instrument():
    register()
    SmolagentsInstrumentor().instrument()