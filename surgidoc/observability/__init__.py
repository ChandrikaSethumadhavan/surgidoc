"""FR-12: observability and evaluation."""
from surgidoc.observability.dashboard import render_dashboard_html, serve  # noqa: F401
from surgidoc.observability.evaluation import (ReportCase, RetrievalCase, check_thresholds, evaluate_reports,  # noqa: F401
                                               evaluate_retrieval)
from surgidoc.observability.telemetry import InstrumentedGateway, Op, Stats, Telemetry  # noqa: F401
