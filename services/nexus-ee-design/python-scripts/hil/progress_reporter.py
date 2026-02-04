"""
Progress Reporter for HIL Testing

This module provides utilities for reporting progress from Python scripts
to the Node.js backend via stdout in the PROGRESS: JSON format.
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class HILEventType(Enum):
    """HIL WebSocket event types."""
    # Instrument events
    INSTRUMENT_DISCOVERED = "instrument_discovered"
    INSTRUMENT_CONNECTED = "instrument_connected"
    INSTRUMENT_DISCONNECTED = "instrument_disconnected"
    INSTRUMENT_ERROR = "instrument_error"
    INSTRUMENT_STATUS_CHANGED = "instrument_status_changed"

    # Test run events
    TEST_RUN_QUEUED = "test_run_queued"
    TEST_RUN_STARTED = "test_run_started"
    TEST_STEP_STARTED = "test_step_started"
    TEST_STEP_COMPLETED = "test_step_completed"
    TEST_MEASUREMENT = "test_measurement"
    TEST_RUN_PROGRESS = "test_run_progress"
    TEST_RUN_COMPLETED = "test_run_completed"
    TEST_RUN_FAILED = "test_run_failed"
    TEST_RUN_ABORTED = "test_run_aborted"

    # Data streaming events
    WAVEFORM_CHUNK = "waveform_chunk"
    LOGIC_TRACE_CHUNK = "logic_trace_chunk"
    LIVE_MEASUREMENT = "live_measurement"
    CAPTURE_STARTED = "capture_started"
    CAPTURE_COMPLETE = "capture_complete"

    # Control events
    MOTOR_STATE_CHANGED = "motor_state_changed"
    POWER_STATE_CHANGED = "power_state_changed"
    FAULT_DETECTED = "fault_detected"
    EMERGENCY_STOP = "emergency_stop"

    # Final states
    COMPLETE = "complete"
    ERROR = "error"


class HILPhase(Enum):
    """HIL operation phases."""
    DISCOVERY = "discovery"
    CONFIGURATION = "configuration"
    SETUP = "setup"
    MEASUREMENT = "measurement"
    CAPTURE = "capture"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    TEARDOWN = "teardown"


@dataclass
class ProgressEvent:
    """Progress event to be emitted via stdout."""
    type: str
    progress_percentage: float
    current_step: str
    phase: Optional[str] = None
    phase_progress: Optional[float] = None
    data: Optional[Dict[str, Any]] = None

    # HIL-specific fields
    hil_event_type: Optional[str] = None
    hil_phase: Optional[str] = None
    instrument_id: Optional[str] = None
    test_run_id: Optional[str] = None
    step_index: Optional[int] = None
    total_steps: Optional[int] = None
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    measurement: Optional[Dict[str, Any]] = None
    waveform_chunk: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        result["timestamp"] = datetime.utcnow().isoformat() + "Z"
        return result


class ProgressReporter:
    """
    Reporter for emitting progress events to the Node.js backend.

    Progress events are written to stdout in the format:
    PROGRESS:{"type":"...", "progress_percentage":50, ...}

    The Node.js backend parses these lines and forwards them via WebSocket.
    """

    def __init__(
        self,
        operation_id: Optional[str] = None,
        test_run_id: Optional[str] = None,
        total_steps: Optional[int] = None,
    ):
        """
        Initialize the progress reporter.

        Args:
            operation_id: Unique operation ID for this test run
            test_run_id: Database test run ID
            total_steps: Total number of steps in the sequence
        """
        self.operation_id = operation_id
        self.test_run_id = test_run_id
        self.total_steps = total_steps
        self._current_step_index = 0
        self._current_phase = HILPhase.SETUP

    def emit(self, event: ProgressEvent) -> None:
        """
        Emit a progress event via stdout.

        Args:
            event: Progress event to emit
        """
        event_dict = event.to_dict()
        if self.test_run_id:
            event_dict["test_run_id"] = self.test_run_id

        try:
            json_str = json.dumps(event_dict, default=str)
            print(f"PROGRESS:{json_str}", flush=True)
        except Exception as e:
            logger.error(f"Failed to emit progress: {e}")

    def emit_dict(self, event_dict: Dict[str, Any]) -> None:
        """
        Emit a raw dictionary as progress event.

        Args:
            event_dict: Event dictionary
        """
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if self.test_run_id:
            event_dict["test_run_id"] = self.test_run_id

        try:
            json_str = json.dumps(event_dict, default=str)
            print(f"PROGRESS:{json_str}", flush=True)
        except Exception as e:
            logger.error(f"Failed to emit progress: {e}")

    def set_phase(self, phase: HILPhase) -> None:
        """Set the current operation phase."""
        self._current_phase = phase
        self.emit(ProgressEvent(
            type="phase_changed",
            hil_phase=phase.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Entering {phase.value} phase",
        ))

    def _calculate_progress(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_steps and self.total_steps > 0:
            return (self._current_step_index / self.total_steps) * 100
        return 0.0

    # ==========================================================================
    # Instrument Events
    # ==========================================================================

    def instrument_discovered(
        self,
        instrument_id: str,
        manufacturer: str,
        model: str,
        instrument_type: str,
    ) -> None:
        """Report instrument discovery."""
        self.emit(ProgressEvent(
            type="instrument_discovered",
            hil_event_type=HILEventType.INSTRUMENT_DISCOVERED.value,
            hil_phase=HILPhase.DISCOVERY.value,
            progress_percentage=0,
            current_step=f"Discovered {manufacturer} {model}",
            instrument_id=instrument_id,
            data={
                "manufacturer": manufacturer,
                "model": model,
                "instrument_type": instrument_type,
            },
        ))

    def instrument_connected(
        self,
        instrument_id: str,
        name: str,
    ) -> None:
        """Report instrument connection."""
        self.emit(ProgressEvent(
            type="instrument_connected",
            hil_event_type=HILEventType.INSTRUMENT_CONNECTED.value,
            progress_percentage=0,
            current_step=f"Connected to {name}",
            instrument_id=instrument_id,
        ))

    def instrument_disconnected(
        self,
        instrument_id: str,
        reason: Optional[str] = None,
    ) -> None:
        """Report instrument disconnection."""
        self.emit(ProgressEvent(
            type="instrument_disconnected",
            hil_event_type=HILEventType.INSTRUMENT_DISCONNECTED.value,
            progress_percentage=0,
            current_step=reason or "Instrument disconnected",
            instrument_id=instrument_id,
        ))

    def instrument_error(
        self,
        instrument_id: str,
        error_message: str,
    ) -> None:
        """Report instrument error."""
        self.emit(ProgressEvent(
            type="instrument_error",
            hil_event_type=HILEventType.INSTRUMENT_ERROR.value,
            progress_percentage=0,
            current_step=f"Instrument error: {error_message}",
            instrument_id=instrument_id,
            error={"code": "INSTRUMENT_ERROR", "message": error_message},
        ))

    # ==========================================================================
    # Test Run Events
    # ==========================================================================

    def test_started(self) -> None:
        """Report test run started."""
        self._current_phase = HILPhase.SETUP
        self.emit(ProgressEvent(
            type="test_run_started",
            hil_event_type=HILEventType.TEST_RUN_STARTED.value,
            hil_phase=HILPhase.SETUP.value,
            progress_percentage=0,
            current_step="Test run started",
        ))

    def step_started(
        self,
        step_index: int,
        step_id: str,
        step_name: str,
    ) -> None:
        """Report test step started."""
        self._current_step_index = step_index
        self.emit(ProgressEvent(
            type="test_step_started",
            hil_event_type=HILEventType.TEST_STEP_STARTED.value,
            hil_phase=self._current_phase.value,
            progress_percentage=self._calculate_progress(),
            current_step=step_name,
            step_index=step_index,
            total_steps=self.total_steps,
            step_id=step_id,
            step_name=step_name,
        ))

    def step_completed(
        self,
        step_index: int,
        step_id: str,
        step_name: str,
        passed: bool,
        measurement_count: int = 0,
    ) -> None:
        """Report test step completed."""
        self._current_step_index = step_index + 1
        self.emit(ProgressEvent(
            type="test_step_completed",
            hil_event_type=HILEventType.TEST_STEP_COMPLETED.value,
            hil_phase=self._current_phase.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Step '{step_name}' {'passed' if passed else 'failed'}",
            step_index=step_index,
            total_steps=self.total_steps,
            step_id=step_id,
            step_name=step_name,
            data={"passed": passed, "measurement_count": measurement_count},
        ))

    def progress(
        self,
        percentage: float,
        message: str,
        phase: Optional[HILPhase] = None,
    ) -> None:
        """Report general progress update."""
        if phase:
            self._current_phase = phase
        self.emit(ProgressEvent(
            type="test_run_progress",
            hil_event_type=HILEventType.TEST_RUN_PROGRESS.value,
            hil_phase=self._current_phase.value,
            progress_percentage=percentage,
            current_step=message,
        ))

    def test_completed(
        self,
        result: str,
        total_measurements: int,
        passed_measurements: int,
        failed_measurements: int,
        duration_ms: int,
    ) -> None:
        """Report test run completed."""
        self._current_phase = HILPhase.TEARDOWN
        self.emit(ProgressEvent(
            type="test_run_completed",
            hil_event_type=HILEventType.TEST_RUN_COMPLETED.value,
            hil_phase=HILPhase.TEARDOWN.value,
            progress_percentage=100,
            current_step=f"Test run {result}",
            data={
                "result": result,
                "summary": {
                    "total_measurements": total_measurements,
                    "passed_measurements": passed_measurements,
                    "failed_measurements": failed_measurements,
                    "duration_ms": duration_ms,
                },
            },
        ))

    def test_failed(
        self,
        error_message: str,
        step_id: Optional[str] = None,
    ) -> None:
        """Report test run failed."""
        self.emit(ProgressEvent(
            type="test_run_failed",
            hil_event_type=HILEventType.TEST_RUN_FAILED.value,
            hil_phase=self._current_phase.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Test failed: {error_message}",
            step_id=step_id,
            error={"code": "TEST_FAILED", "message": error_message},
        ))

    # ==========================================================================
    # Measurement Events
    # ==========================================================================

    def live_measurement(
        self,
        measurement_type: str,
        value: float,
        unit: str,
        channel: Optional[str] = None,
        passed: Optional[bool] = None,
    ) -> None:
        """Report a live measurement."""
        self.emit(ProgressEvent(
            type="live_measurement",
            hil_event_type=HILEventType.LIVE_MEASUREMENT.value,
            hil_phase=HILPhase.MEASUREMENT.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"{measurement_type}: {value}{unit}",
            measurement={
                "type": measurement_type,
                "value": value,
                "unit": unit,
                "channel": channel,
                "passed": passed,
            },
        ))

    def measurement_batch(
        self,
        measurements: List[Dict[str, Any]],
    ) -> None:
        """Report a batch of measurements."""
        self.emit_dict({
            "type": "measurement_batch",
            "hil_event_type": "MEASUREMENT_BATCH",
            "progress_percentage": self._calculate_progress(),
            "current_step": f"Recorded {len(measurements)} measurements",
            "data": {"measurements": measurements, "count": len(measurements)},
        })

    # ==========================================================================
    # Capture Events
    # ==========================================================================

    def capture_started(
        self,
        capture_id: str,
        capture_type: str,
        channels: List[str],
    ) -> None:
        """Report capture started."""
        self._current_phase = HILPhase.CAPTURE
        self.emit(ProgressEvent(
            type="capture_started",
            hil_event_type=HILEventType.CAPTURE_STARTED.value,
            hil_phase=HILPhase.CAPTURE.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Starting {capture_type} capture ({len(channels)} channels)",
            data={"capture_id": capture_id, "capture_type": capture_type, "channels": channels},
        ))

    def waveform_chunk(
        self,
        capture_id: str,
        channels: List[str],
        start_index: int,
        sample_count: int,
        data: List[List[float]],
        is_last: bool = False,
    ) -> None:
        """Report waveform data chunk."""
        self.emit(ProgressEvent(
            type="waveform_chunk",
            hil_event_type=HILEventType.WAVEFORM_CHUNK.value,
            hil_phase=HILPhase.CAPTURE.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Streaming waveform data ({sample_count} samples)",
            waveform_chunk={
                "capture_id": capture_id,
                "channels": channels,
                "start_index": start_index,
                "sample_count": sample_count,
                "data": data,
                "is_last": is_last,
            },
        ))

    def capture_complete(
        self,
        capture_id: str,
        total_samples: int,
        analysis_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report capture complete."""
        self._current_phase = HILPhase.ANALYSIS
        self.emit(ProgressEvent(
            type="capture_complete",
            hil_event_type=HILEventType.CAPTURE_COMPLETE.value,
            hil_phase=HILPhase.ANALYSIS.value,
            progress_percentage=self._calculate_progress(),
            current_step="Waveform capture complete",
            data={
                "capture_id": capture_id,
                "total_samples": total_samples,
                "analysis_results": analysis_results,
            },
        ))

    # ==========================================================================
    # Control Events
    # ==========================================================================

    def motor_state(
        self,
        speed: float,
        torque: float,
        position: float,
        direction: str,
        fault_code: Optional[int] = None,
    ) -> None:
        """Report motor state change."""
        self.emit(ProgressEvent(
            type="motor_state_changed",
            hil_event_type=HILEventType.MOTOR_STATE_CHANGED.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Motor: {speed:.0f} RPM, {direction}",
            data={
                "motor_state": {
                    "speed": speed,
                    "torque": torque,
                    "position": position,
                    "direction": direction,
                    "fault_code": fault_code,
                }
            },
        ))

    def power_state(
        self,
        channel: str,
        voltage: float,
        current: float,
        enabled: bool,
        mode: str,
    ) -> None:
        """Report power supply state change."""
        self.emit(ProgressEvent(
            type="power_state_changed",
            hil_event_type=HILEventType.POWER_STATE_CHANGED.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Power {channel}: {voltage:.2f}V / {current:.3f}A",
            data={
                "power_state": {
                    "channel": channel,
                    "voltage": voltage,
                    "current": current,
                    "enabled": enabled,
                    "mode": mode,
                }
            },
        ))

    def fault_detected(
        self,
        code: str,
        message: str,
        severity: str,
        instrument_id: Optional[str] = None,
    ) -> None:
        """Report fault detected."""
        self.emit(ProgressEvent(
            type="fault_detected",
            hil_event_type=HILEventType.FAULT_DETECTED.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Fault: {message}",
            instrument_id=instrument_id,
            data={
                "fault": {
                    "code": code,
                    "message": message,
                    "severity": severity,
                    "instrument_id": instrument_id,
                }
            },
        ))

    def emergency_stop(self, reason: str) -> None:
        """Report emergency stop."""
        self.emit(ProgressEvent(
            type="emergency_stop",
            hil_event_type=HILEventType.EMERGENCY_STOP.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"EMERGENCY STOP: {reason}",
            error={"code": "EMERGENCY_STOP", "message": reason},
        ))

    # ==========================================================================
    # Final Events
    # ==========================================================================

    def complete(
        self,
        result: Dict[str, Any],
    ) -> None:
        """Report operation complete."""
        self.emit(ProgressEvent(
            type="complete",
            hil_event_type=HILEventType.COMPLETE.value,
            progress_percentage=100,
            current_step="HIL operation complete",
            data={"result": result},
        ))

    def error(
        self,
        error_message: str,
        error_code: str = "OPERATION_FAILED",
        partial_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report operation error."""
        self.emit(ProgressEvent(
            type="error",
            hil_event_type=HILEventType.ERROR.value,
            progress_percentage=self._calculate_progress(),
            current_step=f"Error: {error_message}",
            error={"code": error_code, "message": error_message},
            data={"partial_result": partial_result} if partial_result else None,
        ))


def emit_progress(
    event_type: str,
    progress: float,
    message: str,
    **kwargs,
) -> None:
    """
    Convenience function to emit a single progress event.

    Args:
        event_type: Event type string
        progress: Progress percentage (0-100)
        message: Current step message
        **kwargs: Additional event fields
    """
    event_dict = {
        "type": event_type,
        "progress_percentage": progress,
        "current_step": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **kwargs,
    }

    try:
        json_str = json.dumps(event_dict, default=str)
        print(f"PROGRESS:{json_str}", flush=True)
    except Exception as e:
        logger.error(f"Failed to emit progress: {e}")
