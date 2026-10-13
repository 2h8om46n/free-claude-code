"""Incremental framing for Anthropic-compatible SSE streams."""

from ..stream_contracts import SSEEvent, parse_sse_text


class AnthropicSSEDecoder:
    """Decode split SSE text, optionally filtering named events before JSON parsing."""

    def __init__(self, *, event_names: frozenset[str] | None = None) -> None:
        self._event_names = event_names
        self._parts: list[str] = []
        self._boundary_tail = ""

    def feed(self, chunk: str) -> tuple[SSEEvent, ...]:
        """Consume one wire chunk and return its selected complete events."""

        frames = self.feed_frames(chunk)
        if not frames:
            return ()
        events: list[SSEEvent] = []
        event_names = self._event_names
        for raw in frames:
            events.extend(parse_sse_text(raw, event_names=event_names))
        return tuple(events)

    def feed_frames(self, chunk: str) -> tuple[str, ...]:
        """Retain original framing, including comments and unknown SSE fields."""

        if chunk and "\n" not in chunk and "\r" not in chunk:
            self._parts.append(chunk)
            self._boundary_tail = (self._boundary_tail + chunk)[-3:]
            return ()

        probe = self._boundary_tail + chunk
        if "\n\n" not in probe and "\n\r" not in probe and "\r\r" not in probe:
            if chunk:
                self._parts.append(chunk)
            self._boundary_tail = probe[-3:]
            return ()

        frames: list[str] = []
        prefix_length = len(self._boundary_tail)
        chunk_start = 0
        search_start = 0
        while True:
            lf_lf = probe.find("\n\n", search_start)
            lf_cr = probe.find("\n\r", search_start)
            cr_cr = probe.find("\r\r", search_start)
            boundary = probe_len = len(probe)
            if lf_lf != -1 and lf_lf < boundary:
                boundary = lf_lf
            if lf_cr != -1 and lf_cr < boundary:
                boundary = lf_cr
            if cr_cr != -1 and cr_cr < boundary:
                boundary = cr_cr
            if boundary == probe_len:
                break
            if probe[boundary + 1] == "\n":
                end = boundary + 2
            elif boundary + 2 < probe_len and probe[boundary + 2] == "\n":
                end = boundary + 3
            else:
                end = boundary + 2
            chunk_end = end - prefix_length
            if self._parts:
                self._parts.append(chunk[chunk_start:chunk_end])
                raw = "".join(self._parts)
                self._parts.clear()
            else:
                raw = chunk[chunk_start:chunk_end]
            frames.append(raw)
            chunk_start = chunk_end
            search_start = end

        remainder = chunk[chunk_start:]
        if remainder:
            self._parts.append(remainder)
        if chunk_start:
            self._boundary_tail = remainder[-3:]
        else:
            self._boundary_tail = (self._boundary_tail + chunk)[-3:]
        return tuple(frames)

    def finish(self) -> tuple[SSEEvent, ...]:
        """Return a final unterminated event, if one is present."""

        events: list[SSEEvent] = []
        event_names = self._event_names
        for raw in self.finish_frames():
            events.extend(parse_sse_text(raw, event_names=event_names))
        return tuple(events)

    def finish_frames(self) -> tuple[str, ...]:
        """Return the original trailing frame and clear framing state."""

        remainder = "".join(self._parts)
        self._parts.clear()
        self._boundary_tail = ""
        if not remainder:
            return ()
        return (remainder,)
