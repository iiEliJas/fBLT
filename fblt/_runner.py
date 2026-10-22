import json
import os
import subprocess
import sys
import threading
import time
import typing


def _stream_output(proc: subprocess.Popen[bytes]) -> None:
    stdout_stream = proc.stdout
    stderr_stream = proc.stderr
    assert stdout_stream is not None
    assert stderr_stream is not None

    def _read_stdout() -> None:
        assert stdout_stream is not None
        for line in iter(stdout_stream.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    def _read_stderr() -> None:
        assert stderr_stream is not None
        for line in iter(stderr_stream.readline, b""):
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()

    t_out = threading.Thread(target=_read_stdout)
    t_err = threading.Thread(target=_read_stderr)
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()


def run_binary(
    binary_path: str,
    argv: typing.List[str],
    *,
    capture_output: bool = False,
) -> int:
    if capture_output:
        proc = subprocess.Popen(
            [binary_path] + argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait()
    else:
        proc = subprocess.Popen(
            [binary_path] + argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _stream_output(proc)
        proc.wait()
    return proc.returncode


def tail_jsonl(path, on_event, stop_event, poll=0.25):
    pos = 0
    buf = ""
    while not stop_event.is_set():
        try:
            if not os.path.exists(path):
                time.sleep(poll)
                continue
            with open(path, "r") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except OSError:
            time.sleep(poll)
            continue
        if not chunk:
            time.sleep(poll)
            continue
        buf += chunk
        lines = buf.split("\n")
        buf = lines.pop()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                on_event(json.loads(line))
            except json.JSONDecodeError:
                buf = line + "\n" + buf
                pos -= len(line) + 1
                break


def run_binary_captured(
    binary_path: str,
    argv: typing.List[str],
) -> typing.Tuple[int, str, str]:
    proc = subprocess.Popen(
        [binary_path] + argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    return (
        proc.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )
