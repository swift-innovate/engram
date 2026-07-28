//! JSON-RPC stdio client — mirrors the framing used by `mcp/mod.rs`.
//!
//! Protocol: line-delimited JSON (`\n` framing, flush after each write).
//! Single-threaded and sequential: one outstanding request at a time, which
//! matches the `engram-mcp` server model. The public API is synchronous so
//! the client can be called from sync contexts (e.g. `Executor::query`), but
//! internally `stdout` is read on a dedicated background thread so a wedged
//! (not crashed, not EOF'd — just silent) child cannot block the caller
//! forever: `call_raw` waits on that thread via a bounded `recv_timeout`
//! instead of a direct blocking read.

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use serde_json::Value;

use crate::error::{AqlError, AqlResult};
use crate::mcp::protocol::{ClientRequest, ClientResponse};

/// Default ceiling on how long `call_raw` waits for a response before
/// treating the child as hung. Override with `ENGRAM_AQL_BRIDGE_TIMEOUT_MS`.
const DEFAULT_BRIDGE_TIMEOUT_MS: u64 = 30_000;

/// One line read from the child's stdout, or how the reader thread's loop
/// ended (clean EOF or an I/O error reading the pipe).
enum ReadOutcome {
    Line(String),
    Eof,
    Err(std::io::Error),
}

pub struct JsonRpcClient {
    stdin: ChildStdin,
    /// Fed by a background thread that owns the actual `BufReader<ChildStdout>`
    /// — see the module doc for why a direct blocking read isn't used here.
    rx: Receiver<ReadOutcome>,
    next_id: u64,
    timeout: Duration,
    /// Tracks whether the child is no longer usable (EOF, I/O error, or a
    /// response timeout — which leaves a stale reply that may still arrive
    /// and desync the next call). After a failed call the caller can decide
    /// to respawn; the client itself does not respawn.
    pub eof: bool,
}

impl JsonRpcClient {
    pub fn new(stdin: ChildStdin, stdout: BufReader<ChildStdout>) -> Self {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut stdout = stdout;
            loop {
                let mut line = String::new();
                let outcome = match stdout.read_line(&mut line) {
                    Ok(0) => ReadOutcome::Eof,
                    Ok(_) => ReadOutcome::Line(line),
                    Err(e) => ReadOutcome::Err(e),
                };
                let is_terminal = !matches!(outcome, ReadOutcome::Line(_));
                if tx.send(outcome).is_err() || is_terminal {
                    break;
                }
            }
        });

        let timeout = std::env::var("ENGRAM_AQL_BRIDGE_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_millis(DEFAULT_BRIDGE_TIMEOUT_MS));

        Self {
            stdin,
            rx,
            next_id: 1,
            timeout,
            eof: false,
        }
    }

    /// Override the response timeout (default: `ENGRAM_AQL_BRIDGE_TIMEOUT_MS`
    /// or `DEFAULT_BRIDGE_TIMEOUT_MS`). Mainly a testing seam — production
    /// callers configure via the env var instead of constructing this directly.
    #[cfg(test)]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Monotonically incrementing request id; call before building the request.
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Write a line to the child's stdin. Used by `child.rs` for notifications
    /// (which have no response) and internally for request framing.
    pub fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()
    }

    /// Send a JSON-RPC request and return the `result` value on success,
    /// or an `AqlError` if the response carries an `error` field or if I/O
    /// fails.
    pub fn call(&mut self, method: &str, params: Value) -> AqlResult<Value> {
        let id = self.next_id();
        let req = ClientRequest::new(id, method, params);
        self.call_raw(req)
    }

    /// Send a pre-built request and return the result value.
    ///
    /// Exposed to `child.rs` so the `initialize` call can reuse `next_id`.
    pub fn call_raw(&mut self, req: ClientRequest) -> AqlResult<Value> {
        let json = serde_json::to_string(&req)?;
        self.write_line(&json).map_err(AqlError::Io)?;

        let line = match self.rx.recv_timeout(self.timeout) {
            Ok(ReadOutcome::Line(line)) => line,
            Ok(ReadOutcome::Eof) => {
                self.eof = true;
                return Err(AqlError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "engram-mcp child closed stdout unexpectedly (EOF). \
                     The process may have crashed — check stderr for details.",
                )));
            }
            Ok(ReadOutcome::Err(e)) => {
                self.eof = true;
                return Err(AqlError::Io(e));
            }
            Err(RecvTimeoutError::Timeout) => {
                // The reader thread is still blocked on the original read and
                // may deliver a stale line later; mark unusable so the caller
                // respawns rather than risk desyncing the next call's response.
                self.eof = true;
                return Err(AqlError::Io(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "engram-mcp child did not respond within {:?} — it may be hung. \
                         Set ENGRAM_AQL_BRIDGE_TIMEOUT_MS to adjust the wait, or check \
                         engram-mcp's stderr for what it's stuck on.",
                        self.timeout
                    ),
                )));
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.eof = true;
                return Err(AqlError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "engram-mcp bridge reader thread terminated unexpectedly",
                )));
            }
        };

        let resp: ClientResponse = serde_json::from_str(line.trim())?;

        if let Some(err) = resp.error {
            return Err(AqlError::InvalidQuery(format!(
                "engram-mcp error {}: {}",
                err.code, err.message
            )));
        }

        resp.result.ok_or_else(|| {
            AqlError::InvalidQuery(
                "engram-mcp returned a response with neither result nor error".to_string(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    /// Spawn a child that stays alive but never writes to stdout, so a
    /// `call_raw` against it can only resolve via the timeout path.
    fn spawn_silent_child() -> std::process::Child {
        if cfg!(windows) {
            Command::new("cmd")
                .args(["/C", "ping -n 30 127.0.0.1 >NUL"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("failed to spawn silent test child")
        } else {
            Command::new("sh")
                .args(["-c", "sleep 30"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .expect("failed to spawn silent test child")
        }
    }

    #[test]
    fn call_raw_times_out_on_a_hung_child_instead_of_blocking_forever() {
        let mut child = spawn_silent_child();
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut client = JsonRpcClient::new(stdin, stdout).with_timeout(Duration::from_millis(150));

        let req = ClientRequest::new(client.next_id(), "initialize", serde_json::json!({}));
        let started = std::time::Instant::now();
        let result = client.call_raw(req);
        let elapsed = started.elapsed();

        let _ = child.kill();
        let _ = child.wait();

        let err = result.expect_err("a hung child must not yield a successful response");
        let msg = format!("{err}");
        assert!(
            msg.contains("did not respond") || msg.contains("hung"),
            "unexpected error message: {msg}"
        );
        assert!(
            client.eof,
            "client should mark itself unusable after a timeout"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "call_raw should return promptly on timeout, took {elapsed:?}"
        );
    }
}
