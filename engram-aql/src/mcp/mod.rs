//! MCP stdio server.
//!
//! Hand-rolled JSON-RPC over stdin/stdout. Exposes one tool (`engram_aql`)
//! that agents can call to run AQL read queries against the `.engram` file.

pub mod handlers;
pub mod protocol;

use std::path::Path;

use anyhow::Result;
use tokio::io::{self as tokio_io, AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::executor::Executor;
use protocol::{JsonRpcRequest, JsonRpcResponse};

pub async fn run(db_path: &Path) -> Result<()> {
    let exec = Executor::open(db_path)?;

    let stdin = tokio_io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut stdout = tokio_io::stdout();

    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("invalid JSON-RPC request: {}", e);
                continue;
            }
        };

        // Notifications (no id) — no response required
        if req.id.is_none() {
            tracing::debug!("notification: {}", req.method);
            continue;
        }
        let id = req.id.clone().unwrap();

        let resp = match req.method.as_str() {
            "initialize" => JsonRpcResponse::success(id, handlers::handle_initialize(&req.params)),
            "tools/list" => JsonRpcResponse::success(id, handlers::handle_tools_list()),
            "tools/call" => match handlers::handle_tools_call(&exec, &req.params) {
                Ok(result) => JsonRpcResponse::success(id, result),
                Err(e) => JsonRpcResponse::error(id, -32603, e),
            },
            other => JsonRpcResponse::error(id, -32601, format!("method not found: {}", other)),
        };

        let json = serde_json::to_string(&resp)?;
        stdout.write_all(json.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    Ok(())
    // Executor drops here, closing the SQLite connection cleanly.
}
