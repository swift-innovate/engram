//! MCP method handlers.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::executor::Executor;

pub fn handle_initialize(_params: &Value) -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "engram-aql",
            "version": env!("CARGO_PKG_VERSION")
        },
        "capabilities": {
            "tools": {}
        }
    })
}

pub fn handle_tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "engram_aql",
                "description": "Execute an AQL (Agent Query Language) read query against this agent's memory. \
                    Supports: RECALL, SCAN, LOOKUP, LOAD, WITH LINKS, FOLLOW LINKS, AGGREGATE, ORDER BY, PIPELINE, \
                    and semantic search via RECALL ... LIKE $var / PATTERN $var THRESHOLD t (bind $var in `variables`). \
                    Writes (STORE/UPDATE/FORGET/LINK/REFLECT) are not yet supported — use engram_retain via the TypeScript MCP server instead. \
                    Example: RECALL FROM EPISODIC ORDER BY created_at DESC LIMIT 5",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "AQL query string. Example: RECALL FROM EPISODIC ORDER BY created_at DESC LIMIT 5"
                        },
                        "variables": {
                            "type": "object",
                            "additionalProperties": true,
                            "description": "Bound variables for LIKE $name / PATTERN $name. A value may be query text (string, embedded server-side via engram-mcp) or a precomputed embedding (array of numbers used directly). Example: {\"q\": \"deployment rollback\"} or {\"q\": [0.1, 0.2, ...]}"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    })
}

pub fn handle_tools_call(exec: &Executor, params: &Value) -> Result<Value, String> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing tool name".to_string())?;

    if name != "engram_aql" {
        return Err(format!("unknown tool: {}", name));
    }

    let arguments = params.get("arguments");

    let query = arguments
        .and_then(|a| a.get("query"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing query argument".to_string())?;

    // Optional `variables` object → bound query variables for LIKE/PATTERN.
    // Each key maps to a serde_json value (string text or numeric array probe).
    let vars: BTreeMap<String, Value> = arguments
        .and_then(|a| a.get("variables"))
        .and_then(|v| v.as_object())
        .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();

    let result = exec
        .query_with_vars(query, vars)
        .map_err(|e| e.to_string())?;
    let result_json = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;

    Ok(json!({
        "content": [
            {
                "type": "text",
                "text": result_json
            }
        ],
        "isError": !result.success
    }))
}
