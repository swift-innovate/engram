//! `engram-aql repl` — interactive REPL with rustyline and pretty tables.
//!
//! Launches an interactive prompt where users can enter AQL queries and
//! see formatted table output. REPL commands start with `\` to distinguish
//! them from AQL syntax (e.g., `\help`, `\quit`).

use std::path::Path;

use anyhow::Result;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, ContentArrangement, Table};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde_json::Value as JsonValue;

use crate::executor::Executor;
use crate::result::QueryResult;

pub fn run(db_path: &Path) -> Result<()> {
    println!("engram-aql {} — read-only mode", env!("CARGO_PKG_VERSION"));
    println!("Connected to: {}", db_path.display());
    println!("Type \\help for commands. \\quit to exit.");
    println!();

    let exec = Executor::open(db_path)?;
    let mut rl = DefaultEditor::new()?;

    loop {
        match rl.readline("aql> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                rl.add_history_entry(trimmed).ok();

                // REPL meta-commands (backslash-prefixed)
                if let Some(cmd) = trimmed.strip_prefix('\\') {
                    match cmd {
                        "quit" | "q" | "exit" => break,
                        "help" | "h" => {
                            print_help();
                            continue;
                        }
                        other => {
                            println!("unknown command: \\{}", other);
                            continue;
                        }
                    }
                }

                // AQL query — execute and print result
                match exec.query(trimmed) {
                    Ok(result) => print_result(&result),
                    Err(e) => println!("error: {}", e),
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => {
                println!("readline error: {}", e);
                break;
            }
        }
    }

    // Executor drops here, closing the SQLite connection cleanly.
    Ok(())
}

fn print_help() {
    println!("REPL commands:");
    println!("  \\help, \\h    show this help");
    println!("  \\quit, \\q    exit the REPL");
    println!();
    println!("Otherwise, type an AQL query and press Enter.");
    println!("Example: RECALL FROM EPISODIC ALL LIMIT 5");
}

fn print_result(result: &QueryResult) {
    if !result.success {
        println!(
            "error: {}",
            result.error.as_deref().unwrap_or("unknown error")
        );
        return;
    }

    if result.data.is_empty() {
        println!("(no rows · {} ms)", result.timing_ms);
        return;
    }

    // Collect column names from the first row (preserving insertion order)
    let JsonValue::Object(first) = &result.data[0] else {
        // Non-object data — fall back to raw JSON pretty-print
        println!(
            "{}",
            serde_json::to_string_pretty(&result.data).unwrap_or_default()
        );
        println!("{} rows · {} ms", result.count, result.timing_ms);
        return;
    };
    let columns: Vec<String> = first.keys().cloned().collect();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(columns.iter().map(Cell::new));

    for row in &result.data {
        let JsonValue::Object(obj) = row else {
            continue;
        };
        let cells: Vec<Cell> = columns
            .iter()
            .map(|col| {
                let val = obj.get(col).cloned().unwrap_or(JsonValue::Null);
                let display = match val {
                    JsonValue::Null => "NULL".to_string(),
                    JsonValue::String(s) => {
                        if s.len() > 60 {
                            format!("{}…", &s[..57])
                        } else {
                            s
                        }
                    }
                    v => v.to_string(),
                };
                Cell::new(display)
            })
            .collect();
        table.add_row(cells);
    }

    println!("{}", table);
    println!("{} rows · {} ms", result.count, result.timing_ms);

    if !result.warnings.is_empty() {
        println!();
        for w in &result.warnings {
            println!("warning: {}", w);
        }
    }
}
