use anyhow::{Context as _, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language as TsLanguage, Node, Parser, Query, QueryCursor};

use crate::config::IndexConfig;
use crate::index::scan::Language;

pub struct TreesitterChunker {
    rust_lang: TsLanguage,
    rust_query: Query,

    js_lang: TsLanguage,
    js_query: Query,

    ts_lang: TsLanguage,
    ts_query: Query,

    tsx_lang: TsLanguage,
    tsx_query: Query,

    py_lang: TsLanguage,
    py_query: Query,

    go_lang: TsLanguage,
    go_query: Query,
}

impl TreesitterChunker {
    pub fn new() -> Result<Self> {
        let rust_lang: TsLanguage = tree_sitter_rust::LANGUAGE.into();
        let js_lang: TsLanguage = tree_sitter_javascript::LANGUAGE.into();
        let ts_lang: TsLanguage = tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into();
        let tsx_lang: TsLanguage = tree_sitter_typescript::LANGUAGE_TSX.into();
        let py_lang: TsLanguage = tree_sitter_python::LANGUAGE.into();
        let go_lang: TsLanguage = tree_sitter_go::LANGUAGE.into();

        Ok(Self {
            rust_query: Query::new(&rust_lang, RUST_QUERY).context("compile Rust query")?,
            rust_lang,

            js_query: Query::new(&js_lang, JS_QUERY).context("compile JS query")?,
            js_lang,

            ts_query: Query::new(&ts_lang, TS_QUERY).context("compile TS query")?,
            ts_lang,

            tsx_query: Query::new(&tsx_lang, TS_QUERY).context("compile TSX query")?,
            tsx_lang,

            py_query: Query::new(&py_lang, PY_QUERY).context("compile Python query")?,
            py_lang,

            go_query: Query::new(&go_lang, GO_QUERY).context("compile Go query")?,
            go_lang,
        })
    }

    pub fn chunk(
        &mut self,
        repo_path: &str,
        language: Language,
        text: &str,
        cfg: &IndexConfig,
    ) -> Result<Vec<super::Chunk>> {
        let (ts_lang, query, func_ancestors) = match language {
            Language::Rust => (&self.rust_lang, &self.rust_query, &[][..]),
            Language::Js | Language::Jsx => (
                &self.js_lang,
                &self.js_query,
                &[
                    "function_declaration",
                    "function_expression",
                    "arrow_function",
                    "generator_function",
                    "generator_function_declaration",
                    "method_definition",
                ][..],
            ),
            Language::Ts | Language::Tsx => (
                if language == Language::Ts {
                    &self.ts_lang
                } else {
                    &self.tsx_lang
                },
                if language == Language::Ts {
                    &self.ts_query
                } else {
                    &self.tsx_query
                },
                &[
                    "function_declaration",
                    "function_expression",
                    "arrow_function",
                    "generator_function",
                    "generator_function_declaration",
                    "method_definition",
                ][..],
            ),
            Language::Python => (&self.py_lang, &self.py_query, &["function_definition"][..]),
            Language::Go => (&self.go_lang, &self.go_query, &[][..]),
            Language::Markdown | Language::Unknown => return Ok(Vec::new()),
        };

        let mut parser = Parser::new();
        parser
            .set_language(ts_lang)
            .context("set tree-sitter language")?;
        let tree = match parser.parse(text, None) {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let capture_names = query.capture_names();
        let mut cursor = QueryCursor::new();
        let mut out = Vec::new();

        let mut matches = cursor.matches(query, tree.root_node(), text.as_bytes());
        while let Some(m) = matches.next() {
            let mut def: Option<(Node<'_>, &'static str)> = None;
            let mut sym: Option<Node<'_>> = None;

            for cap in m.captures {
                let cap_name = capture_names.get(cap.index as usize).copied().unwrap_or("");
                match cap_name {
                    "sym" => sym = Some(cap.node),
                    "fn" => def = Some((cap.node, "function")),
                    "method" => def = Some((cap.node, "method")),
                    "type" => def = Some((cap.node, "type")),
                    "module" => def = Some((cap.node, "module")),
                    "const" => def = Some((cap.node, "const")),
                    _ => {}
                }
            }

            let Some((def_node, base_kind)) = def else {
                continue;
            };

            if has_ancestor_kind(def_node, func_ancestors) {
                continue;
            }

            let mut kind = base_kind.to_string();
            if base_kind == "function"
                && language == Language::Python
                && has_ancestor_kind(def_node, &["class_definition"])
            {
                kind = "method".to_string();
            }

            let symbol = sym
                .and_then(|n| n.utf8_text(text.as_bytes()).ok())
                .map(|s| s.to_string());

            let start_line = node_start_line(def_node);
            let end_line = node_end_line(def_node, start_line);

            let start_byte = def_node.start_byte();
            let end_byte = def_node.end_byte();
            if start_byte >= end_byte || end_byte > text.as_bytes().len() {
                continue;
            }

            let raw = &text.as_bytes()[start_byte..end_byte];
            if raw.len() > cfg.max_chunk_bytes {
                let node_text = String::from_utf8_lossy(raw).to_string();
                out.extend(super::fallback_chunks(
                    repo_path,
                    &node_text,
                    start_line,
                    cfg,
                    "block",
                    symbol.clone(),
                ));
                continue;
            }

            let node_text = String::from_utf8_lossy(raw).to_string();
            out.push(super::make_chunk(
                repo_path, start_line, end_line, &kind, symbol, node_text,
            ));
        }

        Ok(out)
    }
}

fn has_ancestor_kind(node: Node<'_>, kinds: &[&str]) -> bool {
    let mut cur = node.parent();
    while let Some(p) = cur {
        let k = p.kind();
        if kinds.iter().any(|want| *want == k) {
            return true;
        }
        cur = p.parent();
    }
    false
}

fn node_start_line(node: Node<'_>) -> i64 {
    (node.start_position().row as i64) + 1
}

fn node_end_line(node: Node<'_>, start_line: i64) -> i64 {
    let end = node.end_position();
    let end_line = if end.column == 0 && (end.row as i64 + 1) > start_line {
        end.row as i64
    } else {
        (end.row as i64) + 1
    };
    end_line.max(start_line)
}

const RUST_QUERY: &str = r#"
(
  (function_item name: (identifier) @sym) @fn
)
(
  (struct_item name: (type_identifier) @sym) @type
)
(
  (enum_item name: (type_identifier) @sym) @type
)
(
  (trait_item name: (type_identifier) @sym) @type
)
(
  (impl_item) @type
)
(
  (type_item name: (type_identifier) @sym) @type
)
(
  (const_item name: (identifier) @sym) @const
)
(
  (static_item name: (identifier) @sym) @const
)
(
  (mod_item name: (identifier) @sym) @module
)
"#;

const JS_QUERY: &str = r#"
(
  (function_declaration name: (identifier) @sym) @fn
)
(
  (class_declaration name: (identifier) @sym) @type
)
(
  (method_definition name: [(property_identifier) (identifier)] @sym) @method
)
(
  (variable_declarator
    name: (identifier) @sym
    value: [(arrow_function) (function_expression)]
  ) @fn
)
"#;

const TS_QUERY: &str = r#"
(
  (function_declaration name: (identifier) @sym) @fn
)
(
  (class_declaration name: (type_identifier) @sym) @type
)
(
  (interface_declaration name: (type_identifier) @sym) @type
)
(
  (type_alias_declaration name: (type_identifier) @sym) @type
)
(
  (enum_declaration name: (identifier) @sym) @type
)
(
  (method_definition name: [(property_identifier) (identifier)] @sym) @method
)
(
  (variable_declarator
    name: (identifier) @sym
    value: [(arrow_function) (function_expression)]
  ) @fn
)
"#;

const PY_QUERY: &str = r#"
(
  (function_definition name: (identifier) @sym) @fn
)
(
  (class_definition name: (identifier) @sym) @type
)
"#;

const GO_QUERY: &str = r#"
(
  (function_declaration name: (identifier) @sym) @fn
)
(
  (method_declaration name: (field_identifier) @sym) @method
)
(
  (type_spec name: (type_identifier) @sym) @type
)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> IndexConfig {
        IndexConfig {
            max_file_bytes: 1_000_000,
            max_chunk_bytes: 64 * 1024,
            fallback_chunk_lines: 200,
            fallback_overlap_lines: 20,
        }
    }

    #[test]
    fn rust_extracts_function_and_type() {
        let mut chunker = TreesitterChunker::new().expect("chunker");
        let text = r#"
pub struct Foo;

pub fn baz() -> i32 {
  42
}
"#;
        let chunks = chunker
            .chunk("src/lib.rs", Language::Rust, text, &cfg())
            .expect("chunk");

        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "type" && c.symbol.as_deref() == Some("Foo"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "function" && c.symbol.as_deref() == Some("baz"))
        );
    }

    #[test]
    fn javascript_extracts_class_function_and_method() {
        let mut chunker = TreesitterChunker::new().expect("chunker");
        let text = r#"
class Foo {
  bar() {}
}

function baz() {}
const qux = () => {};
"#;
        let chunks = chunker
            .chunk("a.js", Language::Js, text, &cfg())
            .expect("chunk");

        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "type" && c.symbol.as_deref() == Some("Foo"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "method" && c.symbol.as_deref() == Some("bar"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "function" && c.symbol.as_deref() == Some("baz"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "function" && c.symbol.as_deref() == Some("qux"))
        );
    }

    #[test]
    fn typescript_extracts_interface_type_alias_and_class() {
        let mut chunker = TreesitterChunker::new().expect("chunker");
        let text = r#"
export interface Foo { x: number }
export type Bar = { y: string }
export class Baz { qux(): void {} }
"#;
        let chunks = chunker
            .chunk("a.ts", Language::Ts, text, &cfg())
            .expect("chunk");

        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "type" && c.symbol.as_deref() == Some("Foo"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "type" && c.symbol.as_deref() == Some("Bar"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "type" && c.symbol.as_deref() == Some("Baz"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "method" && c.symbol.as_deref() == Some("qux"))
        );
    }

    #[test]
    fn python_extracts_class_method_and_function() {
        let mut chunker = TreesitterChunker::new().expect("chunker");
        let text = r#"
class Foo:
  def bar(self):
    return 1

def baz():
  return 2
"#;
        let chunks = chunker
            .chunk("a.py", Language::Python, text, &cfg())
            .expect("chunk");

        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "type" && c.symbol.as_deref() == Some("Foo"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "method" && c.symbol.as_deref() == Some("bar"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "function" && c.symbol.as_deref() == Some("baz"))
        );
    }

    #[test]
    fn go_extracts_type_method_and_function() {
        let mut chunker = TreesitterChunker::new().expect("chunker");
        let text = r#"
package main

type Foo struct {}

func (f Foo) Bar() {}
func Baz() {}
"#;
        let chunks = chunker
            .chunk("a.go", Language::Go, text, &cfg())
            .expect("chunk");

        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "type" && c.symbol.as_deref() == Some("Foo"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "method" && c.symbol.as_deref() == Some("Bar"))
        );
        assert!(
            chunks
                .iter()
                .any(|c| c.kind == "function" && c.symbol.as_deref() == Some("Baz"))
        );
    }
}
