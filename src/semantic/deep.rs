use std::collections::HashMap;

use crate::search;

pub fn prf_terms(query: &str, seeds: &[search::ChunkRecord], max_terms: usize) -> Vec<String> {
    let max_terms = max_terms.max(1);
    let mut freq: HashMap<String, usize> = HashMap::new();

    for seed in seeds {
        if let Some(sym) = &seed.symbol {
            for t in split_identifiers(sym) {
                *freq.entry(t).or_insert(0) += 3;
            }
        }
        for t in extract_identifiers(&seed.content) {
            *freq.entry(t).or_insert(0) += 1;
        }
    }

    // Lightly bias towards query terms.
    for t in extract_identifiers(query) {
        *freq.entry(t).or_insert(0) += 2;
    }

    let mut items: Vec<(String, usize)> =
        freq.into_iter().filter(|(t, _)| is_good_term(t)).collect();

    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items.truncate(max_terms);
    items.into_iter().map(|(t, _)| t).collect()
}

pub fn fts_or_query(terms: &[String]) -> String {
    let mut parts = Vec::new();
    for t in terms {
        let escaped = t.replace('"', "\"\"");
        parts.push(format!("\"{escaped}\""));
    }
    parts.join(" OR ")
}

fn extract_identifiers(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut started = false;

    for ch in text.chars() {
        if !started {
            if ch.is_ascii_alphabetic() || ch == '_' {
                cur.clear();
                cur.push(ch);
                started = true;
            }
            continue;
        }

        if ch.is_ascii_alphanumeric() || ch == '_' {
            cur.push(ch);
            continue;
        }

        if cur.len() >= 3 {
            out.push(cur.clone());
        }
        started = false;
    }

    if started && cur.len() >= 3 {
        out.push(cur);
    }

    out
}

fn split_identifiers(sym: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in sym.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if part.len() >= 3 {
            out.push(part.to_string());
        }
    }
    out
}

fn is_good_term(term: &str) -> bool {
    let lower = term.to_ascii_lowercase();
    if lower.len() < 3 {
        return false;
    }
    if lower.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    !STOPWORDS.contains(&lower.as_str())
}

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "this", "that", "from", "into", "impl", "pub", "fn", "let",
    "const", "static", "struct", "enum", "trait", "class", "def", "return", "import", "package",
    "func", "type", "var", "if", "else", "match", "async", "await", "use", "mod", "crate", "self",
    "super", "true", "false", "none", "nil", "null", "new",
];
