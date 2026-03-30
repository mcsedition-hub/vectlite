use std::process::ExitCode;

use vectlite::{Database, Metadata, MetadataFilter, MetadataValue, SearchOptions};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    let Some(command) = args.get(1).map(String::as_str) else {
        print_usage();
        return Ok(());
    };

    match command {
        "init" => {
            if args.len() != 4 {
                return Err("usage: vectlite init <path> <dimension>".to_owned());
            }

            let path = &args[2];
            let dimension = args[3]
                .parse::<usize>()
                .map_err(|_| "dimension must be a positive integer".to_owned())?;

            Database::create(path, dimension).map_err(|err| err.to_string())?;
            println!("initialized {path} with dimension {dimension}");
            Ok(())
        }
        "insert" => {
            if args.len() < 5 || args.len() > 6 {
                return Err("usage: vectlite insert <path> <id> <vector> [metadata]".to_owned());
            }

            let path = &args[2];
            let id = &args[3];
            let vector = parse_vector(&args[4])?;
            let metadata = parse_metadata(args.get(5).map(String::as_str))?;

            let mut database =
                Database::open_or_create(path, vector.len()).map_err(|err| err.to_string())?;
            database
                .upsert(id, vector, metadata)
                .map_err(|err| err.to_string())?;
            println!("upserted {id}");
            Ok(())
        }
        "get" => {
            if args.len() != 4 {
                return Err("usage: vectlite get <path> <id>".to_owned());
            }

            let database = Database::open(&args[2]).map_err(|err| err.to_string())?;
            if let Some(record) = database.get(&args[3]) {
                println!("id={}", record.id);
                println!("vector={}", format_vector(&record.vector));
                println!("metadata={}", format_metadata(&record.metadata));
            } else {
                println!("not found");
            }
            Ok(())
        }
        "delete" => {
            if args.len() != 4 {
                return Err("usage: vectlite delete <path> <id>".to_owned());
            }

            let mut database = Database::open(&args[2]).map_err(|err| err.to_string())?;
            let deleted = database.delete(&args[3]).map_err(|err| err.to_string())?;
            println!("{}", if deleted { "deleted" } else { "not found" });
            Ok(())
        }
        "search" => {
            if args.len() < 5 || args.len() > 7 {
                return Err("usage: vectlite search <path> <vector> <k> [filters]".to_owned());
            }

            let database = Database::open(&args[2]).map_err(|err| err.to_string())?;
            let vector = parse_vector(&args[3])?;
            let top_k = args[4]
                .parse::<usize>()
                .map_err(|_| "k must be a positive integer".to_owned())?;
            let filter = parse_filters(args.get(5).map(String::as_str))?;

            let results = database
                .search(&vector, SearchOptions { top_k, filter })
                .map_err(|err| err.to_string())?;

            if results.is_empty() {
                println!("no matches");
                return Ok(());
            }

            for result in results {
                println!(
                    "{} score={:.4} metadata={}",
                    result.id,
                    result.score,
                    format_metadata(&result.metadata)
                );
            }
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => Err(format!("unknown command: {command}")),
    }
}

fn parse_vector(input: &str) -> Result<Vec<f32>, String> {
    if input.trim().is_empty() {
        return Err("vector cannot be empty".to_owned());
    }

    input
        .split(',')
        .map(|segment| {
            segment
                .trim()
                .parse::<f32>()
                .map_err(|_| format!("invalid float in vector: {segment}"))
        })
        .collect()
}

fn parse_metadata(input: Option<&str>) -> Result<Metadata, String> {
    let mut metadata = Metadata::new();
    let Some(input) = input else {
        return Ok(metadata);
    };

    if input.trim().is_empty() {
        return Ok(metadata);
    }

    for pair in input.split(',') {
        let Some((key, raw_value)) = pair.split_once('=') else {
            return Err(format!("invalid metadata pair: {pair}"));
        };
        metadata.insert(
            key.trim().to_owned(),
            infer_metadata_value(raw_value.trim()),
        );
    }

    Ok(metadata)
}

fn parse_filters(input: Option<&str>) -> Result<Option<MetadataFilter>, String> {
    let Some(input) = input else {
        return Ok(None);
    };
    if input.trim().is_empty() {
        return Ok(None);
    }

    let mut filters = Vec::new();
    for item in input.split(',') {
        if let Some((key, needle)) = item.split_once('~') {
            filters.push(MetadataFilter::contains(key.trim(), needle.trim()));
            continue;
        }

        if let Some((key, value)) = item.split_once('>') {
            let value = value
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid numeric filter: {item}"))?;
            filters.push(MetadataFilter::gt(key.trim(), value));
            continue;
        }

        if let Some((key, value)) = item.split_once('<') {
            let value = value
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid numeric filter: {item}"))?;
            filters.push(MetadataFilter::lt(key.trim(), value));
            continue;
        }

        if let Some((key, value)) = item.split_once('=') {
            filters.push(MetadataFilter::eq(
                key.trim(),
                infer_metadata_value(value.trim()),
            ));
            continue;
        }

        filters.push(MetadataFilter::exists(item.trim()));
    }

    Ok(Some(if filters.len() == 1 {
        filters.remove(0)
    } else {
        MetadataFilter::and(filters)
    }))
}

fn infer_metadata_value(raw: &str) -> MetadataValue {
    if raw.eq_ignore_ascii_case("true") {
        return MetadataValue::Boolean(true);
    }

    if raw.eq_ignore_ascii_case("false") {
        return MetadataValue::Boolean(false);
    }

    if let Ok(value) = raw.parse::<i64>() {
        return MetadataValue::Integer(value);
    }

    if let Ok(value) = raw.parse::<f64>() {
        return MetadataValue::Float(value);
    }

    MetadataValue::String(raw.to_owned())
}

fn format_vector(vector: &[f32]) -> String {
    vector
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_metadata(metadata: &Metadata) -> String {
    if metadata.is_empty() {
        return "{}".to_owned();
    }

    let body = metadata
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{body}}}")
}

fn print_usage() {
    println!("vectlite <command>");
    println!("  init <path> <dimension>");
    println!("  insert <path> <id> <1.0,2.0,3.0> [key=value,key2=42]");
    println!("  get <path> <id>");
    println!("  delete <path> <id>");
    println!("  search <path> <1.0,2.0,3.0> <k> [source=blog,title~auth,score>5]");
}
