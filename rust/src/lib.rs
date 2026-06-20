//! In-process SBOL/GenBank parsing for synbiotorch, bound from the sbol-rs
//! Rust crates via PyO3.
//!
//! Each entry point parses an input into an SBOL 3 document and flattens it to
//! plain owned records, returned to Python as a JSON string (a list of record
//! objects). The Python side (`synbiotorch.sources.sbol`) maps those into
//! `Design` instances, so no SBOL object or document lifetime ever crosses the
//! FFI boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;

use sbol::prelude::*;
use sbol::{
    Cut, Document, EntireSequence, Iri, RdfFormat, Range, Resource, SbolObject, Term, UpgradeOptions,
};

const SBOL3: &str = "http://sbols.org/v3#";

#[derive(Serialize)]
struct ExtensionDto {
    predicate: String,
    value: String,
}

#[derive(Serialize)]
struct LocationDto {
    start: Option<i64>,
    end: Option<i64>,
    orientation: Option<String>,
}

#[derive(Serialize)]
struct FeatureDto {
    iri: String,
    kind: Option<String>,
    instance_of: Option<String>,
    roles: Vec<String>,
    locations: Vec<LocationDto>,
}

#[derive(Serialize)]
struct SequenceDto {
    elements: String,
    encoding: Option<String>,
}

#[derive(Serialize)]
struct RecordDto {
    iri: String,
    record_class: String,
    display_id: Option<String>,
    name: Option<String>,
    roles: Vec<String>,
    types: Vec<String>,
    sequence: Option<SequenceDto>,
    features: Vec<FeatureDto>,
    /// Non-SBOL annotation triples on the object's identity, by which a numeric
    /// supervised label can be read from a file source.
    extensions: Vec<ExtensionDto>,
}

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn term_str(term: &Term) -> String {
    match term {
        Term::Literal(literal) => literal.value().to_owned(),
        Term::Resource(resource) => resource_str(resource),
        _ => String::new(),
    }
}

fn iris_to_strings(iris: &[Iri]) -> Vec<String> {
    iris.iter().map(|iri| iri.as_str().to_owned()).collect()
}

/// Render a subject resource as a string. IRI resources render bare (the
/// identifiers SBOL top-levels carry); blank nodes render as ``_:id``.
fn resource_str(resource: &Resource) -> String {
    resource.to_string()
}

fn parse_format(fmt: &str) -> PyResult<RdfFormat> {
    RdfFormat::from_extension(fmt)
        .ok_or_else(|| PyValueError::new_err(format!("unknown RDF format: {fmt}")))
}

fn location_dto(doc: &Document, location: &Resource) -> Option<LocationDto> {
    match doc.resolve(location)? {
        SbolObject::Range(Range { start, end, location, .. }) => Some(LocationDto {
            start: *start,
            end: *end,
            orientation: location.orientation.as_ref().map(|o| o.as_str().to_owned()),
        }),
        SbolObject::Cut(Cut { at, location, .. }) => Some(LocationDto {
            start: *at,
            end: *at,
            orientation: location.orientation.as_ref().map(|o| o.as_str().to_owned()),
        }),
        SbolObject::EntireSequence(EntireSequence { location, .. }) => Some(LocationDto {
            start: None,
            end: None,
            orientation: location.orientation.as_ref().map(|o| o.as_str().to_owned()),
        }),
        _ => None,
    }
}

fn feature_dto(doc: &Document, feature: &Resource) -> Option<FeatureDto> {
    let object = doc.resolve(feature)?;
    let (kind, instance_of, roles, locations): (&str, Option<String>, Vec<String>, &[Resource]) =
        match object {
            SbolObject::SubComponent(sc) => (
                "SubComponent",
                sc.instance_of.as_ref().map(resource_str),
                iris_to_strings(&sc.feature.roles),
                &sc.locations,
            ),
            SbolObject::SequenceFeature(sf) => (
                "SequenceFeature",
                None,
                iris_to_strings(&sf.feature.roles),
                &sf.locations,
            ),
            SbolObject::LocalSubComponent(lc) => (
                "LocalSubComponent",
                None,
                iris_to_strings(&lc.feature.roles),
                &lc.locations,
            ),
            _ => return None,
        };
    Some(FeatureDto {
        iri: object_identity(object)?,
        kind: Some(kind.to_owned()),
        instance_of,
        roles,
        locations: locations.iter().filter_map(|loc| location_dto(doc, loc)).collect(),
    })
}

fn object_identity(object: &SbolObject) -> Option<String> {
    match object {
        SbolObject::SubComponent(o) => Some(resource_str(&o.identity)),
        SbolObject::SequenceFeature(o) => Some(resource_str(&o.identity)),
        SbolObject::LocalSubComponent(o) => Some(resource_str(&o.identity)),
        _ => None,
    }
}

fn document_to_records(doc: &Document) -> Vec<RecordDto> {
    let mut records: Vec<RecordDto> = Vec::new();
    let mut referenced: Vec<String> = Vec::new();

    for component in doc.components() {
        let sequence = component.sequences.iter().find_map(|seq_ref| {
            referenced.push(resource_str(seq_ref));
            match doc.resolve(seq_ref) {
                Some(SbolObject::Sequence(seq)) => seq.elements.as_ref().map(|elements| SequenceDto {
                    elements: elements.clone(),
                    encoding: seq.encoding.as_ref().map(|e| e.as_str().to_owned()),
                }),
                _ => None,
            }
        });
        let features = component
            .features
            .iter()
            .filter_map(|feat| feature_dto(doc, feat))
            .collect();
        records.push(RecordDto {
            iri: resource_str(&component.identity),
            record_class: format!("{SBOL3}Component"),
            display_id: component.display_id().map(str::to_owned),
            name: component.name().map(str::to_owned),
            roles: iris_to_strings(&component.roles),
            types: iris_to_strings(&component.types),
            sequence,
            features,
            extensions: component
                .extensions()
                .iter()
                .map(|e| ExtensionDto {
                    predicate: e.predicate.as_str().to_owned(),
                    value: term_str(&e.object),
                })
                .collect(),
        });
    }

    // Sequences not owned by any component become sequence-only records, so a
    // bare SBOL Sequence document (or SBOL2 ComponentDefinition sequence) still
    // yields one record each.
    for seq in doc.sequences() {
        let iri = resource_str(&seq.identity);
        if referenced.iter().any(|r| r == &iri) {
            continue;
        }
        let Some(elements) = seq.elements.as_ref() else {
            continue;
        };
        records.push(RecordDto {
            iri,
            record_class: format!("{SBOL3}Sequence"),
            display_id: seq.display_id().map(str::to_owned),
            name: seq.name().map(str::to_owned),
            roles: Vec::new(),
            types: Vec::new(),
            sequence: Some(SequenceDto {
                elements: elements.clone(),
                encoding: seq.encoding.as_ref().map(|e| e.as_str().to_owned()),
            }),
            features: Vec::new(),
            extensions: seq
                .extensions()
                .iter()
                .map(|e| ExtensionDto {
                    predicate: e.predicate.as_str().to_owned(),
                    value: term_str(&e.object),
                })
                .collect(),
        });
    }

    records
}

fn records_json(doc: &Document) -> PyResult<String> {
    serde_json::to_string(&document_to_records(doc)).map_err(py_err)
}

/// Import GenBank text, rooting top-levels under `namespace`, and return the
/// resulting records as a JSON string.
#[pyfunction]
fn import_genbank(namespace: &str, text: &str) -> PyResult<String> {
    let importer = sbol_genbank::GenbankImporter::new(namespace).map_err(py_err)?;
    let (doc, _report) = importer.read_str(text).map_err(py_err)?;
    records_json(&doc)
}

/// Read an SBOL 3 RDF document (`fmt` is an extension: ttl/rdf/jsonld/nt).
#[pyfunction]
fn read_sbol3(text: &str, fmt: &str) -> PyResult<String> {
    let doc = Document::read(text, parse_format(fmt)?).map_err(py_err)?;
    records_json(&doc)
}

/// Upgrade an SBOL 2 RDF document to SBOL 3 and return its records.
#[pyfunction]
#[pyo3(signature = (text, fmt, namespace=None))]
fn upgrade_sbol2(text: &str, fmt: &str, namespace: Option<&str>) -> PyResult<String> {
    let mut options = UpgradeOptions::default();
    if let Some(ns) = namespace {
        options.default_namespace = Some(Iri::new(ns.to_owned()).map_err(py_err)?);
    }
    let (doc, _report) =
        Document::upgrade_from_sbol2_with(text, parse_format(fmt)?, options).map_err(py_err)?;
    records_json(&doc)
}

#[pymodule]
fn _sbol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(import_genbank, m)?)?;
    m.add_function(wrap_pyfunction!(read_sbol3, m)?)?;
    m.add_function(wrap_pyfunction!(upgrade_sbol2, m)?)?;
    Ok(())
}
