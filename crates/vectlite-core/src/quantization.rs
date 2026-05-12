//! Vector quantization module for memory-efficient similarity search.
//!
//! Supports three quantization strategies:
//! - **Scalar (int8)**: compact in-memory candidate index with minimal recall loss
//! - **Binary**: smallest in-memory candidate index, uses Hamming distance for fast filtering
//! - **Product Quantization (PQ)**: Configurable compression for very large datasets
//!
//! All strategies support a 2-stage pipeline: fast quantized search followed by
//! exact float32 rescoring of top candidates.

use std::io::{Error, ErrorKind, Read, Write};

use crate::{DistanceMetric, Result, VectLiteError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for enabling quantization on a database.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantizationConfig {
    /// Scalar quantization: maps each f32 dimension to int8 using per-dimension
    /// min/max calibration for a compact in-memory candidate index.
    Scalar(ScalarQuantizationConfig),
    /// Binary quantization: maps each f32 dimension to a single bit.
    /// Smallest in-memory candidate index. Best for high-dimensional normalized embeddings.
    Binary(BinaryQuantizationConfig),
    /// Product quantization: splits vector into sub-vectors and quantizes each
    /// to a centroid index. Highest compression for large datasets.
    Product(ProductQuantizationConfig),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarQuantizationConfig {
    /// Number of top candidates from quantized search to rescore with float32.
    /// Default: 10x top_k.
    pub rescore_multiplier: usize,
}

impl Default for ScalarQuantizationConfig {
    fn default() -> Self {
        Self {
            rescore_multiplier: 10,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryQuantizationConfig {
    /// Number of top candidates from Hamming search to rescore with float32.
    /// Default: 10x top_k.
    pub rescore_multiplier: usize,
}

impl Default for BinaryQuantizationConfig {
    fn default() -> Self {
        Self {
            rescore_multiplier: 10,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProductQuantizationConfig {
    /// Number of sub-vectors (sub-spaces). Must divide the vector dimension evenly.
    /// Typical values: 8, 16, 32, 64.
    pub num_sub_vectors: usize,
    /// Number of centroids per sub-vector (k in k-means). Must be <= 256.
    /// Default: 256 (uses u8 codes).
    pub num_centroids: usize,
    /// Number of k-means training iterations.
    pub training_iterations: usize,
    /// Number of top candidates from PQ search to rescore with float32.
    /// Default: 10x top_k.
    pub rescore_multiplier: usize,
}

impl Default for ProductQuantizationConfig {
    fn default() -> Self {
        Self {
            num_sub_vectors: 16,
            num_centroids: 256,
            training_iterations: 20,
            rescore_multiplier: 10,
        }
    }
}

/// Choose a valid default PQ sub-vector count for a database dimension.
///
/// Prefer the historical default of 16 when possible, then fall back to smaller
/// common divisors so dimensions such as 100, 146, and 200 do not require an
/// explicit `num_sub_vectors`.
pub fn default_product_num_sub_vectors(dimension: usize) -> usize {
    [16, 12, 10, 8, 6, 4, 3, 2, 1]
        .into_iter()
        .find(|candidate| dimension % candidate == 0)
        .unwrap_or(1)
}

/// List every valid PQ sub-vector count for a database dimension.
pub fn valid_product_num_sub_vectors(dimension: usize) -> Vec<usize> {
    if dimension == 0 {
        return Vec::new();
    }

    (1..=dimension)
        .filter(|candidate| dimension % candidate == 0)
        .collect()
}

/// Validate quantization settings before an index build can panic.
pub fn validate_quantization_config(config: &QuantizationConfig, dimension: usize) -> Result<()> {
    if let QuantizationConfig::Product(cfg) = config {
        if cfg.num_sub_vectors == 0 {
            return Err(VectLiteError::InvalidFormat(
                "num_sub_vectors must be greater than 0".to_owned(),
            ));
        }
        if dimension % cfg.num_sub_vectors != 0 {
            return Err(VectLiteError::InvalidFormat(format!(
                "dimension ({dimension}) must be divisible by num_sub_vectors ({})",
                cfg.num_sub_vectors
            )));
        }
        if cfg.num_centroids == 0 || cfg.num_centroids > 256 {
            return Err(VectLiteError::InvalidFormat(
                "num_centroids must be between 1 and 256".to_owned(),
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar Quantization
// ---------------------------------------------------------------------------

/// Calibration parameters for scalar quantization (per-dimension min/max).
#[derive(Clone, Debug)]
pub struct ScalarQuantizer {
    pub dimension: usize,
    /// Per-dimension minimum values used for calibration.
    pub mins: Vec<f32>,
    /// Per-dimension maximum values used for calibration.
    pub maxs: Vec<f32>,
    /// Per-dimension scale: 255.0 / (max - min). Pre-computed for fast quantization.
    scales: Vec<f32>,
    /// Quantized vectors stored as flat u8 array (n * dimension).
    pub codes: Vec<u8>,
    /// Number of quantized vectors.
    pub count: usize,
    pub config: ScalarQuantizationConfig,
}

impl ScalarQuantizer {
    /// Train a scalar quantizer by computing per-dimension min/max from training vectors.
    pub fn train(vectors: &[&[f32]], dimension: usize, config: ScalarQuantizationConfig) -> Self {
        assert!(!vectors.is_empty(), "need at least one vector to train");
        assert!(vectors[0].len() == dimension);

        let mut mins = vec![f32::INFINITY; dimension];
        let mut maxs = vec![f32::NEG_INFINITY; dimension];

        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                if val < mins[i] {
                    mins[i] = val;
                }
                if val > maxs[i] {
                    maxs[i] = val;
                }
            }
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&min, &max)| {
                let range = max - min;
                if range < 1e-10 { 0.0 } else { 255.0 / range }
            })
            .collect();

        let mut codes = Vec::with_capacity(vectors.len() * dimension);
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                codes.push(quantize_scalar(val, mins[i], scales[i]));
            }
        }

        Self {
            dimension,
            mins,
            maxs,
            scales,
            codes,
            count: vectors.len(),
            config,
        }
    }

    /// Add vectors to the quantized index (after initial training).
    pub fn add_vectors(&mut self, vectors: &[&[f32]]) {
        for vector in vectors {
            assert_eq!(vector.len(), self.dimension);
            for (i, &val) in vector.iter().enumerate() {
                self.codes
                    .push(quantize_scalar(val, self.mins[i], self.scales[i]));
            }
        }
        self.count += vectors.len();
    }

    /// Quantize a single query vector.
    pub fn quantize_query(&self, query: &[f32]) -> Vec<u8> {
        assert_eq!(query.len(), self.dimension);
        query
            .iter()
            .enumerate()
            .map(|(i, &val)| quantize_scalar(val, self.mins[i], self.scales[i]))
            .collect()
    }

    /// Compute approximate cosine similarity between the query and all stored vectors.
    /// Returns indices sorted by approximate similarity (best first).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        self.search_with_metric(query, top_k, DistanceMetric::Cosine)
    }

    /// Compute approximate metric scores between the query and all stored vectors.
    /// Returns indices sorted by approximate score (best first).
    pub fn search_with_metric(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dimension);
        let rescore_count = rescore_count(top_k, self.config.rescore_multiplier, self.count);
        let mut scores: Vec<(usize, f32)> = (0..self.count)
            .map(|idx| {
                let offset = idx * self.dimension;
                let code_slice = &self.codes[offset..offset + self.dimension];
                let sim = self.approximate_score(query, code_slice, metric);
                (idx, sim)
            })
            .collect();

        // Partial sort: get top rescore_count candidates
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scores.truncate(rescore_count);
        scores
    }

    fn approximate_score(&self, query: &[f32], code_slice: &[u8], metric: DistanceMetric) -> f32 {
        match metric {
            DistanceMetric::Cosine => {
                let mut dot = 0.0_f32;
                let mut query_norm = 0.0_f32;
                let mut vector_norm = 0.0_f32;

                for (((&query_value, &code), &min), &scale) in query
                    .iter()
                    .zip(code_slice.iter())
                    .zip(self.mins.iter())
                    .zip(self.scales.iter())
                {
                    let value = dequantize_scalar(code, min, scale);
                    dot += query_value * value;
                    query_norm += query_value * query_value;
                    vector_norm += value * value;
                }

                if query_norm == 0.0 || vector_norm == 0.0 {
                    0.0
                } else {
                    dot / (query_norm.sqrt() * vector_norm.sqrt())
                }
            }
            DistanceMetric::Euclidean => {
                let mut sum = 0.0_f32;
                for (((&query_value, &code), &min), &scale) in query
                    .iter()
                    .zip(code_slice.iter())
                    .zip(self.mins.iter())
                    .zip(self.scales.iter())
                {
                    let delta = query_value - dequantize_scalar(code, min, scale);
                    sum += delta * delta;
                }
                -sum.sqrt()
            }
            DistanceMetric::DotProduct => {
                let mut dot = 0.0_f32;
                for (((&query_value, &code), &min), &scale) in query
                    .iter()
                    .zip(code_slice.iter())
                    .zip(self.mins.iter())
                    .zip(self.scales.iter())
                {
                    dot += query_value * dequantize_scalar(code, min, scale);
                }
                dot
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0_f32;
                for (((&query_value, &code), &min), &scale) in query
                    .iter()
                    .zip(code_slice.iter())
                    .zip(self.mins.iter())
                    .zip(self.scales.iter())
                {
                    sum += (query_value - dequantize_scalar(code, min, scale)).abs();
                }
                -sum
            }
        }
    }

    /// Rebuild codes from training vectors (used after deserialization with new vectors).
    pub fn rebuild_codes(&mut self, vectors: &[&[f32]]) {
        self.codes.clear();
        self.codes.reserve(vectors.len() * self.dimension);
        for vector in vectors {
            for (i, &val) in vector.iter().enumerate() {
                self.codes
                    .push(quantize_scalar(val, self.mins[i], self.scales[i]));
            }
        }
        self.count = vectors.len();
    }

    /// Serialize the quantizer parameters (not the codes, which are rebuilt on load).
    pub fn write_params(&self, writer: &mut impl Write) -> std::io::Result<()> {
        // Tag byte: 1 = scalar
        writer.write_all(&[1u8])?;
        write_usize(writer, self.dimension)?;
        write_usize(writer, self.config.rescore_multiplier)?;
        for &v in &self.mins {
            writer.write_all(&v.to_le_bytes())?;
        }
        for &v in &self.maxs {
            writer.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    /// Deserialize quantizer parameters.
    pub fn read_params(reader: &mut impl Read) -> std::io::Result<Self> {
        let dimension = read_usize(reader)?;
        let rescore_multiplier = read_usize(reader)?;
        let mut mins = vec![0.0_f32; dimension];
        let mut maxs = vec![0.0_f32; dimension];
        for v in &mut mins {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *v = f32::from_le_bytes(buf);
        }
        for v in &mut maxs {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *v = f32::from_le_bytes(buf);
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&min, &max)| {
                let range = max - min;
                if range < 1e-10 { 0.0 } else { 255.0 / range }
            })
            .collect();

        Ok(Self {
            dimension,
            mins,
            maxs,
            scales,
            codes: Vec::new(),
            count: 0,
            config: ScalarQuantizationConfig { rescore_multiplier },
        })
    }
}

// ---------------------------------------------------------------------------
// Binary Quantization
// ---------------------------------------------------------------------------

/// Binary quantizer: each dimension is mapped to a single bit (sign of the value).
/// Uses Hamming distance for fast candidate selection.
#[derive(Clone, Debug)]
pub struct BinaryQuantizer {
    pub dimension: usize,
    /// Number of bytes per vector: ceil(dimension / 8).
    pub bytes_per_vector: usize,
    /// Binary codes stored as flat byte array (n * bytes_per_vector).
    pub codes: Vec<u8>,
    /// Number of quantized vectors.
    pub count: usize,
    pub config: BinaryQuantizationConfig,
}

impl BinaryQuantizer {
    /// Create a binary quantizer for vectors of the given dimension.
    pub fn new(dimension: usize, config: BinaryQuantizationConfig) -> Self {
        let bytes_per_vector = (dimension + 7) / 8;
        Self {
            dimension,
            bytes_per_vector,
            codes: Vec::new(),
            count: 0,
            config,
        }
    }

    /// Binarize vectors and add to the index.
    pub fn add_vectors(&mut self, vectors: &[&[f32]]) {
        for vector in vectors {
            assert_eq!(vector.len(), self.dimension);
            let binary = binarize_vector(vector);
            self.codes.extend_from_slice(&binary);
        }
        self.count += vectors.len();
    }

    /// Binarize a query vector.
    pub fn binarize_query(&self, query: &[f32]) -> Vec<u8> {
        assert_eq!(query.len(), self.dimension);
        binarize_vector(query)
    }

    /// Search using Hamming distance. Returns candidate indices sorted by
    /// Hamming similarity (fewest differing bits first).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, u32)> {
        let rescore_count = rescore_count(top_k, self.config.rescore_multiplier, self.count);
        let query_binary = self.binarize_query(query);
        let mut distances: Vec<(usize, u32)> = (0..self.count)
            .map(|idx| {
                let offset = idx * self.bytes_per_vector;
                let code_slice = &self.codes[offset..offset + self.bytes_per_vector];
                let dist = hamming_distance(&query_binary, code_slice);
                (idx, dist)
            })
            .collect();

        // Sort by Hamming distance (ascending = most similar first)
        distances.sort_unstable_by_key(|&(_, d)| d);
        distances.truncate(rescore_count);
        distances
    }

    /// Rebuild codes from vectors.
    pub fn rebuild_codes(&mut self, vectors: &[&[f32]]) {
        self.codes.clear();
        self.codes.reserve(vectors.len() * self.bytes_per_vector);
        for vector in vectors {
            let binary = binarize_vector(vector);
            self.codes.extend_from_slice(&binary);
        }
        self.count = vectors.len();
    }

    /// Serialize parameters.
    pub fn write_params(&self, writer: &mut impl Write) -> std::io::Result<()> {
        // Tag byte: 2 = binary
        writer.write_all(&[2u8])?;
        write_usize(writer, self.dimension)?;
        write_usize(writer, self.config.rescore_multiplier)?;
        Ok(())
    }

    /// Deserialize parameters.
    pub fn read_params(reader: &mut impl Read) -> std::io::Result<Self> {
        let dimension = read_usize(reader)?;
        let rescore_multiplier = read_usize(reader)?;
        let bytes_per_vector = (dimension + 7) / 8;
        Ok(Self {
            dimension,
            bytes_per_vector,
            codes: Vec::new(),
            count: 0,
            config: BinaryQuantizationConfig { rescore_multiplier },
        })
    }
}

// ---------------------------------------------------------------------------
// Product Quantization
// ---------------------------------------------------------------------------

/// Product quantizer: divides vector into sub-vectors and maps each to a centroid.
#[derive(Clone, Debug)]
pub struct ProductQuantizer {
    pub dimension: usize,
    pub num_sub_vectors: usize,
    pub sub_dimension: usize,
    pub num_centroids: usize,
    /// Codebooks: shape [num_sub_vectors][num_centroids][sub_dimension].
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// PQ codes: flat array of (n * num_sub_vectors) u8 indices.
    pub codes: Vec<u8>,
    /// Number of quantized vectors.
    pub count: usize,
    pub config: ProductQuantizationConfig,
}

impl ProductQuantizer {
    /// Train a product quantizer using k-means on sub-vectors.
    pub fn train(vectors: &[&[f32]], dimension: usize, config: ProductQuantizationConfig) -> Self {
        assert!(!vectors.is_empty(), "need at least one vector to train PQ");
        assert!(
            dimension % config.num_sub_vectors == 0,
            "dimension ({dimension}) must be divisible by num_sub_vectors ({})",
            config.num_sub_vectors
        );
        assert!(config.num_centroids <= 256, "num_centroids must be <= 256");

        let sub_dimension = dimension / config.num_sub_vectors;
        let mut codebooks = Vec::with_capacity(config.num_sub_vectors);

        for sub_idx in 0..config.num_sub_vectors {
            let offset = sub_idx * sub_dimension;
            // Extract sub-vectors for this partition
            let sub_vectors: Vec<&[f32]> = vectors
                .iter()
                .map(|v| &v[offset..offset + sub_dimension])
                .collect();

            let centroids = kmeans(
                &sub_vectors,
                sub_dimension,
                config.num_centroids,
                config.training_iterations,
            );
            codebooks.push(centroids);
        }

        // Encode all training vectors
        let mut codes = Vec::with_capacity(vectors.len() * config.num_sub_vectors);
        for vector in vectors {
            for sub_idx in 0..config.num_sub_vectors {
                let offset = sub_idx * sub_dimension;
                let sub_vector = &vector[offset..offset + sub_dimension];
                let nearest = find_nearest_centroid(sub_vector, &codebooks[sub_idx]);
                codes.push(nearest as u8);
            }
        }

        Self {
            dimension,
            num_sub_vectors: config.num_sub_vectors,
            sub_dimension,
            num_centroids: config.num_centroids,
            codebooks,
            codes,
            count: vectors.len(),
            config,
        }
    }

    /// Add vectors to the PQ index.
    pub fn add_vectors(&mut self, vectors: &[&[f32]]) {
        for vector in vectors {
            assert_eq!(vector.len(), self.dimension);
            for sub_idx in 0..self.num_sub_vectors {
                let offset = sub_idx * self.sub_dimension;
                let sub_vector = &vector[offset..offset + self.sub_dimension];
                let nearest = find_nearest_centroid(sub_vector, &self.codebooks[sub_idx]);
                self.codes.push(nearest as u8);
            }
        }
        self.count += vectors.len();
    }

    /// Compute asymmetric distance table for a query. This precomputes distances
    /// from the query sub-vectors to all centroids, enabling fast approximate
    /// distance computation.
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(query.len(), self.dimension);
        let mut table = Vec::with_capacity(self.num_sub_vectors);

        for sub_idx in 0..self.num_sub_vectors {
            let offset = sub_idx * self.sub_dimension;
            let query_sub = &query[offset..offset + self.sub_dimension];
            let distances: Vec<f32> = self.codebooks[sub_idx]
                .iter()
                .map(|centroid| l2_distance_sq(query_sub, centroid))
                .collect();
            table.push(distances);
        }

        table
    }

    /// Search using asymmetric distance computation (ADC).
    /// Returns candidate indices sorted by approximate L2 distance.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let rescore_count = rescore_count(top_k, self.config.rescore_multiplier, self.count);
        let distance_table = self.compute_distance_table(query);

        let mut distances: Vec<(usize, f32)> = (0..self.count)
            .map(|idx| {
                let code_offset = idx * self.num_sub_vectors;
                let mut dist = 0.0_f32;
                for sub_idx in 0..self.num_sub_vectors {
                    let centroid_idx = self.codes[code_offset + sub_idx] as usize;
                    dist += distance_table[sub_idx][centroid_idx];
                }
                (idx, dist)
            })
            .collect();

        // Sort by distance (ascending)
        distances.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(rescore_count);
        distances
    }

    /// Rebuild codes from vectors.
    pub fn rebuild_codes(&mut self, vectors: &[&[f32]]) {
        self.codes.clear();
        self.codes.reserve(vectors.len() * self.num_sub_vectors);
        for vector in vectors {
            for sub_idx in 0..self.num_sub_vectors {
                let offset = sub_idx * self.sub_dimension;
                let sub_vector = &vector[offset..offset + self.sub_dimension];
                let nearest = find_nearest_centroid(sub_vector, &self.codebooks[sub_idx]);
                self.codes.push(nearest as u8);
            }
        }
        self.count = vectors.len();
    }

    /// Serialize the codebooks and parameters.
    pub fn write_params(&self, writer: &mut impl Write) -> std::io::Result<()> {
        // Tag byte: 3 = product
        writer.write_all(&[3u8])?;
        write_usize(writer, self.dimension)?;
        write_usize(writer, self.num_sub_vectors)?;
        write_usize(writer, self.num_centroids)?;
        write_usize(writer, self.config.training_iterations)?;
        write_usize(writer, self.config.rescore_multiplier)?;

        // Write codebooks
        for sub_codebook in &self.codebooks {
            for centroid in sub_codebook {
                for &val in centroid {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Deserialize codebooks and parameters.
    pub fn read_params(reader: &mut impl Read) -> std::io::Result<Self> {
        let dimension = read_usize(reader)?;
        let num_sub_vectors = read_usize(reader)?;
        let num_centroids = read_usize(reader)?;
        let training_iterations = read_usize(reader)?;
        let rescore_multiplier = read_usize(reader)?;
        if num_sub_vectors == 0 || dimension % num_sub_vectors != 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "dimension ({dimension}) must be divisible by num_sub_vectors ({num_sub_vectors})"
                ),
            ));
        }
        if num_centroids == 0 || num_centroids > 256 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "num_centroids must be between 1 and 256",
            ));
        }
        let sub_dimension = dimension / num_sub_vectors;

        // Read codebooks
        let mut codebooks = Vec::with_capacity(num_sub_vectors);
        for _ in 0..num_sub_vectors {
            let mut sub_codebook = Vec::with_capacity(num_centroids);
            for _ in 0..num_centroids {
                let mut centroid = vec![0.0_f32; sub_dimension];
                for v in &mut centroid {
                    let mut buf = [0u8; 4];
                    reader.read_exact(&mut buf)?;
                    *v = f32::from_le_bytes(buf);
                }
                sub_codebook.push(centroid);
            }
            codebooks.push(sub_codebook);
        }

        Ok(Self {
            dimension,
            num_sub_vectors,
            sub_dimension,
            num_centroids,
            codebooks,
            codes: Vec::new(),
            count: 0,
            config: ProductQuantizationConfig {
                num_sub_vectors,
                num_centroids,
                training_iterations,
                rescore_multiplier,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Two-Bit Quantization (ColBERTv2-style)
// ---------------------------------------------------------------------------

/// Configuration for 2-bit multi-vector quantization (ColBERTv2-style).
#[derive(Clone, Debug, PartialEq)]
pub struct TwoBitQuantizationConfig {
    /// Number of top candidate docs from quantized search to rescore with
    /// exact float32 MaxSim. Default: 4x top_k.
    pub rescore_multiplier: usize,
}

impl Default for TwoBitQuantizationConfig {
    fn default() -> Self {
        Self {
            rescore_multiplier: 4,
        }
    }
}

/// Two-bit quantizer: maps each dimension to 2 bits (4 levels) using
/// per-dimension quartile boundaries. ~16x compression vs float32.
/// Designed for ColBERT-style token-level vectors.
#[derive(Clone, Debug)]
pub struct TwoBitQuantizer {
    pub dimension: usize,
    /// Per-dimension boundary values: [q25, q50, q75] for each dimension.
    /// Shape: dimension * 3.
    pub boundaries: Vec<f32>,
    /// Quantized codes: 2 bits per dimension, packed into bytes.
    /// Each vector uses ceil(dimension / 4) bytes.
    pub codes: Vec<u8>,
    /// Number of quantized vectors.
    pub count: usize,
    /// Bytes per quantized vector.
    pub bytes_per_vector: usize,
    pub config: TwoBitQuantizationConfig,
}

impl TwoBitQuantizer {
    /// Train a 2-bit quantizer by computing per-dimension quartiles.
    pub fn train(vectors: &[&[f32]], dimension: usize, config: TwoBitQuantizationConfig) -> Self {
        assert!(!vectors.is_empty(), "need at least one vector to train");

        // Collect values per dimension and compute quartile boundaries
        let mut boundaries = Vec::with_capacity(dimension * 3);
        for d in 0..dimension {
            let mut values: Vec<f32> = vectors.iter().map(|v| v[d]).collect();
            values.sort_unstable_by(|a, b| a.total_cmp(b));
            let n = values.len();
            let q25 = values[n / 4];
            let q50 = values[n / 2];
            let q75 = values[(3 * n) / 4];
            boundaries.push(q25);
            boundaries.push(q50);
            boundaries.push(q75);
        }

        let bytes_per_vector = (dimension + 3) / 4;
        let mut codes = Vec::with_capacity(vectors.len() * bytes_per_vector);
        for vector in vectors {
            codes.extend_from_slice(&quantize_two_bit(vector, &boundaries, bytes_per_vector));
        }

        Self {
            dimension,
            boundaries,
            codes,
            count: vectors.len(),
            bytes_per_vector,
            config,
        }
    }

    /// Quantize a single vector to 2-bit codes.
    pub fn quantize_vector(&self, vector: &[f32]) -> Vec<u8> {
        quantize_two_bit(vector, &self.boundaries, self.bytes_per_vector)
    }

    /// Compute approximate dot product between a 2-bit quantized query and
    /// a stored quantized vector. Returns a score where higher = more similar.
    pub fn approx_dot(&self, query_codes: &[u8], idx: usize) -> i32 {
        let offset = idx * self.bytes_per_vector;
        let stored = &self.codes[offset..offset + self.bytes_per_vector];
        two_bit_approx_dot(query_codes, stored, self.dimension)
    }

    /// Search for top-k candidates using approximate 2-bit dot products.
    /// Returns (index, approx_score) pairs sorted best-first.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, i32)> {
        let rescore_count = rescore_count(top_k, self.config.rescore_multiplier, self.count);
        let query_codes = self.quantize_vector(query);

        let mut scores: Vec<(usize, i32)> = (0..self.count)
            .map(|idx| (idx, self.approx_dot(&query_codes, idx)))
            .collect();

        scores.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        scores.truncate(rescore_count);
        scores
    }

    /// Rebuild codes from vectors.
    pub fn rebuild_codes(&mut self, vectors: &[&[f32]]) {
        self.codes.clear();
        self.codes.reserve(vectors.len() * self.bytes_per_vector);
        for vector in vectors {
            self.codes.extend_from_slice(&quantize_two_bit(
                vector,
                &self.boundaries,
                self.bytes_per_vector,
            ));
        }
        self.count = vectors.len();
    }

    /// Serialize parameters (boundaries only, codes rebuilt on load).
    pub fn write_params(&self, writer: &mut impl Write) -> std::io::Result<()> {
        // Tag byte: 4 = two_bit
        writer.write_all(&[4u8])?;
        write_usize(writer, self.dimension)?;
        write_usize(writer, self.config.rescore_multiplier)?;
        // Write boundaries (dimension * 3 floats)
        for &b in &self.boundaries {
            writer.write_all(&b.to_le_bytes())?;
        }
        Ok(())
    }

    /// Deserialize parameters.
    pub fn read_params(reader: &mut impl Read) -> std::io::Result<Self> {
        let dimension = read_usize(reader)?;
        let rescore_multiplier = read_usize(reader)?;
        let mut boundaries = vec![0.0_f32; dimension * 3];
        for b in &mut boundaries {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            *b = f32::from_le_bytes(buf);
        }
        let bytes_per_vector = (dimension + 3) / 4;
        Ok(Self {
            dimension,
            boundaries,
            codes: Vec::new(),
            count: 0,
            bytes_per_vector,
            config: TwoBitQuantizationConfig { rescore_multiplier },
        })
    }
}

// ---------------------------------------------------------------------------
// Multi-vector quantized index (for ColBERT token-level search)
// ---------------------------------------------------------------------------

/// Configuration for multi-vector quantization.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiVectorQuantizationConfig {
    TwoBit(TwoBitQuantizationConfig),
}

/// A quantized index for multi-vector (late interaction) search.
/// Stores all token vectors from all documents in a flat quantized array,
/// with a mapping from document index to token range.
#[derive(Clone, Debug)]
pub struct MultiVectorQuantizedIndex {
    pub quantizer: TwoBitQuantizer,
    /// For each document: (start_index, count) into the quantized vector array.
    pub doc_ranges: Vec<(usize, usize)>,
}

impl MultiVectorQuantizedIndex {
    /// Build a multi-vector quantized index from per-document token vectors.
    /// `doc_token_vectors[i]` is a slice of token-level vectors for document i.
    pub fn build(
        doc_token_vectors: &[&[Vec<f32>]],
        token_dimension: usize,
        config: &MultiVectorQuantizationConfig,
    ) -> Self {
        // Flatten all token vectors for training
        let all_tokens: Vec<&[f32]> = doc_token_vectors
            .iter()
            .flat_map(|tokens| tokens.iter().map(|v| v.as_slice()))
            .collect();

        let MultiVectorQuantizationConfig::TwoBit(cfg) = config;

        let quantizer = if all_tokens.is_empty() {
            // Empty case: create minimal quantizer
            TwoBitQuantizer {
                dimension: token_dimension,
                boundaries: vec![0.0; token_dimension * 3],
                codes: Vec::new(),
                count: 0,
                bytes_per_vector: (token_dimension + 3) / 4,
                config: cfg.clone(),
            }
        } else {
            TwoBitQuantizer::train(&all_tokens, token_dimension, cfg.clone())
        };

        // Build doc_ranges
        let mut doc_ranges = Vec::with_capacity(doc_token_vectors.len());
        let mut offset = 0;
        for tokens in doc_token_vectors {
            doc_ranges.push((offset, tokens.len()));
            offset += tokens.len();
        }

        Self {
            quantizer,
            doc_ranges,
        }
    }

    /// Compute approximate MaxSim score for a document given query token codes.
    /// For each query token, finds the max approximate dot with any document token.
    pub fn approx_maxsim(&self, query_codes: &[Vec<u8>], doc_idx: usize) -> i32 {
        let (start, count) = self.doc_ranges[doc_idx];
        if count == 0 || query_codes.is_empty() {
            return 0;
        }
        let mut total = 0i32;
        for q_code in query_codes {
            let mut best = i32::MIN;
            for i in start..start + count {
                let score = two_bit_approx_dot(
                    q_code,
                    &self.quantizer.codes[i * self.quantizer.bytes_per_vector
                        ..(i + 1) * self.quantizer.bytes_per_vector],
                    self.quantizer.dimension,
                );
                if score > best {
                    best = score;
                }
            }
            total += best;
        }
        total
    }

    /// Search: returns candidate document indices sorted by approximate MaxSim.
    pub fn search(&self, query_tokens: &[&[f32]], top_k: usize) -> Vec<usize> {
        let rescore_count = rescore_count(
            top_k,
            self.quantizer.config.rescore_multiplier,
            self.doc_ranges.len(),
        );
        if query_tokens.is_empty() || self.doc_ranges.is_empty() {
            return Vec::new();
        }

        let query_codes: Vec<Vec<u8>> = query_tokens
            .iter()
            .map(|t| self.quantizer.quantize_vector(t))
            .collect();

        let mut scores: Vec<(usize, i32)> = (0..self.doc_ranges.len())
            .map(|doc_idx| (doc_idx, self.approx_maxsim(&query_codes, doc_idx)))
            .collect();

        scores.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        scores.truncate(rescore_count);
        scores.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Rebuild from document token vectors (after loading parameters from disk).
    pub fn rebuild(&mut self, doc_token_vectors: &[&[Vec<f32>]]) {
        let all_tokens: Vec<&[f32]> = doc_token_vectors
            .iter()
            .flat_map(|tokens| tokens.iter().map(|v| v.as_slice()))
            .collect();
        self.quantizer.rebuild_codes(&all_tokens);

        self.doc_ranges.clear();
        let mut offset = 0;
        for tokens in doc_token_vectors {
            self.doc_ranges.push((offset, tokens.len()));
            offset += tokens.len();
        }
    }

    /// Serialize parameters.
    pub fn write_params(&self, writer: &mut impl Write) -> std::io::Result<()> {
        self.quantizer.write_params(writer)?;
        // Write doc_ranges
        write_usize(writer, self.doc_ranges.len())?;
        for &(start, count) in &self.doc_ranges {
            write_usize(writer, start)?;
            write_usize(writer, count)?;
        }
        Ok(())
    }

    /// Deserialize parameters.
    pub fn read_params(reader: &mut impl Read) -> std::io::Result<Self> {
        // Consume the tag byte written by TwoBitQuantizer::write_params
        let mut tag = [0u8; 1];
        reader.read_exact(&mut tag)?;
        if tag[0] != 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("expected two_bit tag (4), got {}", tag[0]),
            ));
        }
        let quantizer = TwoBitQuantizer::read_params(reader)?;
        let num_docs = read_usize(reader)?;
        let mut doc_ranges = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let start = read_usize(reader)?;
            let count = read_usize(reader)?;
            doc_ranges.push((start, count));
        }
        Ok(Self {
            quantizer,
            doc_ranges,
        })
    }
}

// ---------------------------------------------------------------------------
// Unified quantization index
// ---------------------------------------------------------------------------

/// A quantized index that wraps any of the three quantization strategies.
/// Used by the Database to accelerate search.
#[derive(Clone, Debug)]
pub enum QuantizedIndex {
    Scalar(ScalarQuantizer),
    Binary(BinaryQuantizer),
    Product(ProductQuantizer),
}

impl QuantizedIndex {
    /// Build a quantized index from vectors.
    pub fn build(vectors: &[&[f32]], dimension: usize, config: &QuantizationConfig) -> Self {
        match config {
            QuantizationConfig::Scalar(cfg) => {
                QuantizedIndex::Scalar(ScalarQuantizer::train(vectors, dimension, cfg.clone()))
            }
            QuantizationConfig::Binary(cfg) => {
                let mut quantizer = BinaryQuantizer::new(dimension, cfg.clone());
                quantizer.add_vectors(vectors);
                QuantizedIndex::Binary(quantizer)
            }
            QuantizationConfig::Product(cfg) => {
                QuantizedIndex::Product(ProductQuantizer::train(vectors, dimension, cfg.clone()))
            }
        }
    }

    /// Search the quantized index. Returns candidate indices sorted by
    /// approximate similarity (best first), to be rescored with exact vectors.
    pub fn search_candidates(&self, query: &[f32], top_k: usize) -> Vec<usize> {
        self.search_candidates_with_metric(query, top_k, DistanceMetric::Cosine)
    }

    /// Search the quantized index with the database metric.
    /// Returns candidate indices sorted by approximate score (best first).
    pub fn search_candidates_with_metric(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Vec<usize> {
        match self {
            QuantizedIndex::Scalar(q) => q
                .search_with_metric(query, top_k, metric)
                .into_iter()
                .map(|(i, _)| i)
                .collect(),
            QuantizedIndex::Binary(q) => {
                q.search(query, top_k).into_iter().map(|(i, _)| i).collect()
            }
            QuantizedIndex::Product(q) => {
                q.search(query, top_k).into_iter().map(|(i, _)| i).collect()
            }
        }
    }

    /// Rebuild quantized codes from current vectors (after deserialization or updates).
    pub fn rebuild_codes(&mut self, vectors: &[&[f32]]) {
        match self {
            QuantizedIndex::Scalar(q) => q.rebuild_codes(vectors),
            QuantizedIndex::Binary(q) => q.rebuild_codes(vectors),
            QuantizedIndex::Product(q) => q.rebuild_codes(vectors),
        }
    }

    /// Get the vector count in the quantized index.
    pub fn count(&self) -> usize {
        match self {
            QuantizedIndex::Scalar(q) => q.count,
            QuantizedIndex::Binary(q) => q.count,
            QuantizedIndex::Product(q) => q.count,
        }
    }

    /// Get the rescore multiplier for this quantization strategy.
    pub fn rescore_multiplier(&self) -> usize {
        match self {
            QuantizedIndex::Scalar(q) => q.config.rescore_multiplier,
            QuantizedIndex::Binary(q) => q.config.rescore_multiplier,
            QuantizedIndex::Product(q) => q.config.rescore_multiplier,
        }
    }

    /// Serialize quantization parameters to a writer.
    pub fn write_params(&self, writer: &mut impl Write) -> std::io::Result<()> {
        match self {
            QuantizedIndex::Scalar(q) => q.write_params(writer),
            QuantizedIndex::Binary(q) => q.write_params(writer),
            QuantizedIndex::Product(q) => q.write_params(writer),
        }
    }

    /// Deserialize quantization parameters from a reader.
    pub fn read_params(reader: &mut impl Read) -> std::io::Result<Self> {
        let mut tag = [0u8; 1];
        reader.read_exact(&mut tag)?;
        match tag[0] {
            1 => Ok(QuantizedIndex::Scalar(ScalarQuantizer::read_params(
                reader,
            )?)),
            2 => Ok(QuantizedIndex::Binary(BinaryQuantizer::read_params(
                reader,
            )?)),
            3 => Ok(QuantizedIndex::Product(ProductQuantizer::read_params(
                reader,
            )?)),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown quantization tag: {other}"),
            )),
        }
    }

    /// Get the config used to build this index.
    pub fn config(&self) -> QuantizationConfig {
        match self {
            QuantizedIndex::Scalar(q) => QuantizationConfig::Scalar(q.config.clone()),
            QuantizedIndex::Binary(q) => QuantizationConfig::Binary(q.config.clone()),
            QuantizedIndex::Product(q) => QuantizationConfig::Product(q.config.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helper functions
// ---------------------------------------------------------------------------

#[inline]
fn rescore_count(top_k: usize, rescore_multiplier: usize, count: usize) -> usize {
    top_k
        .max(1)
        .saturating_mul(rescore_multiplier.max(1))
        .min(count)
}

/// Quantize a single f32 value to u8 using the given min and scale.
#[inline]
fn quantize_scalar(val: f32, min: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        128 // midpoint for constant dimensions
    } else {
        ((val - min) * scale).clamp(0.0, 255.0) as u8
    }
}

#[inline]
fn dequantize_scalar(code: u8, min: f32, scale: f32) -> f32 {
    if scale == 0.0 {
        min
    } else {
        min + (code as f32 / scale)
    }
}

/// Convert a float vector to a binary representation (1 bit per dimension).
/// Positive values map to 1, non-positive to 0.
fn binarize_vector(vector: &[f32]) -> Vec<u8> {
    let bytes = (vector.len() + 7) / 8;
    let mut result = vec![0u8; bytes];
    for (i, &val) in vector.iter().enumerate() {
        if val > 0.0 {
            result[i / 8] |= 1 << (i % 8);
        }
    }
    result
}

/// Compute Hamming distance between two binary vectors.
#[inline]
fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0u32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dist += (ai ^ bi).count_ones();
    }
    dist
}

/// Quantize a float vector to 2-bit codes (4 levels per dimension).
/// Level mapping: val <= q25 → 0, val <= q50 → 1, val <= q75 → 2, else → 3.
/// Packed 4 dimensions per byte (least-significant bits first).
fn quantize_two_bit(vector: &[f32], boundaries: &[f32], bytes_per_vector: usize) -> Vec<u8> {
    let mut result = vec![0u8; bytes_per_vector];
    for (i, &val) in vector.iter().enumerate() {
        let b_offset = i * 3;
        let level = if val <= boundaries[b_offset] {
            0u8
        } else if val <= boundaries[b_offset + 1] {
            1u8
        } else if val <= boundaries[b_offset + 2] {
            2u8
        } else {
            3u8
        };
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        result[byte_idx] |= level << bit_offset;
    }
    result
}

/// Approximate dot product between two 2-bit quantized vectors.
/// Uses level values 0,1,2,3 as proxies for the original float magnitudes.
/// Higher score = more similar.
#[inline]
fn two_bit_approx_dot(a: &[u8], b: &[u8], dimension: usize) -> i32 {
    let mut sum = 0i32;
    for i in 0..dimension {
        let byte_idx = i / 4;
        let bit_offset = (i % 4) * 2;
        let a_level = ((a[byte_idx] >> bit_offset) & 0x03) as i32;
        let b_level = ((b[byte_idx] >> bit_offset) & 0x03) as i32;
        sum += a_level * b_level;
    }
    sum
}

/// Squared L2 distance between two vectors.
#[inline]
fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum()
}

/// Find the nearest centroid index for a sub-vector.
fn find_nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;
    for (idx, centroid) in centroids.iter().enumerate() {
        let dist = l2_distance_sq(vector, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    best_idx
}

/// Simple k-means clustering for PQ training.
fn kmeans(vectors: &[&[f32]], dimension: usize, k: usize, iterations: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let actual_k = k.min(n);

    // Initialize centroids using first k vectors (or modular selection if k > n)
    let mut centroids: Vec<Vec<f32>> = (0..actual_k).map(|i| vectors[i % n].to_vec()).collect();

    let mut assignments = vec![0usize; n];

    for _ in 0..iterations {
        // Assign each vector to nearest centroid
        for (i, vector) in vectors.iter().enumerate() {
            assignments[i] = find_nearest_centroid(vector, &centroids);
        }

        // Update centroids
        let mut new_centroids = vec![vec![0.0_f32; dimension]; actual_k];
        let mut counts = vec![0usize; actual_k];

        for (i, vector) in vectors.iter().enumerate() {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for (j, &val) in vector.iter().enumerate() {
                new_centroids[cluster][j] += val;
            }
        }

        for (cluster, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[cluster] > 0 {
                for val in centroid.iter_mut() {
                    *val /= counts[cluster] as f32;
                }
            } else {
                // Keep old centroid for empty clusters
                centroid.copy_from_slice(&centroids[cluster]);
            }
        }

        centroids = new_centroids;
    }

    // Pad with zeros if actual_k < k (shouldn't happen in practice)
    while centroids.len() < k {
        centroids.push(vec![0.0_f32; dimension]);
    }

    centroids
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn write_usize(writer: &mut impl Write, val: usize) -> std::io::Result<()> {
    writer.write_all(&(val as u64).to_le_bytes())
}

fn read_usize(reader: &mut impl Read) -> std::io::Result<usize> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf) as usize)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        // Simple deterministic pseudo-random generator (xorshift64)
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        // Map to [-1, 1]
                        (state as f32 / u64::MAX as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn scalar_quantization_basic() {
        let vectors = random_vectors(100, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let quantizer = ScalarQuantizer::train(&refs, 128, ScalarQuantizationConfig::default());

        assert_eq!(quantizer.count, 100);
        assert_eq!(quantizer.codes.len(), 100 * 128);
        assert_eq!(quantizer.dimension, 128);

        // Search should return candidates
        let query = &vectors[0];
        let results = quantizer.search(query, 10);
        assert!(!results.is_empty());
        // The query vector itself should be among the top results
        assert!(results.iter().take(5).any(|(idx, _)| *idx == 0));
    }

    #[test]
    fn binary_quantization_basic() {
        let vectors = random_vectors(100, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let mut quantizer = BinaryQuantizer::new(128, BinaryQuantizationConfig::default());
        quantizer.add_vectors(&refs);

        assert_eq!(quantizer.count, 100);
        assert_eq!(quantizer.bytes_per_vector, 16); // 128 / 8
        assert_eq!(quantizer.codes.len(), 100 * 16);

        // Search should find the query itself at distance 0
        let query = &vectors[0];
        let results = quantizer.search(query, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // First result should be index 0
        assert_eq!(results[0].1, 0); // Hamming distance 0
    }

    #[test]
    fn rescore_multiplier_controls_candidate_count_without_hidden_floor() {
        let vectors = random_vectors(200, 64, 7);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let scalar = ScalarQuantizer::train(
            &refs,
            64,
            ScalarQuantizationConfig {
                rescore_multiplier: 1,
            },
        );
        assert_eq!(scalar.search(&vectors[0], 10).len(), 10);

        let scalar = ScalarQuantizer::train(
            &refs,
            64,
            ScalarQuantizationConfig {
                rescore_multiplier: 4,
            },
        );
        assert_eq!(scalar.search(&vectors[0], 10).len(), 40);

        let mut binary = BinaryQuantizer::new(
            64,
            BinaryQuantizationConfig {
                rescore_multiplier: 2,
            },
        );
        binary.add_vectors(&refs);
        assert_eq!(binary.search(&vectors[0], 10).len(), 20);
    }

    #[test]
    fn product_quantization_basic() {
        let vectors = random_vectors(200, 128, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = ProductQuantizationConfig {
            num_sub_vectors: 8,
            num_centroids: 16, // Small for testing
            training_iterations: 5,
            rescore_multiplier: 10,
        };

        let quantizer = ProductQuantizer::train(&refs, 128, config);

        assert_eq!(quantizer.count, 200);
        assert_eq!(quantizer.num_sub_vectors, 8);
        assert_eq!(quantizer.sub_dimension, 16);
        assert_eq!(quantizer.codes.len(), 200 * 8);

        // Search should return candidates
        let query = &vectors[0];
        let results = quantizer.search(query, 10);
        assert!(!results.is_empty());
        // Query itself should be the closest (distance 0 with its own centroids)
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn quantized_index_build_and_search() {
        let vectors = random_vectors(100, 64, 123);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        // Test scalar
        let idx = QuantizedIndex::build(
            &refs,
            64,
            &QuantizationConfig::Scalar(ScalarQuantizationConfig::default()),
        );
        let candidates = idx.search_candidates(&vectors[0], 10);
        assert!(!candidates.is_empty());
        assert!(candidates.contains(&0));

        // Test binary
        let idx = QuantizedIndex::build(
            &refs,
            64,
            &QuantizationConfig::Binary(BinaryQuantizationConfig::default()),
        );
        let candidates = idx.search_candidates(&vectors[0], 10);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0], 0);

        // Test product
        let idx = QuantizedIndex::build(
            &refs,
            64,
            &QuantizationConfig::Product(ProductQuantizationConfig {
                num_sub_vectors: 8,
                num_centroids: 16,
                training_iterations: 5,
                rescore_multiplier: 10,
            }),
        );
        let candidates = idx.search_candidates(&vectors[0], 10);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0], 0);
    }

    #[test]
    fn serialization_roundtrip_scalar() {
        let vectors = random_vectors(50, 32, 99);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let original = QuantizedIndex::build(
            &refs,
            32,
            &QuantizationConfig::Scalar(ScalarQuantizationConfig {
                rescore_multiplier: 7,
            }),
        );

        let mut buf = Vec::new();
        original.write_params(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let restored = QuantizedIndex::read_params(&mut cursor).unwrap();

        match (&original, &restored) {
            (QuantizedIndex::Scalar(a), QuantizedIndex::Scalar(b)) => {
                assert_eq!(a.dimension, b.dimension);
                assert_eq!(a.mins, b.mins);
                assert_eq!(a.maxs, b.maxs);
                assert_eq!(a.config, b.config);
            }
            _ => panic!("type mismatch"),
        }
    }

    #[test]
    fn serialization_roundtrip_binary() {
        let original = QuantizedIndex::build(
            &[&[1.0, -1.0, 0.5, -0.3][..]],
            4,
            &QuantizationConfig::Binary(BinaryQuantizationConfig {
                rescore_multiplier: 12,
            }),
        );

        let mut buf = Vec::new();
        original.write_params(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let restored = QuantizedIndex::read_params(&mut cursor).unwrap();

        match (&original, &restored) {
            (QuantizedIndex::Binary(a), QuantizedIndex::Binary(b)) => {
                assert_eq!(a.dimension, b.dimension);
                assert_eq!(a.config, b.config);
            }
            _ => panic!("type mismatch"),
        }
    }

    #[test]
    fn serialization_roundtrip_product() {
        let vectors = random_vectors(50, 32, 77);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let original = QuantizedIndex::build(
            &refs,
            32,
            &QuantizationConfig::Product(ProductQuantizationConfig {
                num_sub_vectors: 4,
                num_centroids: 8,
                training_iterations: 3,
                rescore_multiplier: 5,
            }),
        );

        let mut buf = Vec::new();
        original.write_params(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let restored = QuantizedIndex::read_params(&mut cursor).unwrap();

        match (&original, &restored) {
            (QuantizedIndex::Product(a), QuantizedIndex::Product(b)) => {
                assert_eq!(a.dimension, b.dimension);
                assert_eq!(a.num_sub_vectors, b.num_sub_vectors);
                assert_eq!(a.num_centroids, b.num_centroids);
                assert_eq!(a.codebooks.len(), b.codebooks.len());
                for (ca, cb) in a.codebooks.iter().zip(b.codebooks.iter()) {
                    for (va, vb) in ca.iter().zip(cb.iter()) {
                        assert_eq!(va.len(), vb.len());
                        for (&fa, &fb) in va.iter().zip(vb.iter()) {
                            assert!((fa - fb).abs() < 1e-6);
                        }
                    }
                }
                assert_eq!(a.config, b.config);
            }
            _ => panic!("type mismatch"),
        }
    }

    #[test]
    fn hamming_distance_correctness() {
        assert_eq!(hamming_distance(&[0b00000000], &[0b00000000]), 0);
        assert_eq!(hamming_distance(&[0b11111111], &[0b00000000]), 8);
        assert_eq!(hamming_distance(&[0b10101010], &[0b01010101]), 8);
        assert_eq!(hamming_distance(&[0b10101010], &[0b10101010]), 0);
        assert_eq!(hamming_distance(&[0b10000000], &[0b00000000]), 1);
    }

    #[test]
    fn binarize_vector_correctness() {
        let v = vec![1.0, -1.0, 0.5, -0.5, 0.0, 0.1, -0.1, 0.9];
        let binary = binarize_vector(&v);
        assert_eq!(binary.len(), 1);
        // Bit 0: 1.0 > 0 -> 1
        // Bit 1: -1.0 -> 0
        // Bit 2: 0.5 > 0 -> 1
        // Bit 3: -0.5 -> 0
        // Bit 4: 0.0 -> 0 (not strictly positive)
        // Bit 5: 0.1 > 0 -> 1
        // Bit 6: -0.1 -> 0
        // Bit 7: 0.9 > 0 -> 1
        assert_eq!(binary[0], 0b10100101);
    }

    #[test]
    fn two_bit_quantization_basic() {
        let vectors = random_vectors(100, 64, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = TwoBitQuantizationConfig {
            rescore_multiplier: 4,
        };
        let quantizer = TwoBitQuantizer::train(&refs, 64, config);

        assert_eq!(quantizer.dimension, 64);
        assert_eq!(quantizer.count, 100);
        assert_eq!(quantizer.bytes_per_vector, 16); // 64 dims * 2 bits / 8 = 16
        assert_eq!(quantizer.boundaries.len(), 64 * 3);

        // Search should return candidates including the query itself
        let results = quantizer.search(&vectors[0], 10);
        assert!(!results.is_empty());
        assert!(results.iter().take(5).any(|(idx, _)| *idx == 0));
    }

    #[test]
    fn two_bit_quantize_and_approx_dot() {
        // Manually test quantization of a small vector
        let boundaries = vec![
            -0.5, 0.0, 0.5, // dim 0: quartiles
            -0.5, 0.0, 0.5, // dim 1
            -0.5, 0.0, 0.5, // dim 2
            -0.5, 0.0, 0.5, // dim 3
        ];
        let bytes_per_vector = 1; // 4 dims * 2 bits = 8 bits = 1 byte

        // Vector with values that map to different quantization levels
        let v1 = [-1.0, -0.25, 0.25, 1.0]; // levels: 0, 1, 2, 3
        let v2 = [-1.0, -0.25, 0.25, 1.0]; // levels: 0, 1, 2, 3

        let q1 = quantize_two_bit(&v1, &boundaries, bytes_per_vector);
        let q2 = quantize_two_bit(&v2, &boundaries, bytes_per_vector);

        // Same vectors should have the maximum approx dot product
        let dot = two_bit_approx_dot(&q1, &q2, 4);
        assert!(dot > 0); // 0*0 + 1*1 + 2*2 + 3*3 = 0 + 1 + 4 + 9 = 14
        assert_eq!(dot, 14);
    }

    #[test]
    fn two_bit_serialization_roundtrip() {
        use std::io::Read;

        let vectors = random_vectors(50, 32, 99);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = TwoBitQuantizationConfig {
            rescore_multiplier: 6,
        };
        let original = TwoBitQuantizer::train(&refs, 32, config);

        let mut buf = Vec::new();
        original.write_params(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        // Consume the tag byte written by write_params
        let mut tag = [0u8; 1];
        cursor.read_exact(&mut tag).unwrap();
        assert_eq!(tag[0], 4);
        let restored = TwoBitQuantizer::read_params(&mut cursor).unwrap();

        assert_eq!(original.dimension, restored.dimension);
        assert_eq!(original.boundaries.len(), restored.boundaries.len());
        for (a, b) in original.boundaries.iter().zip(restored.boundaries.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(
            original.config.rescore_multiplier,
            restored.config.rescore_multiplier
        );
    }

    #[test]
    fn multi_vector_quantized_index_basic() {
        // Create 5 "documents", each with 3-5 token vectors of dimension 16
        let mut doc_tokens: Vec<Vec<Vec<f32>>> = Vec::new();
        for doc_idx in 0..5 {
            let n_tokens = 3 + (doc_idx % 3); // 3, 4, 5, 3, 4 tokens
            let tokens = random_vectors(n_tokens, 16, 100 + doc_idx as u64);
            doc_tokens.push(tokens);
        }

        let doc_refs: Vec<&[Vec<f32>]> = doc_tokens.iter().map(|v| v.as_slice()).collect();
        let config = MultiVectorQuantizationConfig::TwoBit(TwoBitQuantizationConfig {
            rescore_multiplier: 4,
        });

        let index = MultiVectorQuantizedIndex::build(&doc_refs, 16, &config);

        assert_eq!(index.doc_ranges.len(), 5);
        // Total token count: 3+4+5+3+4 = 19
        let total_tokens: usize = index.doc_ranges.iter().map(|(_, count)| count).sum();
        assert_eq!(total_tokens, 19);

        // Search with a query that matches document 0's tokens
        let query_tokens: Vec<&[f32]> = doc_tokens[0].iter().map(Vec::as_slice).collect();
        let results = index.search(&query_tokens, 3);
        assert!(!results.is_empty());
        // Document 0 should be among top results (its own tokens should
        // score highest MaxSim against themselves)
        assert!(results.iter().take(3).any(|&idx| idx == 0));
    }

    #[test]
    fn multi_vector_quantized_index_serialization_roundtrip() {
        let mut doc_tokens: Vec<Vec<Vec<f32>>> = Vec::new();
        for i in 0..3 {
            doc_tokens.push(random_vectors(4, 8, 200 + i));
        }
        let doc_refs: Vec<&[Vec<f32>]> = doc_tokens.iter().map(|v| v.as_slice()).collect();

        let config = MultiVectorQuantizationConfig::TwoBit(TwoBitQuantizationConfig {
            rescore_multiplier: 2,
        });
        let original = MultiVectorQuantizedIndex::build(&doc_refs, 8, &config);

        let mut buf = Vec::new();
        original.write_params(&mut buf).unwrap();

        let mut cursor = std::io::Cursor::new(&buf);
        let restored = MultiVectorQuantizedIndex::read_params(&mut cursor).unwrap();

        assert_eq!(original.doc_ranges, restored.doc_ranges);
        assert_eq!(original.quantizer.dimension, restored.quantizer.dimension);
        assert_eq!(
            original.quantizer.boundaries.len(),
            restored.quantizer.boundaries.len()
        );
        assert_eq!(
            original.quantizer.config.rescore_multiplier,
            restored.quantizer.config.rescore_multiplier,
        );
    }
}
