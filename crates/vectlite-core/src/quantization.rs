//! Vector quantization module for memory-efficient similarity search.
//!
//! Supports three quantization strategies:
//! - **Scalar (int8)**: 4x memory reduction with minimal recall loss
//! - **Binary**: 32x memory reduction, uses Hamming distance for fast filtering
//! - **Product Quantization (PQ)**: Configurable compression for very large datasets
//!
//! All strategies support a 2-stage pipeline: fast quantized search followed by
//! exact float32 rescoring of top candidates.

use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for enabling quantization on a database.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantizationConfig {
    /// Scalar quantization: maps each f32 dimension to int8 using per-dimension
    /// min/max calibration. 4x memory reduction.
    Scalar(ScalarQuantizationConfig),
    /// Binary quantization: maps each f32 dimension to a single bit.
    /// 32x memory reduction. Best for high-dimensional normalized embeddings.
    Binary(BinaryQuantizationConfig),
    /// Product quantization: splits vector into sub-vectors and quantizes each
    /// to a centroid index. Highest compression for large datasets.
    Product(ProductQuantizationConfig),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarQuantizationConfig {
    /// Number of top candidates from quantized search to rescore with float32.
    /// Default: 5x top_k (minimum 100).
    pub rescore_multiplier: usize,
}

impl Default for ScalarQuantizationConfig {
    fn default() -> Self {
        Self {
            rescore_multiplier: 5,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryQuantizationConfig {
    /// Number of top candidates from Hamming search to rescore with float32.
    /// Default: 10x top_k (minimum 100).
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

    /// Compute approximate cosine distance between a quantized query and all stored vectors.
    /// Returns indices sorted by approximate similarity (best first).
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let rescore_count = (top_k * self.config.rescore_multiplier)
            .max(100)
            .min(self.count);
        let query_quantized = self.quantize_query(query);
        let mut scores: Vec<(usize, f32)> = (0..self.count)
            .map(|idx| {
                let offset = idx * self.dimension;
                let code_slice = &self.codes[offset..offset + self.dimension];
                let sim = scalar_quantized_dot(&query_quantized, code_slice);
                (idx, sim)
            })
            .collect();

        // Partial sort: get top rescore_count candidates
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scores.truncate(rescore_count);
        scores
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
        let rescore_count = (top_k * self.config.rescore_multiplier)
            .max(100)
            .min(self.count);
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
        let rescore_count = (top_k * self.config.rescore_multiplier)
            .max(100)
            .min(self.count);
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
        match self {
            QuantizedIndex::Scalar(q) => {
                q.search(query, top_k).into_iter().map(|(i, _)| i).collect()
            }
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

/// Quantize a single f32 value to u8 using the given min and scale.
#[inline]
fn quantize_scalar(val: f32, min: f32, scale: f32) -> u8 {
    if scale == 0.0 {
        128 // midpoint for constant dimensions
    } else {
        ((val - min) * scale).clamp(0.0, 255.0) as u8
    }
}

/// Approximate dot product between two u8-quantized vectors.
/// Higher value = more similar (analogous to cosine similarity for normalized vectors).
#[inline]
fn scalar_quantized_dot(a: &[u8], b: &[u8]) -> f32 {
    let mut sum = 0i32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        sum += (ai as i32) * (bi as i32);
    }
    sum as f32
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
}
