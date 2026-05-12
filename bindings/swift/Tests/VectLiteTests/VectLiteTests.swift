import XCTest
@testable import VectLite

final class VectLiteTests: XCTestCase {
    func testDatabaseRoundTripAndSearch() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vectlite-swift-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let dbPath = root.appendingPathComponent("smoke.vdb").path
        let db = try Database.openOrCreate(path: dbPath, dimension: 3, metric: "cosine")
        defer { try? db.close() }

        XCTAssertEqual(db.dimension(), 3)
        XCTAssertEqual(db.metric(), "cosine")

        try db.upsert(
            id: "doc1",
            vector: [1.0, 0.0, 0.0],
            metadataJson: #"{"source":"swift","rank":1}"#,
            namespace: nil,
            ttl: nil
        )
        try db.upsert(
            id: "doc2",
            vector: [0.0, 1.0, 0.0],
            metadataJson: #"{"source":"other","rank":2}"#,
            namespace: "alt",
            ttl: nil
        )

        XCTAssertEqual(db.count(namespace: nil, filterJson: #"{"source":"swift"}"#), 1)

        let record = db.get(id: "doc1", namespace: nil)
        XCTAssertEqual(record?.id, "doc1")
        XCTAssertEqual(record?.namespace, "")

        let results = try db.search(
            query: [1.0, 0.0, 0.0],
            k: 1,
            filterJson: nil,
            namespace: nil,
            sparseJson: nil,
            fusion: nil,
            denseWeight: nil,
            sparseWeight: nil,
            mmrLambda: nil
        )
        XCTAssertEqual(results.first?.id, "doc1")

        let page = try db.listCursor(namespace: "", filterJson: nil, limit: 10, cursor: nil)
        XCTAssertEqual(page.records.map(\.id), ["doc1"])
        XCTAssertNil(page.cursor)
    }
}
