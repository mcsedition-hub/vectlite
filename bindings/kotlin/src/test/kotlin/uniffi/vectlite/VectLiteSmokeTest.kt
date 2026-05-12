package uniffi.vectlite

import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class VectLiteSmokeTest {
    @Test
    fun databaseRoundTripAndSearch() {
        val root = Files.createTempDirectory("vectlite-kotlin-")
        try {
            val dbPath = root.resolve("smoke.vdb").toString()
            val db = Database.openOrCreate(dbPath, 3u, "cosine")
            try {
                assertEquals(3u, db.dimension())
                assertEquals("cosine", db.metric())

                db.upsert(
                    "doc1",
                    listOf(1.0f, 0.0f, 0.0f),
                    """{"source":"kotlin","rank":1}""",
                    null,
                    null,
                )
                db.upsert(
                    "doc2",
                    listOf(0.0f, 1.0f, 0.0f),
                    """{"source":"other","rank":2}""",
                    "alt",
                    null,
                )

                assertEquals(1u, db.count(null, """{"source":"kotlin"}"""))

                val record = assertNotNull(db.get("doc1", null))
                assertEquals("doc1", record.id)
                assertEquals("", record.namespace)

                val results = db.search(
                    listOf(1.0f, 0.0f, 0.0f),
                    1u,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                )
                assertEquals("doc1", results.first().id)

                val page = db.listCursor("", null, 10u, null)
                assertEquals(listOf("doc1"), page.records.map { it.id })
                assertNull(page.cursor)
            } finally {
                db.close()
            }
        } finally {
            root.toFile().deleteRecursively()
        }
    }
}
