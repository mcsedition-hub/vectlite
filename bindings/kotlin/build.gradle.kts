plugins {
    kotlin("jvm") version "2.0.21"
    `maven-publish`
}

group = "org.mcsedition"
version = "0.9.2"

val workspaceDir = layout.projectDirectory.dir("../..").asFile
val nativeProfile = providers.gradleProperty("nativeProfile").orElse("debug").get()
val generatedUniffiKotlin = layout.projectDirectory.file("../uniffi/generated/kotlin/uniffi/vectlite/vectlite.kt")
val patchedUniffiKotlinDir = layout.buildDirectory.dir("generated/uniffi/kotlin")
val patchedUniffiKotlin = patchedUniffiKotlinDir.map { it.file("uniffi/vectlite/vectlite.kt") }

fun nativeLibraryFileName(): String {
    val osName = System.getProperty("os.name").lowercase()
    return when {
        osName.contains("mac") || osName.contains("darwin") -> "libvectlite_uniffi.dylib"
        osName.contains("windows") -> "vectlite_uniffi.dll"
        else -> "libvectlite_uniffi.so"
    }
}

val nativeLibrary = workspaceDir.resolve("target/$nativeProfile/${nativeLibraryFileName()}")

val prepareUniffiKotlin by tasks.registering {
    description = "Copies the UniFFI Kotlin binding and patches Database.close() overload collision."
    inputs.file(generatedUniffiKotlin)
    outputs.file(patchedUniffiKotlin)

    doLast {
        val outputFile = patchedUniffiKotlin.get().asFile
        outputFile.parentFile.mkdirs()

        val autoCloseBlock = """
    @Synchronized
    override fun close() {
        this.destroy()
    }

""".trimStart()

        outputFile.writeText(
            generatedUniffiKotlin.asFile
                .readText()
                .replaceFirst(autoCloseBlock, "")
        )
    }
}

kotlin {
    // Compile with whatever JDK the host provides but emit JDK 17 bytecode
    // so the library runs on Java 17+.  We intentionally do NOT set
    // jvmToolchain(17) because that forces Gradle to locate or provision a
    // JDK 17, which breaks on machines that only have a newer JDK (e.g. 20+)
    // and no toolchain resolver configured.
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
    sourceSets {
        val main by getting {
            kotlin.srcDir(patchedUniffiKotlinDir)
        }
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
    withSourcesJar()
}

dependencies {
    implementation("net.java.dev.jna:jna:5.14.0")
    testImplementation(kotlin("test-junit5"))
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.register<Exec>("buildNative") {
    description = "Builds the vectlite UniFFI native library for the current host."
    group = "build"
    workingDir = workspaceDir
    executable = "cargo"
    args("build", "-p", "vectlite-uniffi")
    if (nativeProfile == "release") {
        args("--release")
    }
}

tasks.withType<Test>().configureEach {
    dependsOn("buildNative")
    useJUnitPlatform()
    systemProperty("uniffi.component.vectlite.libraryOverride", nativeLibrary.absolutePath)
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    dependsOn(prepareUniffiKotlin)
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            artifactId = "vectlite-kotlin"
        }
    }
}
