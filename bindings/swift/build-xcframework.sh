#!/usr/bin/env bash
#
# build-xcframework.sh — Cross-compile vectlite-uniffi for Apple platforms
# and produce VectLiteFFI.xcframework for use with Swift Package Manager.
#
# Prerequisites:
#   rustup target add aarch64-apple-darwin         # macOS Apple Silicon
#   rustup target add x86_64-apple-darwin          # macOS Intel
#   rustup target add aarch64-apple-ios            # iOS device
#   rustup target add aarch64-apple-ios-sim        # iOS simulator (arm64)
#   rustup target add x86_64-apple-ios             # iOS simulator (x86_64)
#
# Usage:
#   cd bindings/swift
#   ./build-xcframework.sh           # debug build
#   ./build-xcframework.sh --release # release build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
UNIFFI_DIR="$ROOT_DIR/bindings/uniffi"
SWIFT_DIR="$SCRIPT_DIR"

# Parse args
PROFILE="debug"
CARGO_FLAG=""
if [[ "${1:-}" == "--release" ]]; then
    PROFILE="release"
    CARGO_FLAG="--release"
fi

echo "==> Building vectlite-uniffi ($PROFILE) for Apple platforms..."

# The targets we build for.
# Adjust this list based on what you need.
TARGETS=(
    aarch64-apple-darwin        # macOS arm64
    x86_64-apple-darwin         # macOS x86_64
    aarch64-apple-ios           # iOS device
    aarch64-apple-ios-sim       # iOS simulator arm64
    x86_64-apple-ios            # iOS simulator x86_64
)

# Build each target
for target in "${TARGETS[@]}"; do
    echo "--- Building for $target ---"
    cargo build -p vectlite-uniffi $CARGO_FLAG --target "$target"
done

LIB_NAME="libvectlite_uniffi.a"

# Create fat libraries where needed (simulator = arm64 + x86_64)
STAGING="$SWIFT_DIR/.build-staging"
rm -rf "$STAGING"
mkdir -p "$STAGING/macos" "$STAGING/ios-device" "$STAGING/ios-simulator"

# macOS universal (arm64 + x86_64)
lipo -create \
    "$ROOT_DIR/target/aarch64-apple-darwin/$PROFILE/$LIB_NAME" \
    "$ROOT_DIR/target/x86_64-apple-darwin/$PROFILE/$LIB_NAME" \
    -output "$STAGING/macos/$LIB_NAME"

# iOS device (arm64 only)
cp "$ROOT_DIR/target/aarch64-apple-ios/$PROFILE/$LIB_NAME" \
   "$STAGING/ios-device/$LIB_NAME"

# iOS simulator universal (arm64 + x86_64)
lipo -create \
    "$ROOT_DIR/target/aarch64-apple-ios-sim/$PROFILE/$LIB_NAME" \
    "$ROOT_DIR/target/x86_64-apple-ios/$PROFILE/$LIB_NAME" \
    -output "$STAGING/ios-simulator/$LIB_NAME"

# Copy headers + modulemap into each staging folder
HEADER="$UNIFFI_DIR/generated/swift/vectliteFFI.h"
MODULEMAP="$UNIFFI_DIR/generated/swift/vectliteFFI.modulemap"

for platform in macos ios-device ios-simulator; do
    mkdir -p "$STAGING/$platform/Headers"
    cp "$HEADER" "$STAGING/$platform/Headers/"
    cp "$MODULEMAP" "$STAGING/$platform/Headers/module.modulemap"
done

# Build XCFramework
XCFRAMEWORK="$SWIFT_DIR/VectLiteFFI.xcframework"
rm -rf "$XCFRAMEWORK"

xcodebuild -create-xcframework \
    -library "$STAGING/macos/$LIB_NAME" \
    -headers "$STAGING/macos/Headers" \
    -library "$STAGING/ios-device/$LIB_NAME" \
    -headers "$STAGING/ios-device/Headers" \
    -library "$STAGING/ios-simulator/$LIB_NAME" \
    -headers "$STAGING/ios-simulator/Headers" \
    -output "$XCFRAMEWORK"

rm -rf "$STAGING"

echo ""
echo "==> VectLiteFFI.xcframework created at:"
echo "    $XCFRAMEWORK"
echo ""
echo "The Swift package at $SWIFT_DIR is now ready to use."
echo "Add it as a local dependency or push to a repository."
