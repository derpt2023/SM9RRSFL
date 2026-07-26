"""Build the native SM3 accelerator and SM9 v2 group-operation bridge."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys
import zipfile

from setuptools import Extension, find_packages, setup
from setuptools.command.build import build as _build


PROJECT_ROOT = Path(__file__).resolve().parent
# Setuptools may execute this file from a PEP 517 temporary working directory.
# Anchor every relative extension/package path to the checked-out project.
os.chdir(PROJECT_ROOT)
PINNED_GMSSL_ARCHIVE_SHA256 = (
    "6dc97c6b4f7d2f6df9d44f014cca0561a7b4776017efd4486d341e986051fab4"
)
MAX_GMSSL_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
REQUIRED_GMSSL_FILES = (
    "include/gmssl/sm9_z256.h",
    "include/gmssl/version.h",
    "src/sm9_z256.c",
    "src/sm9_z256_table.c",
    "src/sm3.c",
    "src/debug.c",
    "src/rand.c",
    "src/rand_win.c",
)


def _validate_gmssl_source(candidate: Path) -> Path:
    root = candidate.expanduser().resolve()
    missing = [name for name in REQUIRED_GMSSL_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"incomplete GmSSL source tree at {root}: missing {missing[0]}"
        )
    version = (root / "include/gmssl/version.h").read_text(
        encoding="utf-8",
        errors="strict",
    )
    if 'GMSSL_VERSION_STR\t"GmSSL 3.3.0-dev.1183"' not in version:
        raise RuntimeError(
            "the SM9 bridge is pinned to GmSSL 3.3.0-dev.1183; "
            "supply that source revision"
        )
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pinned_archive(archive: Path) -> Path:
    digest = _sha256(archive)
    if digest != PINNED_GMSSL_ARCHIVE_SHA256:
        raise RuntimeError(
            "GmSSL archive digest mismatch; expected the pinned "
            "3.3.0-dev.1183 source archive"
        )

    cache = PROJECT_ROOT / "build/gmssl-v2-source" / digest
    extracted_root = cache / "GmSSL-master"
    if cache.exists():
        return _validate_gmssl_source(extracted_root)

    cache.parent.mkdir(parents=True, exist_ok=True)
    staging = cache.parent / f".{digest}.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if sum(member.file_size for member in members) > MAX_GMSSL_UNCOMPRESSED_BYTES:
                raise RuntimeError("GmSSL archive exceeds the extraction size limit")
            for member in members:
                relative = PurePosixPath(member.filename)
                if (
                    not relative.parts
                    or relative.is_absolute()
                    or relative.parts[0] != "GmSSL-master"
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise RuntimeError("GmSSL archive contains an unsafe path")
                file_type = (member.external_attr >> 16) & 0o170000
                if file_type == 0o120000 or member.flag_bits & 0x1:
                    raise RuntimeError("GmSSL archive contains a link or encrypted member")
                target = staging.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        _validate_gmssl_source(staging / "GmSSL-master")
        os.replace(staging, cache)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _validate_gmssl_source(extracted_root)


def find_gmssl_source() -> Path | None:
    """Locate an explicit source tree or unpack the pinned source archive."""

    if os.environ.get("SM9_SKIP_NATIVE") == "1":
        return None
    if os.environ.get("GMSSL_SOURCE"):
        return _validate_gmssl_source(Path(os.environ["GMSSL_SOURCE"]))

    archive = Path(
        os.environ.get(
            "GMSSL_ARCHIVE",
            str(Path.home() / "Downloads/GmSSL-master.zip"),
        )
    ).expanduser()
    if not archive.exists():
        if os.environ.get("GMSSL_ARCHIVE"):
            raise FileNotFoundError(f"GMSSL_ARCHIVE does not exist: {archive}")
        return None
    return _extract_pinned_archive(archive.resolve())


compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]
extensions = [
    Extension(
        "sm9rrsfl._native_sm3",
        sources=["sm9rrsfl/_native_sm3.c"],
        extra_compile_args=compile_args,
        optional=True,
    )
]

gmssl_source = find_gmssl_source()
if gmssl_source is not None:
    rand_source = "rand_win.c" if sys.platform == "win32" else "rand.c"
    extensions.append(
        Extension(
            "sm9rrsfl._native_sm9",
            sources=[
                "sm9rrsfl/_native_sm9.c",
                str(gmssl_source / "src/sm9_z256.c"),
                str(gmssl_source / "src/sm9_z256_table.c"),
                str(gmssl_source / "src/sm3.c"),
                str(gmssl_source / "src/debug.c"),
                str(gmssl_source / f"src/{rand_source}"),
            ],
            include_dirs=[str(gmssl_source / "include")],
            define_macros=[("DEBUG", "0")],
            extra_compile_args=compile_args,
            libraries=["advapi32"] if sys.platform == "win32" else [],
            optional=False,
        )
    )

# ``_native_rrs.c`` is the retired v1 trapdoor protocol.  It is deliberately
# not registered as an extension; v2 keeps one protocol implementation in
# Python and accelerates only the standard SM9 group primitives above.


class ProtocolBuild(_build):
    """Remove retired extension artefacts from reused setuptools build trees."""

    def run(self) -> None:
        super().run()
        package_dir = Path(self.build_lib) / "sm9rrsfl"
        for pattern in ("_native_rrs*.so", "_native_rrs*.pyd", "_native_rrs*.dll"):
            for stale_extension in package_dir.glob(pattern):
                stale_extension.unlink()


setup(
    name="sm9rrsfl",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=extensions,
    exclude_package_data={
        "sm9rrsfl": [
            "_native_rrs*.so",
            "_native_rrs*.pyd",
            "_native_rrs*.dll",
        ]
    },
    cmdclass={"build": ProtocolBuild},
)
