# Group HDF5 Alignment Manifest v1

Status: Draft implementation contract for `gds-h5/1.0`.

## Scope

`gds-h5/1.0` adds a language-neutral alignment-family representation under
`/gds/alignments`. It is deliberately separate from the legacy fmrigds R object
serialization stored at `/gds/alignments/<family>/serialized`.

## Writer Layout

Native Python alignment families are written as:

```text
/gds/alignments
  attrs:
    format = "fmrimod.alignment_manifest"
    schema_version = "gds-h5/1.0"
  /<family>
    attrs:
      format = "fmrimod.alignment_manifest"
      schema_version = "gds-h5/1.0-alignment"
    manifest            JSON string
    /matrices/<name>    float64 numeric matrix dataset
```

The `manifest` dataset is JSON. Each entry names a matrix dataset by relative
path:

```json
{
  "schema_version": "gds-h5/1.0-alignment",
  "format": "fmrimod.alignment_manifest",
  "family": "native_to_mni",
  "entries": [
    {
      "name": "affine",
      "kind": "affine",
      "source_space": "native",
      "target_space": "MNI152NLin2009cAsym",
      "matrix_dataset": "matrices/affine",
      "shape": [4, 4],
      "dtype": "float64"
    }
  ]
}
```

Python writes this layout from `GroupDataset.metadata["alignment_families"]`.
Each family is a mapping with an `entries` sequence. Each entry must include a
2-D `matrix`; all non-matrix fields must be JSON-serializable.

## Reader Semantics

`read_hdf5()` reconstructs portable families into
`metadata["alignment_families"]`, replacing each entry's `matrix_dataset` link
with a loaded float64 `matrix` while retaining the dataset path and manifest
fields.

Legacy fmrigds files that contain R serialized alignment blobs are not
semantically decoded. By default, a file with
`/gds/alignments/<family>/serialized` raises `UnsupportedGroupFeatureError`.
Passing `allow_opaque_alignments=True` preserves the raw serialized payload
under `metadata["opaque_alignment_families"]`; this is a transport mechanism,
not semantic map-family support.

## Schema Negotiation

The reader accepts scoped `gds-h5/0.x` files and `gds-h5/1.x` files. New
alignment manifests require the `gds-h5/1.0-alignment` manifest version. Future
major schema versions must fail explicitly until supported.
