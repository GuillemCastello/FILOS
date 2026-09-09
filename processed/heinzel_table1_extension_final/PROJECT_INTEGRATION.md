# Project integration

The complete supplied calculation package is preserved in this directory.
The runtime copy is:

`synthetic_filaments/data/heinzel_calibrated_extension_1_to_100Mm.csv`

Its approved SHA-256 checksum is:

`eb53e508e39774ceee93ebcc9ffc7faf83cd40c6c07631c1a88494c7d4d8d684`

The loader in `synthetic_filaments/opacity_table.py` checks the checksum,
required columns, complete Cartesian grid, finite values, ionization bounds,
and positive $f$ values before the table can be used. The scientific regression
also compares the runtime copy byte-for-byte with the archived master and
requires exact agreement with all 75 published Table-1 anchors.

Only the $i$ and $f$ columns replace the old hard-coded opacity table. This
package does not extend the source-function values, so the source function
remains separately based on the published 10, 20, and 30 Mm values.
