"""Materialize only the eight audited public Git blobs in local pins/. No data downloads."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

FILES = [
  [
    "ofdm",
    "hypothesis_frontend",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/hypothesis_frontend.py",
    "735233d5ad34f77788f7ea2536048eb489be34ae"
  ],
  [
    "ofdm",
    "group_b_posterior_fusion",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_b_posterior_fusion.py",
    "dd32790b893bb432376622f2bf7a85561955df0a"
  ],
  [
    "ofdm",
    "ofdm_data",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/ofdm_data.py",
    "e2ed1a2e7a71415ef205498bb3f104827a1468ae"
  ],
  [
    "ofdm",
    "group_c_transport_attention",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/group_c_transport_attention.py",
    "6dd69afa4a1747e2072bd3c1766ecf08dafa85f0"
  ],
  [
    "pluto",
    "receivers",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/receivers.py",
    "4569d3dfdb5e7e1d3a2552cdb30b1c25f72b35c5"
  ],
  [
    "ofdm",
    "stress_channel",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/stress_channel.py",
    "898bccc0c4d4bcd1ced2100272c5ea4a4977ea06"
  ],
  [
    "pluto",
    "evaluator",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/evaluator.py",
    "03c9ee07aa79e6c8dc8e3454b53e4a576a35d0db"
  ],
  [
    "pluto",
    "waveform_frontend",
    "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/waveform_frontend.py",
    "a3752a434132f2730e807b215c4487e6e41fbd2b"
  ]
]
root = Path(__file__).resolve().parent / "pins"
for group, name, url, expected in FILES:
    dest = root / group / (name + ".py")
    if dest.exists():
        data = dest.read_bytes()
    else:
        with urlopen(url, timeout=30) as response:
            data = response.read()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != expected:
        raise ValueError(f"Git blob mismatch: {dest}")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as out:
            out.write(data)
    print(group, name, digest)
