"""Sequence handling, Tier-0 (spec section 03.2).

    "Biopython is *in Pyodide* -- a student can read a FASTA, translate it and
    check their codon-wheel answer from G9-L7 entirely client-side."

True, and this module still does not import it. Biopython in Pyodide is a
multi-megabyte download on a metered Indian school line (spec section 02,
constraint 2) for four functions a school actually needs. FASTA parsing,
translation, ORF finding and GC content are eighty lines of standard library,
so the first lesson loads in seconds instead of minutes. Biopython stays
available in Tier 1 for the things that genuinely need it.

The translation table is the full NCBI table 11 (bacterial/phage), because
phage genomes really do use GTG and TTG starts and a student who annotates one
by hand will meet them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .transcript import bound

__all__ = ["parse_fasta", "translate", "find_orfs", "gc_content", "reverse_complement",
           "codon_usage", "CODON_TABLE"]

_BASES = set("ACGTU")
_COMPLEMENT = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")

CODON_TABLE: Dict[str, str] = {}
_B = "TCAG"
_AA = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
for _i, _a in enumerate(_AA):
    CODON_TABLE["".join((_B[_i // 16], _B[(_i // 4) % 4], _B[_i % 4]))] = _a

# NCBI table 11: bacterial, archaeal and plant plastid code. Alternative starts.
START_CODONS = {"ATG", "GTG", "TTG", "ATT", "CTG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}

AA_NAMES = {
    "A": "Alanine", "R": "Arginine", "N": "Asparagine", "D": "Aspartate", "C": "Cysteine",
    "Q": "Glutamine", "E": "Glutamate", "G": "Glycine", "H": "Histidine", "I": "Isoleucine",
    "L": "Leucine", "K": "Lysine", "M": "Methionine", "F": "Phenylalanine", "P": "Proline",
    "S": "Serine", "T": "Threonine", "W": "Tryptophan", "Y": "Tyrosine", "V": "Valine",
    "*": "STOP",
}


def _clean(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in _BASES).replace("U", "T")


@bound("parse_fasta", data_args=("text",))
def parse_fasta(text: str, max_records: int = 500) -> Dict[str, Any]:
    """Read a FASTA file. Reports what it skipped rather than silently dropping it."""
    records: List[Dict[str, Any]] = []
    name, parts = None, []
    skipped = 0

    def flush():
        nonlocal name, parts, skipped
        if name is None:
            return
        raw = "".join(parts)
        seq = _clean(raw)
        bad = len(re.sub(r"\s", "", raw)) - len(seq)
        records.append({
            "id": name.split()[0], "description": name,
            "sequence": seq, "length": len(seq),
            "non_standard_bases": bad,
        })
        name, parts = None, []

    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            if len(records) >= max_records:
                skipped += 1
                name = None
                continue
            name = line[1:].strip() or f"record_{len(records) + 1}"
        elif name is not None:
            parts.append(line)
    flush()

    if not records:
        return {"refused": True,
                "reason": ("No FASTA records found. A FASTA file has a '>' line naming each "
                           "sequence, then the sequence itself on the lines below it.")}

    total = sum(r["length"] for r in records)
    odd = [r for r in records if r["non_standard_bases"] > 0]
    return {
        "records": records, "n_records": len(records), "total_length": total,
        "skipped_over_limit": skipped,
        "notes": ([f"{len(odd)} record(s) contained characters that are not A, C, G, T or U "
                   f"(often N for 'unknown base'). Those characters were left out of the "
                   f"sequence, and the lengths above reflect that."] if odd else []),
        "plain_language": (
            f"Read {len(records)} sequence{'s' if len(records) != 1 else ''}, "
            f"{total:,} bases in total"
            + (f". Longest: {max(records, key=lambda r: r['length'])['id']} at "
               f"{max(r['length'] for r in records):,} bases." if records else ".")),
        "p_value": None,
    }


@bound("reverse_complement", data_args=("sequence",))
def reverse_complement(sequence: str) -> Dict[str, Any]:
    seq = _clean(sequence)
    return {"sequence": seq.translate(_COMPLEMENT)[::-1], "length": len(seq), "p_value": None}


@bound("gc_content", data_args=("sequence",))
def gc_content(sequence: str, window: int = 0) -> Dict[str, Any]:
    """GC content, optionally as a sliding window.

    G8-L11 is about DNA stability: G-C pairs have three hydrogen bonds, A-T have
    two, so a GC-rich genome holds together at a higher temperature. The sliding
    window makes that visible along a real genome.
    """
    seq = _clean(sequence)
    if not seq:
        return {"refused": True, "reason": "No usable bases in the sequence."}
    gc = sum(seq.count(b) for b in "GC")
    out: Dict[str, Any] = {
        "gc_content": gc / len(seq), "length": len(seq),
        "counts": {b: seq.count(b) for b in "ACGT"},
        "melting_temperature_estimate_C": (
            float(64.9 + 41 * (gc - 16.4) / len(seq)) if len(seq) > 13 else
            float(2 * (seq.count("A") + seq.count("T")) + 4 * gc)),
        "teaches": ("G pairs with C using three hydrogen bonds; A pairs with T using two. More GC "
                    "means a stiffer, more heat-stable molecule -- which is why the melting "
                    "temperature above moves with GC and not with length alone."),
        "p_value": None,
    }
    if window and window >= 10:
        w = int(min(window, len(seq)))
        step = max(1, w // 4)
        xs, ys = [], []
        for i in range(0, len(seq) - w + 1, step):
            sub = seq[i:i + w]
            xs.append(i + w // 2)
            ys.append(sum(sub.count(b) for b in "GC") / w)
        out["window"] = {"size": w, "step": step, "positions": xs, "gc": ys}
        if ys:
            out["window"]["most_gc_rich_position"] = int(xs[int(max(range(len(ys)), key=ys.__getitem__))])
    tail = "."
    win = out.get("window")
    if win and win["gc"]:
        tail = (f", ranging {min(win['gc']):.1%} to {max(win['gc']):.1%} across the genome in "
                f"{win['size']}-base windows.")
    out["plain_language"] = f"{len(seq):,} bases, {out['gc_content']:.1%} G+C{tail}"
    return out


@bound("translate", data_args=("sequence",))
def translate(sequence: str, frame: int = 1, to_stop: bool = False) -> Dict[str, Any]:
    """Translate DNA to protein, and show the working.

    G9-L7 has students do exactly this by hand with a codon wheel. The point of
    this function is not to replace that -- it is to let them CHECK it, codon by
    codon, so a mismatch tells them where they went wrong rather than just that
    they did.
    """
    seq = _clean(sequence)
    if frame < 0:
        seq = seq.translate(_COMPLEMENT)[::-1]
        offset = -frame - 1
    else:
        offset = frame - 1
    if offset < 0 or offset > 2:
        return {"refused": True, "reason": "Frame must be 1, 2, 3, -1, -2 or -3."}
    seq = seq[offset:]
    codons = [seq[i:i + 3] for i in range(0, len(seq) - len(seq) % 3, 3)]

    protein, detail = [], []
    for i, c in enumerate(codons):
        aa = CODON_TABLE.get(c, "X")
        if to_stop and aa == "*":
            detail.append({"position": i + 1, "codon": c, "amino_acid": aa,
                           "name": AA_NAMES.get(aa, "unknown")})
            break
        protein.append(aa)
        if len(detail) < 200:
            detail.append({"position": i + 1, "codon": c, "amino_acid": aa,
                           "name": AA_NAMES.get(aa, "unknown")})

    p = "".join(protein)
    return {
        "protein": p, "length": len(p), "frame": frame,
        "n_codons": len(codons),
        "codon_detail": detail,
        "trailing_bases": len(seq) % 3,
        "stops": p.count("*"),
        "starts_with_start_codon": bool(codons and codons[0] in START_CODONS),
        "note": ("Bacteria and their phage use NCBI translation table 11, which allows GTG and TTG "
                 "as start codons as well as ATG. All three are read as Methionine when they start "
                 "a gene, which is why a real phage gene can begin with a codon that means Valine "
                 "in the middle of one."),
        "plain_language": (
            f"Frame {frame}: {len(codons)} codons give a {len(p)}-residue protein with "
            f"{p.count('*')} stop{'s' if p.count('*') != 1 else ''}. "
            + (f"The first {min(6, len(codons))} codons are "
               + ", ".join(f"{d['codon']}->{d['amino_acid']}" for d in detail[:6]) + "."
               if detail else "")
            + (f" {len(seq) % 3} base(s) left over at the end -- a reading frame needs a multiple "
               f"of three, so those cannot be read." if len(seq) % 3 else "")),
        "p_value": None,
    }


@bound("find_orfs", data_args=("sequence",))
def find_orfs(sequence: str, min_length_aa: int = 30, all_frames: bool = True,
              require_start: bool = True) -> Dict[str, Any]:
    """Find open reading frames -- a gene caller a student can read in full.

    Pharokka wraps PHANOTATE and Prodigal-gv and is Tier 2 (spec section 07),
    and it should be: real gene calling is far more than this. But the spec's
    pedagogy is that *where two callers disagree is where a student learns
    annotation is a judgement, not a lookup*. To have that conversation a
    student first needs to see what the simplest possible caller does, and
    where it obviously fails.

    So this is the naive longest-ORF caller, and it says so: it over-calls in
    GC-rich genomes, misses short genes, and cannot see overlapping genes.
    """
    seq = _clean(sequence)
    if len(seq) < 60:
        return {"refused": True, "reason": "Sequence is too short to contain a gene."}

    frames = [1, 2, 3, -1, -2, -3] if all_frames else [1, 2, 3]
    orfs = []
    for fr in frames:
        s = seq.translate(_COMPLEMENT)[::-1] if fr < 0 else seq
        off = abs(fr) - 1
        body = s[off:]
        n_cod = len(body) // 3
        codons = [body[i * 3:i * 3 + 3] for i in range(n_cod)]

        start_idx = None
        for i, c in enumerate(codons):
            if start_idx is None and (c in START_CODONS if require_start else True):
                start_idx = i
            elif c in STOP_CODONS and start_idx is not None:
                aa_len = i - start_idx
                if aa_len >= min_length_aa:
                    nt_start = off + start_idx * 3
                    nt_end = off + (i + 1) * 3
                    orfs.append({
                        "frame": fr,
                        "start": (nt_start if fr > 0 else len(seq) - nt_end) + 1,
                        "end": (nt_end if fr > 0 else len(seq) - nt_start),
                        "strand": "+" if fr > 0 else "-",
                        "length_nt": nt_end - nt_start,
                        "length_aa": aa_len,
                        "start_codon": codons[start_idx],
                        "stop_codon": c,
                        "protein": "".join(CODON_TABLE.get(x, "X")
                                           for x in codons[start_idx:i]),
                    })
                start_idx = None

    orfs.sort(key=lambda o: -o["length_aa"])
    # Coding density is the fraction of the genome COVERED by at least one ORF,
    # not the sum of ORF lengths: six reading frames overlap heavily, and summing
    # them gives nonsense like 160% of a genome being coding.
    covered = bytearray(len(seq))
    for o in orfs:
        covered[max(0, o["start"] - 1):min(len(seq), o["end"])] = b"\x01" * (
            min(len(seq), o["end"]) - max(0, o["start"] - 1))
    density = sum(covered) / len(seq)

    return {
        "orfs": orfs[:300], "n_orfs": len(orfs),
        "genome_length": len(seq),
        "coding_density_estimate": float(density),
        "longest": orfs[0] if orfs else None,
        "parameters": {"min_length_aa": int(min_length_aa), "all_frames": all_frames,
                       "require_start": require_start},
        "limitations": [
            "This is the simplest possible gene caller: longest open stretch between a start and a "
            "stop. Real callers (PHANOTATE, Prodigal) also use codon statistics, ribosome binding "
            "sites and gene-length priors.",
            "It over-calls in GC-rich sequence, where stop codons are rarer by chance alone, so "
            "long open stretches appear without any gene being there.",
            "It cannot find overlapping genes, which phage use constantly to pack a genome.",
            "Change min_length_aa and the answer changes. Try it: that sensitivity is the point.",
        ],
        "teaches": (
            "Compare this caller's answer to a real one and they will disagree, most often about "
            "exactly where a gene STARTS -- the stop is unambiguous, the start is a judgement. "
            "Those disagreements are where annotation actually happens."),
        "plain_language": (
            f"Found {len(orfs)} open reading frames of at least {min_length_aa} amino acids in "
            f"{len(seq):,} bases, covering {density:.0%} of the genome. "
            + (f"The longest is {orfs[0]['length_aa']} amino acids on the {orfs[0]['strand']} "
               f"strand at position {orfs[0]['start']:,}. " if orfs else "")
            + (f"A coding density of {density:.0%} is low even for this crude caller -- real phage "
               f"genomes are usually over 85% coding, so genes are being missed."
               if density < 0.6 else
               f"Do not read {density:.0%} as confirmation that this genome is gene-dense: "
               f"searching six frames for any stretch without a stop codon reaches 80-90% "
               f"coverage on RANDOM sequence too. Coding density only means something once a "
               f"caller that uses codon statistics -- Prodigal or PHANOTATE -- has had a look.")),
        "p_value": None,
    }


@bound("codon_usage", data_args=("sequence",))
def codon_usage(sequence: str, frame: int = 1) -> Dict[str, Any]:
    """Codon usage bias -- which synonym this genome prefers.

    The bridge between G9-L7's codon wheel and the Grade 11 question of why a
    phage's codon usage looks like its host's.
    """
    seq = _clean(sequence)
    if frame < 0:
        seq = seq.translate(_COMPLEMENT)[::-1]
    off = abs(frame) - 1
    body = seq[off:]
    codons = [body[i:i + 3] for i in range(0, len(body) - len(body) % 3, 3)]
    if not codons:
        return {"refused": True, "reason": "No complete codons."}

    counts: Dict[str, int] = {}
    for c in codons:
        if c in CODON_TABLE:
            counts[c] = counts.get(c, 0) + 1

    by_aa: Dict[str, Dict[str, int]] = {}
    for c, n in counts.items():
        by_aa.setdefault(CODON_TABLE[c], {})[c] = n

    rscu: Dict[str, float] = {}
    for aa, group in by_aa.items():
        total = sum(group.values())
        k = len(group)
        for c, n in group.items():
            rscu[c] = float(n * k / total) if total else 0.0

    biased = sorted(rscu.items(), key=lambda kv: -kv[1])[:5]
    return {
        "counts": counts, "total_codons": len(codons),
        "by_amino_acid": {aa: {"codons": g, "total": sum(g.values()),
                               "n_synonyms": len(g)} for aa, g in by_aa.items()},
        "rscu": rscu,
        "most_preferred": [{"codon": c, "amino_acid": CODON_TABLE[c], "rscu": v} for c, v in biased],
        "note": ("RSCU (relative synonymous codon usage) is the observed count divided by what you "
                 "would expect if every synonym for that amino acid were used equally. 1.0 means no "
                 "preference; above 1 means this genome favours that spelling."),
        "plain_language": (
            f"{len(codons):,} codons. The strongest preference is {biased[0][0]} for "
            f"{AA_NAMES.get(CODON_TABLE[biased[0][0]], '?')} (RSCU {biased[0][1]:.2f}, where 1.00 "
            f"would mean no preference). A phage often shares its host's preferences, because it "
            f"uses the host's machinery to translate."
            if biased else "No codon preference could be computed."),
        "p_value": None,
    }
