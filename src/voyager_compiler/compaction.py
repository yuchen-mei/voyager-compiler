"""Operators of order-preserving row compaction and its small-integer counts.

EVS keeps a data-dependent subset of a prompt's rows. These operators find
the dropped rows, count them as exact small integers and bit fields, and move
the kept rows to the front in stages whose reads use fixed offsets only.
"""

from typing import List

import torch
from torch.library import Library, impl

compaction_lib = Library("quantized_ops", "FRAGMENT")


def _bits_value(bits: torch.Tensor) -> torch.Tensor:
    """Return the int64 value of little-endian 0/1 bit fields [..., B]."""
    weights = 2 ** torch.arange(bits.shape[-1], device=bits.device)
    return (bits.to(torch.int64) * weights).sum(dim=-1)


def _value_bits(value: torch.Tensor, bits: int, dtype) -> torch.Tensor:
    """Return the low ``bits`` bits of int64 ``value`` as 0/1 fields [..., bits]."""
    shifts = torch.arange(bits, device=value.device)
    return ((value[..., None] >> shifts) & 1).to(dtype)


compaction_lib.define(
    "records_after(Tensor records, Tensor threshold, int[] order) -> Tensor"
)


@impl(compaction_lib, "records_after", "CompositeExplicitAutograd")
def records_after(records: torch.Tensor, threshold: torch.Tensor, order: List[int]) -> torch.Tensor:
    """Flag the records that a sort by ``order`` places after ``threshold``.

    ``records`` is [n, F] and ``threshold`` [1, F]; ``order`` compares fields
    as in bitonic_merge_step (-1 larger first, 1 smaller first, 0 not
    compared). Returns [n] in the records' dtype: 1 where record i must come
    after the threshold record, 0 where it comes before or is equal. Only
    comparisons are used, so the result is exact.
    """
    after = torch.zeros_like(records[:, 0], dtype=torch.bool)
    decided = torch.zeros_like(after)
    for field, direction in enumerate(order):
        if direction == 0:
            continue
        value, limit = records[:, field], threshold[0, field]
        later, earlier = (value < limit, value > limit) if direction < 0 else (value > limit, value < limit)
        after = after | (~decided & later)
        decided = decided | later | earlier
    return after.to(records.dtype)


compaction_lib.define("flag_record(Tensor input, int bits) -> Tensor")


@impl(compaction_lib, "flag_record", "CompositeExplicitAutograd")
def flag_record(input: torch.Tensor, bits: int) -> torch.Tensor:
    """Write each 0/1 flag of ``input`` as a bit-field record of ``bits`` fields.

    Field 0 is the flag and the other fields are 0, so the record is the
    integer 0 or 1 in the form bits_add reads. Returns [..., bits] in the
    input's dtype.
    """
    record = torch.zeros(*input.shape, bits, dtype=input.dtype, device=input.device)
    record[..., 0] = input
    return record


compaction_lib.define("bits_add(Tensor input, Tensor other) -> Tensor")


@impl(compaction_lib, "bits_add", "CompositeExplicitAutograd")
def bits_add(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Add two little-endian bit-field integers [..., B] modulo 2**B.

    Both operands hold 0/1 fields, field b being bit b. Returns the
    sum's B bit fields in the input's dtype.
    """
    bits = input.shape[-1]
    total = (_bits_value(input) + _bits_value(other)) % (1 << bits)
    return _value_bits(total, bits, input.dtype)


compaction_lib.define("bits_mask(Tensor input, Tensor drop) -> Tensor")


@impl(compaction_lib, "bits_mask", "CompositeExplicitAutograd")
def bits_mask(input: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """Clear the bit-field records of dropped rows.

    ``input`` is [..., B] and ``drop`` [...], 1 for a dropped row and 0 for a
    kept one. Returns ``input`` with every field of a dropped row set to 0.
    """
    return input * (1 - drop)[..., None]


compaction_lib.define(
    "compaction_control_step(Tensor now, Tensor ahead, int bit) -> Tensor"
)


@impl(compaction_lib, "compaction_control_step", "CompositeExplicitAutograd")
def compaction_control_step(now: torch.Tensor, ahead: torch.Tensor, bit: int) -> torch.Tensor:
    """Move the control records of one compaction stage.

    ``now`` [L, B] holds the records of rows 0..L-1 and ``ahead`` [L, B] those
    of rows 2**bit .. 2**bit + L - 1. A record whose field ``bit`` is 1 moves
    2**bit rows toward the start. Row i takes the record from ``ahead`` when
    that one moves, else keeps its own record, cleared to zeros when that
    one moves away. Returns the [L, B] records after the stage.
    """
    moves_in = ahead[:, bit : bit + 1] == 1
    moves_out = now[:, bit : bit + 1] == 1
    return torch.where(moves_in, ahead, torch.where(moves_out, torch.zeros_like(now), now))


compaction_lib.define(
    "compaction_shift_step(Tensor now, Tensor ahead, Tensor move) -> Tensor"
)


@impl(compaction_lib, "compaction_shift_step", "CompositeExplicitAutograd")
def compaction_shift_step(now: torch.Tensor, ahead: torch.Tensor, move: torch.Tensor) -> torch.Tensor:
    """Move the data rows of one compaction stage.

    ``now`` [L, C] holds rows 0..L-1, ``ahead`` [L, C] the rows a fixed
    distance later, and ``move`` [L, 1] is 1 where row i takes its ``ahead``
    row and 0 where it keeps its own. Returns the [L, C] rows after the stage;
    values are only selected, never changed.
    """
    return torch.where(move == 1, ahead, now)


compaction_lib.define("block_interleave(Tensor input, Tensor other) -> Tensor")


@impl(compaction_lib, "block_interleave", "CompositeExplicitAutograd")
def block_interleave(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Join the two parts of each block.

    ``input`` [F, A] and ``other`` [F, B] hold the first A and the next B
    values of each of F blocks. Returns [F, A + B]; values are only copied.
    """
    return torch.cat([input, other], dim=1)
