"""ASCII banner displayed when PEMD starts."""

from __future__ import annotations

from shutil import get_terminal_size
from typing import List, Optional, Tuple, Iterable
import os
import re
import sys

# -------- Utilities ---------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    """Remove ANSI color codes."""
    return _ANSI_RE.sub("", s)


def _visible_len(s: str) -> int:
    """Visible length of a string (without ANSI)."""
    return len(_strip_ansi(s))


def _center(line: str, width: int) -> str:
    """Center a string by left padding; width computed on visible length."""
    vis = _visible_len(line)
    if vis >= width:
        return line
    pad = (width - vis) // 2
    return " " * pad + line


def _supports_color(stream) -> bool:
    """Heuristic color support detection (TTY + env)."""
    # honor explicit env knobs first
    if os.environ.get("PEMD_BANNER_COLOR", "1") == "0":
        return False
    if os.environ.get("NO_COLOR"):  # https://no-color.org/
        return False
    try:
        is_tty = hasattr(stream, "isatty") and stream.isatty()
    except Exception:
        is_tty = False

    if not is_tty:
        return False
    # Windows: many modern terminals support ANSI; keep simple & permissive
    if os.name == "nt":
        return bool(
            os.environ.get("ANSICON")
            or os.environ.get("WT_SESSION")
            or os.environ.get("ConEmuANSI") == "ON"
            or os.environ.get("TERM_PROGRAM") == "vscode"
        )
    # Unix-ish terminals
    term = os.environ.get("TERM", "")
    return term not in ("", "dumb")


class _Ansi:
    """Lightweight ANSI palette with auto-disable."""
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _c(self, code: str) -> str:
        return code if self.enabled else ""

    @property
    def reset(self) -> str: return self._c("\033[0m")
    @property
    def bold(self) -> str: return self._c("\033[1m")
    @property
    def dim(self) -> str: return self._c("\033[2m")
    @property
    def cyan(self) -> str: return self._c("\033[36m")
    @property
    def yellow(self) -> str: return self._c("\033[33m")
    @property
    def green(self) -> str: return self._c("\033[32m")
    @property
    def red(self) -> str: return self._c("\033[31m")


def _term_width(min_width: int) -> int:
    """Safe terminal width with lower bound."""
    try:
        return max(min_width, get_terminal_size((min_width, 20)).columns)
    except Exception:
        return max(min_width, 80)

# -------- Banner ------------------------------------------------------------

def print_pemd_info(
    version: Optional[str] = None,
    project_name: str = "Polymer Electrolyte Modeling & Discovery (PEMD)",
    org_1: str = "Institute of Materials Research (iMR), Tsinghua SIGS, Shenzhen, China",
    # org_2: str = "School of Materials Science and Engineering, Tsinghua University, Beijing, China",
    cite_lines: Tuple[str, ...] = (
        "S. Tan, T. Hou*, et al., PEMD: An open-source framework for",
        "high-throughput simulation and analysis of polymer electrolytes,",
        "journal name, volume, page, 2025",
    ),
    show_color: bool = True,
    *,
    force: bool = False,
    stream = sys.stdout,
    return_str: bool = False,   # New: return the banner string instead of printing
) -> bool | str:
    """Print the PEMD start-up banner.

    Returns:
        bool: True if printed to stream; False if suppressed. When
        return_str=True, returns the banner string (and does not print).
    """
    if os.environ.get("PEMD_BANNER", "1") == "0" and not force and not return_str:
        return False

    if version is None:
        try:
            from PEMD import __version__ as _v
            version = _v
        except Exception:
            version = "1.0.0"

    ansi = _Ansi(enabled=(show_color and _supports_color(stream)))

    top_block: List[str] = [
        "",
        "                        ---------  PPPPPP      EEEEEEEE   MMMMMMMMMM   DDDDDD    ---------                        ",
        "                -----------------  PP    PP    EE         MM  MM  MM   DD    DD  -----------------                ",
        "        -------------------------  PP    PP    EE         MM  MM  MM   DD    DD  -------------------------        ",
        " --------------------------------  PPPPPP      EEEEEE     MM  MM  MM   DD    DD  -------------------------------- ",
        "        -------------------------  PP          EE         MM  MM  MM   DD    DD  -------------------------        ",
        "                -----------------  PP          EE         MM  MM  MM   DD    DD  -----------------                ",
        "                        ---------  PP          EEEEEEEE   MM  MM  MM   DDDDDD    ---------                        ",
        " --------------------------------------------------------------------------------------------------- ",
        "          ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **          ",
    ]

    base_width = max(_visible_len(ln) for ln in top_block)
    width = _term_width(base_width)

    # Compose lines (use visible width for centering)
    lines: List[str] = []
    for ln in top_block:
        lines.append(_center(ln, width))

    title = (
        f"{ansi.bold}{project_name}{ansi.reset}  "
        f"version = {ansi.yellow}{version}{ansi.reset}"
    )
    lines.append(_center(title, width))
    lines.append(_center(f"{ansi.dim}Developed at Hou Group{ansi.reset}", width))
    lines.append(_center(f"{ansi.cyan}{org_1}{ansi.reset}", width))
    # lines.append(_center(f"{ansi.cyan}{org_2}{ansi.reset}", width))
    lines.append(_center(" ", width))

    lines.append(_center(f"{ansi.bold}Cite this work as:{ansi.reset}", width))
    for cl in cite_lines:
        lines.append(_center(cl, width))

    lines.append(_center("  ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **  ", width))

    banner = "\n".join(lines)

    if return_str:
        return banner
    try:
        print(banner, file=stream)
    except Exception:
        # Fallback: print without ANSI colors
        print(_strip_ansi(banner), file=sys.stdout)

    return True

# -------- Small status helpers ---------------------------------------------


def print_input(inp: str, *, stream = sys.stdout) -> None:
    """Standardized 'start' message."""
    ansi = _Ansi(enabled=_supports_color(stream))
    print(f"{ansi.bold}[PEMD]{ansi.reset} {inp} started...\n", file=stream)


def print_output(out: str, *, status: str = "successfully", stream = sys.stdout) -> None:
    """Standardized 'done' message."""
    print("", file=stream)  # Leading blank line
    ansi = _Ansi(enabled=_supports_color(stream))
    tick = f"{ansi.green}✓{ansi.reset}" if ansi.enabled else "OK"
    print(f"{tick} {out} {status}!!!\n", file=stream)

# -------- Table printer -----------------------------------------------------


def _truncate(s: str, maxlen: int) -> str:
    if s is None:
        return "—"
    s = str(s)
    return (s[: max(0, maxlen - 1)] + "…") if len(s) > maxlen else s


def print_poly_info(
    name: str,
    *,
    smiles_A: Optional[str] = None,
    smiles_B: Optional[str] = None,
    left_cap: Optional[str] = None,
    right_cap: Optional[str] = None,
    mode: Optional[str] = None,
    length: Optional[int] = None,
    block_sizes: Optional[Iterable[int]] = None,
    style: str = "unicode",        # 'unicode' or 'ascii'
    title: str = "Polymer Build Parameters",
    return_markdown: bool = False, # Return Markdown instead of printing when True
    max_value_width: int = 120,    # Prevent wide SMILES from overflowing the table
    stream = sys.stdout,
):
    """Pretty-print build parameters as a table (or return Markdown)."""

    def fmt(v):  # Normalize empty values
        return "—" if v is None or v == "" else str(v)

    rows: List[Tuple[str, str]] = [("Polymer Name", fmt(name))]
    if smiles_A is not None and (smiles_B is None or smiles_A == smiles_B):
        rows.append(("SMILES", fmt(smiles_A)))
    else:
        rows.append(("Monomer A", fmt(smiles_A)))
        rows.append(("Monomer B", fmt(smiles_B)))

    if mode == 'random':
        rows.append(("Mode", "Random Copolymer"))
    elif mode == 'block':
        rows.append(("Mode", "Block Copolymer"))
    elif mode == 'alternating':
        rows.append(("Mode", "Alternating copolymer"))
    else:
        rows.append(("Mode", "Homopolymer"))

    if block_sizes is not None:
        rows.append(("Block Sizes", fmt(list(block_sizes))))
    if length is not None:
        rows.append(("Length", fmt(length)))

    if left_cap is not None:
        rows.append(("Left Cap", fmt(left_cap)))
    if right_cap is not None:
        rows.append(("Right Cap", fmt(right_cap)))

    # Truncate long values (Markdown output still uses full values)
    clipped_rows = [(k, _truncate(v, max_value_width)) for k, v in rows]

    if return_markdown:
        md: List[str] = [f"**{title}**", "", "| Parameter | Value |", "|---|---|"]
        md += [f"| {k} | {fmt(v)} |" for k, v in rows]
        return "\n".join(md)

    # Compute column widths
    header = ("Parameter", "Value")
    w1 = max(len(header[0]), *(len(k) for k, _ in clipped_rows))
    w2 = max(len(header[1]), *(len(v) for _, v in clipped_rows))

    # Choose glyphs based on the style
    if style == "ascii":
        H, V = "-", "|"
        TL, TR, BL, BR = "+", "+", "+", "+"
        T, B, L, R, C = "+", "+", "+", "+", "+"
    else:  # unicode
        H, V = "─", "│"
        TL, TR, BL, BR = "┌", "┐", "└", "┘"
        T, B, L, R, C = "┬", "┴", "├", "┤", "┼"

    def hline(left: str, mid: str, right: str) -> str:
        return f"{left}{H*(w1+2)}{mid}{H*(w2+2)}{right}"

    # Emit the formatted table
    print(title, file=stream)
    print(hline(TL, T, TR), file=stream)
    print(f"{V} {header[0]:<{w1}} {V} {header[1]:<{w2}} {V}", file=stream)
    print(hline(L, C, R), file=stream)
    for k, v in clipped_rows:
        print(f"{V} {k:<{w1}} {V} {v:<{w2}} {V}", file=stream)
    print(hline(BL, B, BR), file=stream)
    print("", file=stream)  # Trailing blank line


def print_box_composition(
    molecule_list,
    *,
    title: str = "Box Composition",
    style: str = "unicode",         # 'unicode' or 'ascii'
    return_markdown: bool = False,  # Return Markdown instead of printing when True
    aliases: Optional[dict] = None, # Optional mapping from keys to user-friendly names
    stream = sys.stdout,
):
    """
    Print box composition (showing only Species and Count; omits Fraction and
    Total rows).

    - ``molecule_list``: ``dict[str, int]`` or an iterable of ``(name, count)``
    """
    # Normalize the input structure
    if isinstance(molecule_list, dict):
        items = list(molecule_list.items())   # Dicts preserve insertion order in Py3.7+
    else:
        items = list(molecule_list)           # Assume an iterable of (name, count)

    # Assemble rows containing just the name and count
    rows = []
    for name, count in items:
        c = int(count)
        disp = aliases.get(name, name) if aliases else name
        rows.append((str(disp), str(c)))

    if return_markdown:
        md = [f"**{title}**", "", "| Species | Count |", "|---|---:|"]
        md += [f"| {k} | {v} |" for (k, v) in rows]
        return "\n".join(md)

    # Compute column widths
    header = ("Species", "Count")
    w1 = max(len(header[0]), *(len(k) for k, _ in rows)) if rows else len(header[0])
    w2 = max(len(header[1]), *(len(v) for _, v in rows)) if rows else len(header[1])

    # Choose glyphs based on the style
    if style == "ascii":
        H, V = "-", "|"
        TL, TR, BL, BR = "+", "+", "+", "+"
        T, B, L, R, C = "+", "+", "+", "+", "+"
    else:  # unicode
        H, V = "─", "│"
        TL, TR, BL, BR = "┌", "┐", "└", "┘"
        T, B, L, R, C = "┬", "┴", "├", "┤", "┼"

    def hline(left: str, mid: str, right: str) -> str:
        return f"{left}{H*(w1+2)}{mid}{H*(w2+2)}{right}"

    # Print the table
    print(title, file=stream)
    print(hline(TL, T, TR), file=stream)
    print(f"{V} {header[0]:<{w1}} {V} {header[1]:>{w2}} {V}", file=stream)
    print(hline(L, C, R), file=stream)
    for k, v in rows:
        print(f"{V} {k:<{w1}} {V} {v:>{w2}} {V}", file=stream)
    print(hline(BL, B, BR), file=stream)
    print("", file=stream)


