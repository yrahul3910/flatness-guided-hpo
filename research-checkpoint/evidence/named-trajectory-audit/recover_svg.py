"""Recover approximate retained vertices from the committed HPOBench figure.

This does not reconstruct raw training records or hidden proxy components. The known SVG is calibrated against its tick
coordinates, preserving all vertices for the three reviewer-named datasets and checking the integer epoch grid.
"""

# This evidence script runs directly and is not an importable package.

# ruff: noqa: INP001

import csv
import hashlib
import json
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper/img/hpobench.svg"
EXPECTED_HASH = "1f4fc37b2e55bdb99053cff17ffaf61fd43b6be83679b3e657cb0904e212f3fb"
NAMESPACE = {"s": "http://www.w3.org/2000/svg"}
NAMED_AXES = {"axes_1": "vehicle", "axes_2": "btsc", "axes_6": "segment"}
EPOCH_TOLERANCE = 1e-5


def ticks(axis: ET.Element, kind: str) -> list[tuple[float, str | None]]:
    result: list[tuple[float, str | None]] = []

    for tick in axis.findall("s:g/s:g", NAMESPACE):
        if not tick.get("id", "").startswith(f"{kind}tick_"):
            continue
        use = tick.find(".//s:use", NAMESPACE)

        if use is None:
            msg = "Tick coordinate missing"
            raise ValueError(msg)

        labels = [node.text for node in tick.iter() if node.tag is ET.Comment]
        result.append((float(use.attrib[kind]), labels[0] if labels else None))

    return result


def parse_audited_svg() -> ET.Element:
    svg = SOURCE.read_bytes()

    if hashlib.sha256(svg).hexdigest() != EXPECTED_HASH:
        msg = "The audited SVG changed; review the extraction assumptions"
        raise ValueError(msg)

    # Only the fixed, hash-verified repository SVG is parsed; no external XML is accepted.
    return ET.fromstring(svg, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))  # noqa: S314


def recover() -> None:
    root = parse_audited_svg()
    shared_axis = root.find('.//s:g[@id="axes_7"]', NAMESPACE)

    if shared_axis is None:
        msg = "Shared x-axis labels missing"
        raise ValueError(msg)

    x_labels = [float(label) for _, label in ticks(shared_axis, "x") if label is not None]
    datasets = {}
    rows: list[tuple[str, str, int, float, float, float]] = []

    for axis_id, dataset in NAMED_AXES.items():
        axis = root.find(f'.//s:g[@id="{axis_id}"]', NAMESPACE)

        if axis is None:
            msg = f"Missing {axis_id}"
            raise ValueError(msg)

        x_ticks = ticks(axis, "x")
        y_ticks = [(position, float(label)) for position, label in ticks(axis, "y") if label is not None]
        x_fit = statistics.linear_regression([position for position, _ in x_ticks], x_labels)
        y_fit = statistics.linear_regression([position for position, _ in y_ticks], [label for _, label in y_ticks])
        series = {}

        for group in axis:
            if not group.get("id", "").startswith("line2d_") or (path := group.find("s:path", NAMESPACE)) is None:
                continue

            style = path.get("style", "")
            metric = "val_loss" if "#ff0000" in style else "sharpness"

            if "#ff0000" not in style and "#008000" not in style:
                msg = f"Unrecognized series style: {style}"
                raise ValueError(msg)

            commands = re.findall(r"[A-Za-z]", path.attrib["d"])

            if commands != ["M"] + ["L"] * (len(commands) - 1):
                msg = "Unexpected SVG path; do not treat as per-epoch vertices"
                raise ValueError(msg)

            coordinates = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path.attrib["d"])]
            vertices: list[dict[str, float]] = []

            for x, y in zip(coordinates[::2], coordinates[1::2], strict=True):
                epoch_approx = x_fit.slope * x + x_fit.intercept
                epoch = round(epoch_approx)

                if abs(epoch_approx - epoch) > EPOCH_TOLERANCE:
                    msg = "Non-integer epoch vertex; inspect simplification or axis mapping"
                    raise ValueError(msg)

                value = y_fit.slope * y + y_fit.intercept
                vertices.append({"epoch": epoch, "value_approx": value, "svg_x": x, "svg_y": y})
                rows.append((dataset, metric, epoch, value, x, y))

            epochs = [vertex["epoch"] for vertex in vertices]
            series[metric] = {
                "path_id": group.attrib["id"],
                "contiguous_from_epoch_one": epochs == list(range(1, len(vertices) + 1)),
                "retained_vertices": vertices,
            }

        datasets[dataset] = {
            "axis_id": axis_id,
            "x_ticks": list(zip(x_ticks, x_labels, strict=True)),
            "y_ticks": y_ticks,
            "x_value_per_svg_unit": x_fit.slope,
            "y_value_per_svg_unit": y_fit.slope,
            "series": series,
        }

    report = {
        "evidence_kind": "approximate_retained_svg_vertices",
        "source_sha256": EXPECTED_HASH,
        "datasets": datasets,
    }
    _ = (OUTPUT / "approximate-svg-vertices.json").write_text(json.dumps(report, indent=2) + "\n")

    with (OUTPUT / "approximate-svg-vertices.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset", "metric", "epoch", "value_approx", "svg_x", "svg_y"))
        writer.writerows(rows)


if __name__ == "__main__":
    recover()
