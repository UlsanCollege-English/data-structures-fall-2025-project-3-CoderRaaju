from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

MIN_LAYOVER_MINUTES = 60
Cabin = Literal["economy", "business", "first"]


@dataclass(frozen=True)
class Flight:
    origin: str
    dest: str
    flight_number: str
    depart: int
    arrive: int
    economy: int
    business: int
    first: int

    def price_for(self, cabin: Cabin) -> int:
        return {
            "economy": self.economy,
            "business": self.business,
            "first": self.first,
        }[cabin]


@dataclass
class Itinerary:
    flights: List[Flight]

    def is_empty(self) -> bool:
        return not self.flights

    @property
    def origin(self) -> Optional[str]:
        return self.flights[0].origin if self.flights else None

    @property
    def dest(self) -> Optional[str]:
        return self.flights[-1].dest if self.flights else None

    @property
    def depart_time(self) -> Optional[int]:
        return self.flights[0].depart if self.flights else None

    @property
    def arrive_time(self) -> Optional[int]:
        return self.flights[-1].arrive if self.flights else None

    def total_price(self, cabin: Cabin) -> int:
        return sum(f.price_for(cabin) for f in self.flights)

    def num_stops(self) -> int:
        return max(0, len(self.flights) - 1)


Graph = Dict[str, List[Flight]]


# ---------------- Time helpers ---------------- #

def parse_time(hhmm: str) -> int:
    if ":" not in hhmm:
        raise ValueError("Invalid time format")
    h, m = hhmm.split(":")
    try:
        h = int(h)
        m = int(m)
    except ValueError:
        raise ValueError("Invalid time format")
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError("Invalid time range")
    return h * 60 + m


def format_time(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


# ---------------- Loading flights ---------------- #

def parse_flight_line_txt(line: str) -> Optional[Flight]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split()
    if len(parts) != 8:
        raise ValueError("Malformed flight line")

    o, d, num, dep, arr, e, b, f = parts
    dep_m = parse_time(dep)
    arr_m = parse_time(arr)
    if arr_m <= dep_m:
        raise ValueError("Arrival must be after departure")

    return Flight(
        o, d, num, dep_m, arr_m,
        int(e), int(b), int(f)
    )


def load_flights_txt(path: str) -> List[Flight]:
    flights = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            try:
                fl = parse_flight_line_txt(line)
                if fl:
                    flights.append(fl)
            except ValueError as e:
                raise ValueError(f"{path}:{i}: {e}")
    return flights


def load_flights_csv(path: str) -> List[Flight]:
    flights = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {
            "origin", "dest", "flight_number",
            "depart", "arrive", "economy", "business", "first"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required CSV columns")

        for row in reader:
            dep = parse_time(row["depart"])
            arr = parse_time(row["arrive"])
            if arr <= dep:
                raise ValueError("Arrival must be after departure")
            flights.append(
                Flight(
                    row["origin"],
                    row["dest"],
                    row["flight_number"],
                    dep,
                    arr,
                    int(row["economy"]),
                    int(row["business"]),
                    int(row["first"]),
                )
            )
    return flights


def load_flights(path: str) -> List[Flight]:
    if Path(path).suffix.lower() == ".csv":
        return load_flights_csv(path)
    return load_flights_txt(path)


# ---------------- Graph ---------------- #

def build_graph(flights: Iterable[Flight]) -> Graph:
    graph: Graph = {}
    for f in flights:
        graph.setdefault(f.origin, []).append(f)
    return graph


# ---------------- Searches ---------------- #

def find_earliest_itinerary(
    graph: Graph,
    start: str,
    dest: str,
    earliest_departure: int,
) -> Optional[Itinerary]:

    pq = [(earliest_departure, start)]
    best_time = {start: earliest_departure}
    prev: Dict[str, Flight] = {}

    while pq:
        time, airport = heapq.heappop(pq)
        if airport == dest:
            break

        for fl in graph.get(airport, []):
            min_dep = (
                earliest_departure if airport == start
                else time + MIN_LAYOVER_MINUTES
            )
            if fl.depart < min_dep:
                continue

            if fl.dest not in best_time or fl.arrive < best_time[fl.dest]:
                best_time[fl.dest] = fl.arrive
                prev[fl.dest] = fl
                heapq.heappush(pq, (fl.arrive, fl.dest))

    if dest not in prev and start != dest:
        return None

    flights = []
    cur = dest
    while cur != start:
        fl = prev.get(cur)
        if not fl:
            return None
        flights.append(fl)
        cur = fl.origin

    flights.reverse()
    return Itinerary(flights)


def find_cheapest_itinerary(
    graph: Graph,
    start: str,
    dest: str,
    earliest_departure: int,
    cabin: Cabin,
) -> Optional[Itinerary]:

    pq = [(0, earliest_departure, start)]
    best_cost = {start: 0}
    best_time = {start: earliest_departure}
    prev: Dict[str, Flight] = {}

    while pq:
        cost, time, airport = heapq.heappop(pq)
        if airport == dest:
            break

        for fl in graph.get(airport, []):
            min_dep = (
                earliest_departure if airport == start
                else time + MIN_LAYOVER_MINUTES
            )
            if fl.depart < min_dep:
                continue

            new_cost = cost + fl.price_for(cabin)
            if (
                fl.dest not in best_cost
                or new_cost < best_cost[fl.dest]
            ):
                best_cost[fl.dest] = new_cost
                best_time[fl.dest] = fl.arrive
                prev[fl.dest] = fl
                heapq.heappush(
                    pq, (new_cost, fl.arrive, fl.dest)
                )

    if dest not in prev:
        return None

    flights = []
    cur = dest
    while cur != start:
        fl = prev[cur]
        flights.append(fl)
        cur = fl.origin

    flights.reverse()
    return Itinerary(flights)


# ---------------- Formatting ---------------- #

@dataclass
class ComparisonRow:
    mode: str
    cabin: Optional[Cabin]
    itinerary: Optional[Itinerary]
    note: str = ""


def format_comparison_table(
    origin: str,
    dest: str,
    earliest_departure: int,
    rows: List[ComparisonRow],
) -> str:

    lines = []
    lines.append(f"Comparison for {origin} → {dest}")
    lines.append(
        "Mode | Cabin | Dep | Arr | Stops | Total Price | Notes"
    )
    lines.append("-" * 70)

    for r in rows:
        if r.itinerary:
            dep = format_time(r.itinerary.depart_time)
            arr = format_time(r.itinerary.arrive_time)
            stops = r.itinerary.num_stops()
            price = (
                r.itinerary.total_price(r.cabin)
                if r.cabin else ""
            )
        else:
            dep = arr = stops = price = "N/A"

        lines.append(
            f"{r.mode} | {r.cabin or ''} | {dep} | {arr} | "
            f"{stops} | {price} | {r.note}"
        )

    return "\n".join(lines)


# ---------------- CLI ---------------- #

def run_compare(args: argparse.Namespace) -> None:
    earliest = parse_time(args.departure_time)
    flights = load_flights(args.flight_file)
    graph = build_graph(flights)

    rows = [
        ComparisonRow(
            "Earliest arrival",
            None,
            find_earliest_itinerary(
                graph, args.origin, args.dest, earliest
            ),
        )
    ]

    for cabin in ("economy", "business", "first"):
        itin = find_cheapest_itinerary(
            graph, args.origin, args.dest, earliest, cabin
        )
        rows.append(
            ComparisonRow(
                f"Cheapest ({cabin.capitalize()})",
                cabin,
                itin,
                "" if itin else "no valid itinerary",
            )
        )

    print(
        format_comparison_table(
            args.origin, args.dest, earliest, rows
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare")
    c.add_argument("flight_file")
    c.add_argument("origin")
    c.add_argument("dest")
    c.add_argument("departure_time")
    c.set_defaults(func=run_compare)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
