"""
Lab 2 of 8 - Informed Search: Greedy Best-First Search and A*
Objective: Add a Haversine heuristic to the Lab 1 planner, check that it is
admissible, and measure how much search effort GBFS and A* save over UCS.

Plain Python script (no Jupyter needed).
Save this file in:  all_experiments/
Run it with the Run button (top right in VS Code) or:  python lab02_informed_search.py
"""

from heapq import heappop, heappush
from math import inf
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")                       # save figures without opening a window
import matplotlib.pyplot as plt

from utils import astar, best_first, gbfs, haversine_km, load_openflights, route_is_valid, ucs

HERE = Path(__file__).resolve().parent      # .../all_experiments
ROOT = HERE.parent                          # the course folder
DATA = ROOT / "all_datasets"
RESULTS = ROOT / "results" / "lab02"
RESULTS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


airports, routes, coords, graph, stats = load_openflights(
    DATA / "lab01-03_airports.csv",
    DATA / "lab01-03_routes.csv",
)
queries = pd.read_csv(HERE / "utils" / "search_queries.csv").head(3)   # the three standard queries


def heuristic_to(goal, weight=1.0):
    """h(n): great-circle distance from airport n to the goal, in km (optionally inflated)."""
    goal_lat, goal_lon = coords[goal]["latitude"], coords[goal]["longitude"]

    def h(airport):
        return weight * haversine_km(coords[airport]["latitude"], coords[airport]["longitude"], goal_lat, goal_lon)

    return h


def row_for(query, name, result):
    return {"query_id": query.query_id, "start": query.start, "goal": query.goal,
            "algorithm": name, **result.as_dict(), "path": " -> ".join(result.path)}


# ---------------------------------------------------------------------------
# Question 1 - Establish the baseline (UCS)
# ---------------------------------------------------------------------------
heading("Question 1 - Establish the baseline with UCS")

baseline = {q.query_id: ucs(graph, q.start, q.goal) for q in queries.itertuples()}
ucs_table = pd.DataFrame([row_for(q, "UCS", baseline[q.query_id]) for q in queries.itertuples()])
print(ucs_table[["query_id", "start", "goal", "hops", "path_cost_km", "expanded", "path"]].round(1).to_string(index=False))
print("\nUCS is exact for non-negative costs, so these costs are the reference for every other method.")


# ---------------------------------------------------------------------------
# Question 2 - Implement the heuristic and run GBFS and A*
# ---------------------------------------------------------------------------
heading("Question 2 - Haversine heuristic, GBFS and A*")

for q in queries.itertuples():
    h = heuristic_to(q.goal)
    print(f"{q.query_id} {q.start}->{q.goal}: h(start) = {h(q.start):,.0f} km, "
          f"h(goal) = {h(q.goal):.0f} km, UCS cost = {baseline[q.query_id].path_cost_km:,.0f} km")

rows = []
for q in queries.itertuples():
    h = heuristic_to(q.goal)
    for name, result in [("UCS", baseline[q.query_id]),
                         ("GBFS", gbfs(graph, q.start, q.goal, h)),
                         ("A*", astar(graph, q.start, q.goal, h))]:
        assert result.found and route_is_valid(result.path, graph)
        rows.append(row_for(q, name, result))

runs = pd.DataFrame(rows)
print()
print(runs[["query_id", "algorithm", "hops", "path_cost_km", "expanded", "path"]].round(1).to_string(index=False))


# ---------------------------------------------------------------------------
# Question 3 - Check admissibility empirically
# ---------------------------------------------------------------------------
heading("Question 3 - Check admissibility empirically")


def exact_cost_to(goal):
    """Dijkstra on the reversed graph: h*(n), the true cheapest cost from every airport to the goal."""
    reverse = {}
    for a, neighbours in graph.items():
        for b, cost in neighbours:
            reverse.setdefault(b, []).append((a, cost))
    dist = {goal: 0.0}
    heap = [(0.0, goal)]
    while heap:
        d, node = heappop(heap)
        if d > dist.get(node, inf):
            continue
        for prev, cost in reverse.get(node, ()):
            nd = d + cost
            if nd < dist.get(prev, inf):
                dist[prev] = nd
                heappush(heap, (nd, prev))
    return dist


rng = np.random.default_rng(42)
checks = []
for q in queries.itertuples():
    h_star = exact_cost_to(q.goal)
    h = heuristic_to(q.goal)
    candidates = sorted(a for a in h_star if a != q.goal)
    sample = [q.start] + list(rng.choice(candidates, size=9, replace=False))     # ten airports per query
    for airport in sample:
        checks.append({"query_id": q.query_id, "airport": airport, "goal": q.goal,
                       "h_km": h(airport), "h_star_km": h_star[airport],
                       "slack_km": h_star[airport] - h(airport)})

admissibility = pd.DataFrame(checks)
admissibility.to_csv(RESULTS / "admissibility.csv", index=False)
print(admissibility.round(1).to_string(index=False))
violations = int((admissibility.slack_km < -1e-6).sum())
print(f"\nViolations (h > h*): {violations} of {len(admissibility)} checks")

# Consistency: h(a) <= cost(a, b) + h(b) on every edge, checked for every goal.
inconsistent = 0
edges = 0
for q in queries.itertuples():
    h = heuristic_to(q.goal)
    for a, neighbours in graph.items():
        ha = h(a)
        for b, cost in neighbours:
            edges += 1
            inconsistent += ha > cost + h(b) + 1e-6
print(f"Consistency violations: {inconsistent} of {edges:,} edge checks")
print("Haversine never overestimates: a flight path can never be shorter than the great circle,")
print("so h is admissible and (by the triangle inequality) consistent, and A* stays optimal.")


# ---------------------------------------------------------------------------
# Question 4 - Compare the methods
# ---------------------------------------------------------------------------
heading("Question 4 - Compare UCS, GBFS and A*")

metrics = runs.copy()
ucs_cost = metrics[metrics.algorithm == "UCS"].set_index("query_id").path_cost_km
metrics["optimality_gap_pct"] = [100 * (c / ucs_cost[qid] - 1) for qid, c in zip(metrics.query_id, metrics.path_cost_km)]
ucs_expanded = metrics[metrics.algorithm == "UCS"].set_index("query_id").expanded
metrics["expansions_saved_vs_ucs_pct"] = [100 * (1 - e / ucs_expanded[qid]) for qid, e in zip(metrics.query_id, metrics.expanded)]
metrics = metrics[["query_id", "start", "goal", "algorithm", "hops", "path_cost_km", "optimality_gap_pct",
                   "expanded", "max_frontier", "runtime_ms", "expansions_saved_vs_ucs_pct", "path"]]
metrics.to_csv(RESULTS / "metrics.csv", index=False)
print(metrics.drop(columns=["start", "goal", "path"]).round(2).to_string(index=False))

summary = metrics.groupby("algorithm", sort=False)[["path_cost_km", "optimality_gap_pct", "expanded",
                                                    "max_frontier", "runtime_ms"]].mean().round(2)
print("\nMean over the three queries:")
print(summary)
a_star = metrics[metrics.algorithm == "A*"]
print(f"\nA* matches the UCS cost on all queries: {bool(np.allclose(a_star.optimality_gap_pct, 0))}; "
      f"it expands {a_star.expansions_saved_vs_ucs_pct.mean():.1f}% fewer airports on average.")


# ---------------------------------------------------------------------------
# Question 5 - Stress the guarantee (weighted A*, w = 1.5)
# ---------------------------------------------------------------------------
heading("Question 5 - Weighted A* with the inflated heuristic 1.5h")

weighted_rows = []
for q in queries.itertuples():
    h_inflated = heuristic_to(q.goal, weight=1.5)
    result = best_first(graph, q.start, q.goal, h_inflated, mode="astar")
    exact = baseline[q.query_id]
    weighted_rows.append({"query_id": q.query_id, "start": q.start, "goal": q.goal,
                          "ucs_cost_km": exact.path_cost_km, "weighted_cost_km": result.path_cost_km,
                          "gap_pct": 100 * (result.path_cost_km / exact.path_cost_km - 1),
                          "ucs_expanded": exact.expanded, "weighted_expanded": result.expanded,
                          "hops": result.hops, "path": " -> ".join(result.path)})

weighted = pd.DataFrame(weighted_rows)
weighted.to_csv(RESULTS / "weighted_astar.csv", index=False)
print(weighted.drop(columns=["start", "goal"]).round(2).to_string(index=False))
print("\n1.5h can overestimate, so the optimality guarantee is lost. The returned cost changes when")
print("the inflated estimate makes a geographically direct but more expensive route look cheaper than")
print("the true optimum, so the search commits to it before the optimal path is popped.")


# ---------------------------------------------------------------------------
# Question 6 - Save the evidence
# ---------------------------------------------------------------------------
heading("Question 6 - Save the evidence")

pivot = metrics.pivot(index="query_id", columns="algorithm", values="expanded")[["UCS", "GBFS", "A*"]]
ax = pivot.plot.bar(figsize=(8, 4.5), logy=True, rot=0)
ax.set(ylabel="Expanded airports (log scale)", xlabel="Query",
       title="Search effort: UCS vs GBFS vs A* (Haversine heuristic)")
ax.grid(axis="y", alpha=0.25)
for container in ax.containers:
    ax.bar_label(container, fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS / "expanded_nodes.png", dpi=160)
plt.close()

print("Saved:", sorted(p.name for p in RESULTS.iterdir()))
print("\nLab 2 complete.")
