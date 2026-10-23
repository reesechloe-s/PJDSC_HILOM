"""Independent check: Erlang C vs a brute-force FCFS M/M/c simulation, plus the
sensitivity analysis that answers 'when can a queue/diversion policy help at all?'.
Run:  python validate_queueing.py"""
import heapq
import random

from queueing_theory import effective_arrival_rate, mmc_metrics, servers_needed


def simulate_mmc(lam_per_hr, mean_service_min, c, n=300_000, seed=1, warmup=20_000):
    rng = random.Random(seed)
    lam = lam_per_hr / 60.0
    free = [0.0] * c
    heapq.heapify(free)
    t, waits = 0.0, []
    for i in range(n):
        t += rng.expovariate(lam)
        start = max(t, heapq.heappop(free))
        heapq.heappush(free, start + rng.expovariate(1.0 / mean_service_min))
        if i >= warmup:
            waits.append(start - t)
    return sum(waits) / len(waits), sum(w > 0 for w in waits) / len(waits)


def main():
    print("== 1. Erlang C vs simulation (mean wait, minutes) ==")
    print(f"{'c':>3} {'lam/h':>6} {'S(min)':>6} {'rho':>5} {'theory':>8} {'sim':>8} {'P(wait) th/sim':>16}")
    for c, lam, s in [(1, 1.5, 30), (5, 8, 30), (13, 20, 30), (20, 35, 30), (70, 95, 42)]:
        m = mmc_metrics(lam, s, c)
        sim_w, sim_p = simulate_mmc(lam, s, c)
        print(f"{c:>3} {lam:>6} {s:>6} {m['utilization']:>5.2f} {m['mean_wait_min']:>8.2f} "
              f"{sim_w:>8.2f} {m['p_wait']:>7.3f}/{sim_p:<7.3f}")

    print("\n== 2. Where does a queue appear? c=70 beds ==")
    print("(utilisation by arrival rate and mean bed-occupancy time)")
    print(f"{'lam/h':>6} | " + " ".join(f"S={s:>3}m" for s in (42, 120, 240, 360)))
    for lam in (5, 10, 15, 20, 30, 40, 60):
        row = [mmc_metrics(lam, s, 70)["utilization"] for s in (42, 120, 240, 360)]
        print(f"{lam:>6} | " + " ".join(f"{r:>6.0%}" for r in row))

    print("\n== 3. Does diverting non-urgent patients cut waits? (M/M/c, c=70, S=240 min) ==")
    print(f"{'lam/h':>6} {'divertible':>10} {'wait none':>10} {'wait diverted':>14} {'reduction':>10}")
    for lam in (14, 16, 16.5):
        for frac in (0.10, 0.40):
            base = mmc_metrics(lam, 240, 70)["mean_wait_min"]
            div = mmc_metrics(effective_arrival_rate(lam, frac, 1.0), 240, 70)["mean_wait_min"]
            red = (base - div) / base if base > 0 else float("nan")
            print(f"{lam:>6} {frac:>10.0%} {base:>10.2f} {div:>14.2f} {red:>10.0%}")
    print("\nBeds needed for mean wait <= 45 min at lam=16.5/h, S=240:",
          servers_needed(16.5, 240, 45))


if __name__ == "__main__":
    main()