# Source verification

**One source for every number:** the official Formula 1 race results, read from the open database
[F1DB](https://github.com/f1db/f1db) (CC BY 4.0). Snapshot `data/raw/f1db-csv.zip`, downloaded
2026-10-21 08:14 UTC (after the Singapore GP), SHA-256 in `data/raw/f1db-csv.zip.sha256`.

To make sure F1DB reproduces the official results, we compared it with **formula1.com**
(checked 22 October 2026):

| Check | Official page | Result |
|---|---|---|
| Winner of all 17 Grands Prix of 2026 | [formula1.com/en/results/2026/races](https://www.formula1.com/en/results/2026/races) | 17 / 17 identical |
| Singapore 2026: full classification (22 drivers, positions and points) | [race result](https://www.formula1.com/en/results/2026/races/1296/singapore/race-result) | 22 / 22 identical |
| Singapore 2026: starting grid | [starting grid](https://www.formula1.com/en/results/2026/races/1296/singapore/starting-grid) | identical order; the 3 pit-lane starters (Alonso, Stroll, Hadjar) are all placed at the back in our data, as the official page notes |
| Winner of all 21 Grands Prix of 2019 | [formula1.com/en/results/2019/races](https://www.formula1.com/en/results/2019/races) | 21 / 21 identical |
| Brazil 2019: full classification | [race result](https://www.formula1.com/en/results/2019/races/1019/brazil/race-result) | positions 1–16 and points identical; Vettel, Leclerc and Stroll retired late but are classified 17th–19th under FIA rules (≥ 90% distance), which is how our data stores them |

**Derived numbers** (team form, car performance, probabilities) are computed by our code from these
results only, and can be regenerated with `make all`.
