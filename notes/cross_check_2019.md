# 2019 parameter cross-check

Updated 2026-10-23. Done for the three dry 2019 validation races using the v4 extractor (`results/v4_frozen/extract_race.py`, unchanged) run offline on the 2019 data in `fastf1_cache`. All three pass `Races.Condition == DRY`, `Included == True`, `LoadStatus == OK`. No TUM parameters were changed.

Full table: [evidence/cross_check_2019/comparison.csv](evidence/cross_check_2019/comparison.csv). Per-race fits and stops are next to it.

| Race | Quantity | FastF1 (ours) | TUM `.ini` |
|---|---|---:|---:|
| Spielberg (R9) | Green pit loss, s | 21.66 | 16.83 |
| Le Castellet (R8) | Green pit loss, s | 24.96 | 24.04 |
| Budapest (R12) | Green pit loss, s | 21.29 | 20.52 |
| Spielberg (R9) | Fuel + track trend, s/lap faster | 0.036 | 0.029 (fuel only) |
| Le Castellet (R8) | Fuel + track trend, s/lap faster | 0.072 | 0.073 (fuel only) |
| Budapest (R12) | Fuel + track trend, s/lap faster | 0.050 | 0.044 (fuel only) |
| Spielberg (R9) | Hard k1, s/lap | 0.0308 | 0.0258 |
| Le Castellet (R8) | Hard k1, s/lap | 0.0195 | 0.0201 |
| Budapest (R12) | Hard k1, s/lap | 0.0380 | 0.0308 |
| Spielberg (R9) | Medium / soft k1, s/lap | −0.005 / −0.007 | 0.010 / 0.010 |
| Le Castellet (R8) | Medium / soft k1, s/lap | −0.019 / not reliable | 0.010 / 0.010 |
| Budapest (R12) | Medium / soft k1, s/lap | 0.033 / 0.037 | 0.019 / 0.012 |

TUM k1 values are medians over 20 drivers. TUM pit loss = in-lap transit + out-lap transit + minimum standstill + median team addition (its 1.0 s cold-tyre penalty is separate). Our pit loss = median of (in-lap + out-lap − 2 × reference clean lap) over green-flag stops.

## Findings

- **Hard-tyre degradation and the fuel/track trend agree** at all three tracks (k1 within 0.007 s/lap, trend within 0.007 s/lap). The pipeline reproduces TUM where the data can identify the parameter.
- **Medium and soft slopes are not identifiable at Spielberg and Le Castellet.** Our race-level fit goes negative (too little spread in stint start laps); TUM sits at its 0.01 s/lap floor (12/20 and 18/20 drivers at Spielberg). Neither number should be trusted there.
- **Spielberg pit loss differs by 4.8 s** (ours 21.7 s vs TUM 16.8 s; 3.8 s even after adding TUM's 1.0 s cold-tyre penalty). The other two tracks agree within 1 s. A pit loss that is too low makes extra or early stops look cheap, so this matters directly for pit-timing labels. Not yet explained.
- **MAG's lap-13 "stop" in TUM's file is a drive-through penalty** (race control: out of position at the start). TUM encodes it as a stop that refits the same 2-lap-old set. Our stop detector classifies it correctly as not a stop.
