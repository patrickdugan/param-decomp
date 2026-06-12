# LSPS-0: Loop-Spline Patch Search v0

Final label: `UNDERPOWERED_LSPS0`

## Blocker

LSPS-0 was **not run for evidence** because the loop-spline discriminator precondition was not satisfied.

- Discriminator final label: `STATIC_ENDPOINT_SUFFICIENT`
- Blocking labels: `PROFILE_UNSTABLE`, `STATIC_ENDPOINT_SUFFICIENT`, `UNDERPOWERED_PROBES`, `BLOCKED_NO_RECURSION_TRACE`.

Per the LSPS-0 protocol, when the discriminator label is blocking we write this blocker section and do not claim LSPS-0 evidence. No patch-search comparison between spline-derived and static-derived proposal scores is asserted here.

To run the machinery anyway for descriptive (non-evidential) numbers, re-run with `--override`. The result will still be reported honestly and cannot claim a spline advantage.

No actual Goodfire VPD is claimed. Cached 4D ARC heads were not used. The base model was not updated; no policy, RCPI, or consolidation was built.
