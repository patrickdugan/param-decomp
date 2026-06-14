# Powered Intellect-3 VPD-Gated Adapter Report

Model: `artifacts\powered_int3_logic\model`
Train rows: `64`; validation rows: `32`; eval rows: `64`; steps: `0`
Selection source: `vpd`
Selection method: `validated`

The selected arm applies and optionally trains scalar coefficients on chosen recursive-trunk component directions. The control arm uses the same number of random recursive-trunk component directions from the same source.

## Results

| arm | components | eval loss | bucket acc | bucket drift |
|---|---:|---:|---:|---:|
| base_frozen | 0 | 1.001876 | 0.515625 | 0.0 |
| vpd_validated | 4 | 0.999113 | 0.515625 | 0.030397 |
| random_trunk_control | 4 | 1.001876 | 0.515625 | 0.0 |

## Validation Selection

| component | init coeff | validation score | validation loss | validation drift | alive | damage |
|---|---:|---:|---:|---:|---:|---:|
| `networks.0.layers.0.1.to_out:c000` | 0.1 | 0.896682 | 0.895692 | 0.019792 | 0.764282 | 0.302251 |
| `networks.0.layers.0.1.to_out:c013` | -0.1 | 0.896977 | 0.896374 | 0.012056 | 0.127441 | 0.107139 |
| `networks.0.layers.0.1.to_out:c017` | -0.1 | 0.897168 | 0.896588 | 0.011596 | 0.250854 | 0.129122 |
| `networks.0.layers.0.1.to_out:c003` | -0.1 | 0.897443 | 0.896932 | 0.01022 | 0.290405 | 0.140283 |

## Selected Components

### base_frozen


### vpd_validated

- `networks.0.layers.0.1.to_out:c000` damage=0.302251 scale=1.501069 alive=0.764282 init=0.1
- `networks.0.layers.0.1.to_out:c013` damage=0.107139 scale=1.293743 alive=0.127441 init=-0.1
- `networks.0.layers.0.1.to_out:c017` damage=0.129122 scale=1.37157 alive=0.250854 init=-0.1
- `networks.0.layers.0.1.to_out:c003` damage=0.140283 scale=1.503469 alive=0.290405 init=-0.1

### random_trunk_control

- `networks.0.layers.0.1.to_out:c008` damage=0.123793 scale=1.402293 alive=0.376648 init=0.0
- `networks.0.layers.0.1.to_out:c016` damage=0.15847 scale=1.450827 alive=0.399719 init=0.0
- `networks.0.layers.0.1.to_out:c015` damage=0.167758 scale=1.50777 alive=0.333984 init=0.0
- `networks.0.layers.0.1.to_out:c028` damage=0.082291 scale=1.261111 alive=0.299194 init=0.0
