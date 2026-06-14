# Powered Intellect-3 VPD-Gated Adapter Report

Model: `artifacts\powered_int3_logic\model`
Train rows: `64`; eval rows: `64`; steps: `12`
Selection source: `vpd`

The selected arm trains scalar coefficients on chosen recursive-trunk component directions. The control arm trains the same number of random recursive-trunk component directions from the same source.

## Results

| arm | components | eval loss | bucket acc | bucket drift |
|---|---:|---:|---:|---:|
| base_frozen | 0 | 1.001876 | 0.515625 | 0.0 |
| vpd_selected | 4 | 1.016567 | 0.5 | 0.060441 |
| random_trunk_control | 4 | 0.991228 | 0.515625 | 0.070457 |

## Selected Components

### base_frozen


### vpd_selected

- `networks.0.layers.0.1.to_out:c000` damage=0.302251 scale=1.501069
- `networks.0.layers.0.1.to_out:c007` damage=0.247017 scale=1.434686
- `networks.0.layers.0.1.to_out:c026` damage=0.228342 scale=1.295545
- `networks.0.layers.0.1.to_out:c021` damage=0.22149 scale=1.281061

### random_trunk_control

- `networks.0.layers.0.1.to_out:c018` damage=0.105657 scale=1.237499
- `networks.0.layers.0.1.to_out:c012` damage=0.092241 scale=1.424583
- `networks.0.layers.0.1.to_out:c031` damage=0.094245 scale=1.442149
- `networks.0.layers.0.1.to_out:c016` damage=0.15847 scale=1.450827
