# Powered Intellect-3 VPD-Gated Adapter Report

Model: `artifacts\powered_int3_logic\model`
Train rows: `64`; eval rows: `64`; steps: `12`

The selected arm trains scalar coefficients on high-static-damage recursive-trunk component directions. The control arm trains the same number of random recursive-trunk component directions.

## Results

| arm | components | eval loss | bucket acc | bucket drift |
|---|---:|---:|---:|---:|
| base_frozen | 0 | 1.001876 | 0.515625 | 0.0 |
| vpd_static_selected | 4 | 0.972768 | 0.53125 | 0.24266 |
| random_trunk_control | 4 | 1.0022 | 0.515625 | 0.033372 |

## Selected Components

### base_frozen


### vpd_static_selected

- `networks.0.layers.0.1.to_out.weight::svd000` damage=12.0 scale=1.501414
- `networks.0.layers.0.1.to_v.weight::svd000` damage=11.0 scale=2.423624
- `networks.0.layers.2.1.to_out.weight::svd000` damage=7.0 scale=1.111977
- `networks.0.layers.2.1.to_v.weight::svd000` damage=6.0 scale=1.81578

### random_trunk_control

- `networks.0.layers.3.1.ff.0.0.weight::svd000` damage=1.0 scale=1.7291
- `networks.0.layers.9.1.ff.0.0.weight::svd000` damage=1.0 scale=1.984274
- `networks.0.layers.6.1.to_out.weight::svd000` damage=1.0 scale=0.898727
- `networks.0.layers.1.1.ff.2.weight::svd000` damage=3.0 scale=0.906225
