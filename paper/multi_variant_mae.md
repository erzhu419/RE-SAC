# Multi-variant Q-MAE (review #7)

Reviewer concern: Oracle MAE picks the best head per state — not deployable.
Below: Oracle (oracle upper bound) | Mean (vanilla ensemble) | LCB (actor target with β=-2) | Min (clipped-twin)

## Hopper-v2

| Algo | Oracle | Mean-head | LCB-head | Min-head |
|---|---:|---:|---:|---:|
| RE-SAC v1 | 13.0 | 13.9 | 13.8 | 13.9 |
| BAC | 13.3 | 13.8 | 13.9 | 13.9 |
| SAC | 15.5 | 16.0 | 15.9 | 16.0 |
| DSAC | 16.6 | 17.0 | 17.0 | 17.0 |
| TD3 | 6.8 | 7.5 | 7.7 | 7.5 |

## Walker2d-v2

| Algo | Oracle | Mean-head | LCB-head | Min-head |
|---|---:|---:|---:|---:|
| RE-SAC v1 | 10.7 | 11.3 | 11.4 | 11.3 |
| BAC | 7.5 | 7.9 | 8.0 | 7.9 |
| SAC | 10.1 | 10.5 | 10.5 | 10.5 |
| DSAC | 10.6 | 11.0 | 11.0 | 11.0 |
| TD3 | 10.7 | 11.5 | 11.7 | 11.6 |

## HalfCheetah-v2

| Algo | Oracle | Mean-head | LCB-head | Min-head |
|---|---:|---:|---:|---:|
| RE-SAC v1 | 82.1 | 88.1 | 89.4 | 89.0 |
| BAC | 42.2 | 43.8 | 44.2 | 43.9 |
| SAC | 59.0 | 60.7 | 61.3 | 60.9 |
| DSAC | 45.9 | 47.1 | 47.7 | 47.4 |
| TD3 | 47.6 | 51.0 | 52.4 | 51.6 |

## Ant-v2

| Algo | Oracle | Mean-head | LCB-head | Min-head |
|---|---:|---:|---:|---:|
| RE-SAC v1 | 74.3 | 83.2 | 85.5 | 85.1 |
| BAC | 86.5 | 89.1 | 90.4 | 89.7 |
| SAC | 81.7 | 85.0 | 86.5 | 85.7 |
| DSAC | 107.7 | 110.5 | 111.6 | 111.0 |
| TD3 | 61.4 | 65.1 | 67.8 | 66.3 |

