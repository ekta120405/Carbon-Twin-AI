# Layer 2 Model Comparison (R^2 by regime, test_clean)


## plant_1 (CR=1.0278)

| regime | physics-only | PI-GBQR | Ridge | LSTM |
|---|---|---|---|---|
| overall | 0.835 | 0.866 | 0.790 | 0.870 |
| clear (Kt>=0.75) | 0.796 | 0.843 | 0.705 | 0.774 |
| partly_cloudy (0.4<=Kt<0.75) | 0.213 | 0.355 | 0.513 | 0.511 |
| overcast (Kt<0.4) | 0.658 | 0.658 | -5.834 | -0.057 |

## plant_2 (CR=0.8973)

| regime | physics-only | PI-GBQR | Ridge | LSTM |
|---|---|---|---|---|
| overall | 0.751 | 0.681 | -0.026 | -0.728 |
| clear (Kt>=0.75) | 0.349 | 0.307 | -1.066 | -0.501 |
| partly_cloudy (0.4<=Kt<0.75) | 0.457 | 0.140 | -1.942 | -1.503 |
| overcast (Kt<0.4) | 0.749 | 0.749 | -0.039 | -108.600 |
