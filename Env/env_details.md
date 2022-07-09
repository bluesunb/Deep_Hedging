# Implementation
## Reward

### time evolution


### P&L
- reward
    $$
    \begin{aligned}
    R_t=P_{t+1} + h_{t+1}\Delta S_{t+1}-kS_t\|\Delta h_{t+1}\|
    \end{aligned}
    $$

    if $t<T-1$, $\rightarrow P_{t+1} = \Delta V_{t+1}$  
    else $\rightarrow P_{t+1} = -\text{Payoff}(V_{t+1}, K) - kS_{t+1} h_{t+1} $  
    (여기서 payoff가 음수인 것은 call option seller입장에서 option 행사는 손해이기 때문)

| $t$ | 0 | 0 | 1 | 1 | 2(maturity) | 2(maturity) |
| --- | --- | --- | --- | --- | --- | --- |
| $S_t$ | $S_0$ |  | $S_1$ | | $S_2$ |  |
| $V_t$ | $V_0$ |  | $V_1$ | | $V_2$ | disposal |
| $h_t$ | $h_0$ | $h_1$ | | $h_2$ || 0 |
| $R_t$ |  |  | $R_0$ |  | $R_1$ | $-P(V_2,K)-kh_2S_2$ |
