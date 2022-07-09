# Deep Hedging with RL

---
### TO-Do List
- ~~learning sequence 코드 작성~~
- ~~config 저장, 로드 코드 작성~~
  - ~~abc class 저장, 로드~~
  - ~~env class 저장, 로드 (매우 큰 크기 때문에 yaml 저장이 효율적이지 않아, class name만 저장하고 로드 시에 env instance를 얻도록 하였음.)~~
- ~~pnl evaluation 코드 작성~~
  - 검증하기
- ~~pnl graph 표시 코드 작성~~

- ~~DDPG 성능 확인~~

- ~~No-Transaction Band Net과 pnl 비교~~

- **IMPORTANT**
  1. eval env를 따로 구현할 것. (pnl_eval, delta eval등이 자연스럽게 도출되는 step + 다른 seed)
  2. 파일 트리 재구성할 것. stable_baseline3 와 같이 ddpg/policies.py, ddpg.py + config.py
  3. 