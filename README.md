# Deep Hedging with RL

---
### TO-Do List
- ~~learning sequence 코드 작성~~
- ~~config 저장, 로드 코드 작성~~
  - ~~abc class 저장, 로드~~
  - ~~env class 저장, 로드 (매우 큰 크기 때문에 yaml 저장이 효율적이지 않아, class name만 저장하고 로드 시에 env instance를 얻도록 하였음.)~~
- pnl evaluation 코드 작성
  - 검증하기
- pnl graph 표시 코드 작성


- No-Transaction Band Net과 pnl 비교

## Error now
### 1. 학습이 되지 않는 문제
- actor loss 중 mean Q1 만 썼을경우 => actor loss가 줄면서 critic loss 늘고 reward 증가
  (ddpg_220523-0959_3)

- actor loss 중 std Q2-Q1^2 썼을 경우 => actor loss가 0 고정  
  (ddpg_220525-0622_1)

**해결책 제안**
1. 