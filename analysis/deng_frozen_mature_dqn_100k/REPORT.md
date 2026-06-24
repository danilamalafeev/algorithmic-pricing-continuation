# Frozen Mature TQL Design Diagnostic

The frozen-mature arm contains 30 runs: 10 independent
10M Q-vs-Q checkpoints and at least
3 DQN initializations per checkpoint.
The checkpoint is the statistical unit.

- frozen-mature DQN profit: 0.300520
- joint-from-scratch DQN profit: 0.288464
- design premium: +0.012055
- hierarchical bootstrap 95% CI: [-0.013116, +0.039539]
- no permutation p-value is reported for this secondary contrast because the
  checkpoint means and joint-learning seeds have different sampling structures
- adaptive-minus-frozen evaluation after training:
  -0.043904
- adaptive-minus-frozen bootstrap 95% CI:
  [-0.068590, -0.021540]
- adaptive-minus-frozen exact two-sided sign-test p-value:
  0.001953
- frozen DQN-minus-incumbent profit:
  +0.065779
- adaptive DQN-minus-incumbent profit:
  +0.007274
- adaptive-minus-frozen entrant advantage:
  -0.058504
- all mature Q-tables unchanged during DQN training:
  True
- all mature Victim clocks unchanged during DQN training:
  True

This comparison identifies the full design difference between entry against a
frozen mature incumbent and joint learning from scratch. It does not isolate
maturity from freezing.
