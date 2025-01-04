### Gradualy introduce noises
- Train model by gradually introducing noises.
- Randomly use different types of noises to avoid model being trained to a specific type of noise.

### Use smoother reward function (more robust to noise)
- Reward can be noisy with control noise
- Use smoother reward
- Trace moving trajectory rather than directly using lat accel?