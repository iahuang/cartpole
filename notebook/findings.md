# Findings

## REINFORCE

- REINFORCE is hugely finnicky. I had one model jump to a very pleasant 10sec average time alive. Most of my subsequent models failed to learn, or plateaued around 1sec even after seemingly changing nothing.
- Going to try bumping LR down from 1e-3 to 1e-4. Maybe kinda sorta helped?
- No it didn't help.
- Trying to add entropy regularization to the loss. Not sure if it reliably helps much.
- Okay I got a run where the avg time alive peaked at 2sec and then immediately started cratering again. Clearly something needs to change.
- Realizing my grad norm is enormous, partially as a function of episodic length growing, partially because my reward metric is very large.

## PPO

